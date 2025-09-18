#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import json
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene.gaussian_model import build_scaling_rotation
import time
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    if dataset.cap_max == -1:
        print("Please specify the maximum number of Gaussians using --cap_max.")
        exit()

    # Training tracking
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    # Initialize loss tracking
    loss_history = {
        'iterations': [],
        'total_loss': [],
        'l1_loss': [],
        'ssim_loss': [],
        'psnr': [],
        'gaussian_count': [],
        'iteration_times': []
    }

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):
        iter_wall_time_start = time.time()  # Track wall time for each iteration        
        # if network_gui.conn == None:
        #     network_gui.try_connect()
        # while network_gui.conn != None:
        #     try:
        #         net_image_bytes = None
        #         custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
        #         if custom_cam != None:
        #             net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
        #             net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
        #         network_gui.send(net_image_bytes, dataset.source_path)
        #         if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
        #             break
        #     except Exception as e:
        #         network_gui.conn = None

        iter_start.record()

        xyz_lr = gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        else:
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image = render_pkg["render"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        ssim_loss = (1.0 - ssim(image, gt_image))
        current_psnr = psnr(image, gt_image).mean().item()

        # Adaptive SSIM weight based on training progress
        ssim_weight = opt.lambda_dssim 
        # Primary loss function: L1 + SSIM combination
        # Conservative SSIM weighting to maintain PSNR performance
        loss = (1.0 - ssim_weight) * Ll1 + ssim_weight * ssim_loss
        # Enhanced regularization with adaptive weights
        # Sparsity regularization to encourage compact model
        sparsity_reg = args.opacity_reg * torch.abs(gaussians.get_opacity).mean()
        # Scale diversity regularization to prevent over-concentration
        scale_diversity = args.scale_reg * torch.abs(gaussians.get_scaling).mean()
        # Combined loss with conservative regularization
        # Primary focus on reconstruction quality (L1 + SSIM)
        # Secondary focus on model efficiency (regularization terms)
        loss = loss + sparsity_reg + scale_diversity

        loss.backward()

        iter_end.record()
        # Store current loss for MH acceptance
        current_loss = loss.detach().clone()

        # Track iteration metrics
        iter_wall_time_end = time.time()
        iter_time = iter_wall_time_end - iter_wall_time_start

        # Record loss history every 10 iterations to reduce memory usage
        if iteration % 10 == 0:
            loss_history['iterations'].append(iteration)
            loss_history['total_loss'].append(loss.item())
            loss_history['l1_loss'].append(Ll1.item())
            loss_history['ssim_loss'].append(ssim_loss.item())
            loss_history['psnr'].append(current_psnr)
            loss_history['gaussian_count'].append(gaussians.get_xyz.shape[0])
            loss_history['iteration_times'].append(iter_time)

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                print("Output folder: {}".format(args.model_path))
                scene.save(iteration)

            if iteration < opt.densify_until_iter and iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                dead_mask = (gaussians.get_opacity <= 0.005).squeeze(-1)
                gaussians.relocate_gs(dead_mask=dead_mask)
                gaussians.add_new_gs(cap_max=args.cap_max)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

                L = build_scaling_rotation(gaussians.get_scaling, gaussians.get_rotation)
                actual_covariance = L @ L.transpose(1, 2)

                def op_sigmoid(x, k=100, x0=0.995):
                    return 1 / (1 + torch.exp(-k * (x - x0)))
                
                noise = torch.randn_like(gaussians._xyz) * (op_sigmoid(1- gaussians.get_opacity))*args.noise_lr*xyz_lr
                noise = torch.bmm(actual_covariance, noise.unsqueeze(-1)).squeeze(-1)
                gaussians._xyz.add_(noise)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

    return loss_history

def prepare_output_and_logger(args):    
    args.model_path = os.path.join("./output/", args.model_path)
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config

def save_training_metrics(args, loss_history, total_training_time, total_iterations):
    """Save training metrics and generate loss curves"""
    # Create output directories
    output_dir = "output/results"
    # loss_dir = os.path.join(output_dir, "loss")
    loss_dir = os.path.join(output_dir, args.model_path)
    os.makedirs(loss_dir, exist_ok=True)


    # Save loss curves
    save_loss_curves(loss_history, loss_dir, total_iterations)

    # Save training metrics to JSON
    training_metrics = {
        # "timestamp": datetime.now().strftime("%H%M%S%Y%m%d"),
        "model_id": args.model_path,
        "source_path": args.source_path,
        "total_training_time": "{:.2f}s".format(total_training_time),
        "total_iterations": total_iterations,
        "avg_iteration_time": "{:.2f}ms".format(np.mean(loss_history['iteration_times']) * 1000 if loss_history['iteration_times'] else 0),
    }

    # Save to unified training metrics file
    training_file = os.path.join(loss_dir, "training_metrics.json")
    if os.path.exists(training_file):
        try:
            with open(training_file, 'r') as f:
                existing_data = json.load(f)
                if isinstance(existing_data, list):
                    existing_data.append(training_metrics)
                else:
                    existing_data = [existing_data, training_metrics]
        except (json.JSONDecodeError, IOError):
            existing_data = [training_metrics]
    else:
        existing_data = [training_metrics]

    with open(training_file, 'w') as f:
        json.dump(existing_data, f, indent=2)

    print(f"Training metrics saved to: {training_file}")


def save_loss_curves(loss_history, loss_dir, total_iterations):
    """Generate and save loss curve plots"""
    if loss_history is None or not loss_history['iterations']:
        print("Warning: No loss history data available for plotting")
        return

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Training Loss Curves', fontsize=16)

    # Plot 1: Total Loss
    axes[0, 0].plot(loss_history['iterations'], loss_history['total_loss'], 'b-', linewidth=2)
    axes[0, 0].set_title('Total Training Loss')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: L1 and SSIM Loss
    axes[0, 1].plot(loss_history['iterations'], loss_history['l1_loss'], 'r-', label='L1 Loss', linewidth=2)
    axes[0, 1].plot(loss_history['iterations'], loss_history['ssim_loss'], 'g-', label='SSIM Loss', linewidth=2)
    axes[0, 1].set_title('Component Losses')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: PSNR
    axes[1, 0].plot(loss_history['iterations'], loss_history['psnr'], 'm-', linewidth=2)
    axes[1, 0].set_title('PSNR')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('PSNR (dB)')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Gaussian Count
    axes[1, 1].plot(loss_history['iterations'], loss_history['gaussian_count'], 'c-', linewidth=2)
    axes[1, 1].set_title('Gaussian Count')
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Number of Gaussians')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the plot
    plot_filename = os.path.join(loss_dir, f"curves_{total_iterations}.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Loss curves saved to: {plot_filename}")


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    
    if args.config is not None:
        # Load the configuration file
        config = load_config(args.config)
        # Set the configuration parameters on args, if they are not already set by command line arguments
        for key, value in config.items():
            setattr(args, key, value)

    args.save_iterations.append(args.iterations)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.model_path = f"{timestamp}"

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # Create output directory if it doesn't exist
    os.makedirs("output/results", exist_ok=True)

    # Start training and track time
    training_start_time = time.time()
    loss_history = training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)
    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time

    # Save training metrics and loss curves
    save_training_metrics(args, loss_history, total_training_time, op.extract(args).iterations)
    print("Output:" + os.path.join("./output/",args.model_path))
    print("\nTraining complete.")