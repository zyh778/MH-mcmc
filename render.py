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

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import time
import json
from datetime import datetime

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    # Track render time
    start_time = time.time()
    render_times = []

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        frame_start_time = time.time()
        rendering = render(view, gaussians, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

        frame_end_time = time.time()
        frame_render_time = frame_end_time - frame_start_time
        render_times.append(frame_render_time)

    total_render_time = time.time() - start_time
    avg_frame_time = sum(render_times) / len(render_times) if render_times else 0
    fps = len(views) / total_render_time if total_render_time > 0 else 0

    # Get image resolution
    if views:
        img_height, img_width = views[0].image_height, views[0].image_width
        resolution = f"{img_width}x{img_height}"
    else:
        resolution = "N/A"

    return {
        "total_render_time": total_render_time,
        "avg_frame_time": avg_frame_time,
        "fps": fps,
        "image_count": len(views),
        "resolution": resolution
    }

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_metrics = {}

        if not skip_train:
            train_metrics = render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)
            render_metrics["train"] = train_metrics

        if not skip_test:
            test_metrics = render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)
            render_metrics["test"] = test_metrics

        # Save render metrics to unified file
        save_render_metrics(dataset.model_path, iteration, render_metrics, gaussians)

def save_render_metrics(model_path, iteration, render_metrics, gaussians):
    """Save render metrics to unified render.json file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    render_file = "render.json"

    # Get model info
    num_gaussians = len(gaussians._xyz) if hasattr(gaussians, '_xyz') else 0

    # Calculate total pixels
    total_pixels = 0
    for set_name, metrics in render_metrics.items():
        if 'resolution' in metrics and metrics['resolution'] != 'N/A':
            width, height = map(int, metrics['resolution'].split('x'))
            total_pixels += width * height * metrics['image_count']

    # Load existing render data if file exists and is not empty
    existing_data = {"timestamp": timestamp, "scenes": {}}
    if os.path.exists(render_file):
        try:
            with open(render_file, 'r') as fp:
                file_content = fp.read().strip()
                if file_content:
                    existing_data = json.load(fp)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load existing render.json file: {e}")
            existing_data = {"timestamp": timestamp, "scenes": {}}

    # Update timestamp
    existing_data["timestamp"] = timestamp

    # Initialize scene path if not exists
    scene_key = f"{model_path}/ours_{iteration}"
    if scene_key not in existing_data["scenes"]:
        existing_data["scenes"][scene_key] = {}

    # Update with render metrics
    for set_name, metrics in render_metrics.items():
        existing_data["scenes"][scene_key][set_name] = {
            "total_render_time": metrics["total_render_time"],
            "avg_frame_time": metrics["avg_frame_time"],
            "fps": metrics["fps"],
            "resolution": metrics["resolution"],
            "image_count": metrics["image_count"],
            "num_gaussians": num_gaussians,
            "total_pixels": total_pixels
        }

    # Save to file
    with open(render_file, 'w') as fp:
        json.dump(existing_data, fp, indent=True)

    print(f"Render metrics saved to: {render_file}")

    # Print summary
    for set_name, metrics in render_metrics.items():
        print(f"\n{set_name.capitalize()} Render Metrics:")
        print(f"  Total Render Time: {metrics['total_render_time']:.2f} seconds")
        print(f"  Average Frame Time: {metrics['avg_frame_time']:.4f} seconds")
        print(f"  FPS: {metrics['fps']:.2f}")
        print(f"  Resolution: {metrics['resolution']}")
        print(f"  Image Count: {metrics['image_count']}")
        print(f"  Gaussian Count: {num_gaussians}")
        print(f"  Total Pixels: {total_pixels}")

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)