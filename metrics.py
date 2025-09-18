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

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
from datetime import datetime


def get_ply_info(ply_path):
    if not os.path.exists(ply_path):
        return 0, 0

    file_size = os.path.getsize(ply_path)

    num_points = 0
    try:
        with open(ply_path, 'r') as f:
            for line in f:
                if line.startswith("element vertex"):
                    num_points = int(line.split()[-1])
                    break
    except UnicodeDecodeError:
        # Handle binary ply files
        with open(ply_path, 'rb') as f:
            # Simple binary PLY parsing can be complex, this is a basic example
            # It might be better to use a library if binary PLY files are common
            # For now, we will assume the header is ASCII and then the data is binary
            header = b""
            while b"end_header" not in header:
                header += f.readline()

            header_str = header.decode('ascii')
            for line in header_str.split('\n'):
                if line.startswith("element vertex"):
                    num_points = int(line.split()[-1])
                    break

    return num_points, file_size


def calculate_image_metrics(renders):
    """Calculate image metrics"""
    image_metrics = {}

    if renders and len(renders) > 0:
        # Get image dimensions
        img_height, img_width = renders[0].shape[2], renders[0].shape[3]
        total_pixels = img_height * img_width * len(renders)

        image_metrics['total_pixels'] = total_pixels
        image_metrics['image_count'] = len(renders)
        image_metrics['resolution'] = f"{img_width}x{img_height}"
    else:
        image_metrics['total_pixels'] = 0
        image_metrics['image_count'] = 0
        image_metrics['resolution'] = "N/A"

    return image_metrics


def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
    return renders, gts


def evaluate(model_paths):
    full_dict = {}

    # Create unified metrics file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_file = "metric.json"

    # Load existing metrics if file exists
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as fp:
            existing_metrics = json.load(fp)
    else:
        existing_metrics = {"timestamp": timestamp, "scenes": {}}

    # Update timestamp
    existing_metrics["timestamp"] = timestamp

    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}

            # Initialize scene in existing metrics if not exists
            if scene_dir not in existing_metrics["scenes"]:
                existing_metrics["scenes"][scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}

                # Initialize method in existing metrics
                existing_metrics["scenes"][scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir / "gt"
                renders_dir = method_dir / "renders"
                renders, gts = readImages(renders_dir, gt_dir)

                ssims = []
                psnrs = []
                lpipss = []

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    ssims.append(ssim(renders[idx], gts[idx]))
                    psnrs.append(psnr(renders[idx], gts[idx]))
                    lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))

                print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))

                # Get Gaussian point count and model size
                iteration = int(method.split('_')[-1])
                ply_path = Path(scene_dir) / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
                num_points, model_size = get_ply_info(ply_path)

                if num_points > 0:
                    print(f"  Number of Gaussians: {num_points}")
                if model_size > 0:
                    print(f"  Model Size (MB): {model_size / 1024 / 1024:.2f}")

                # Calculate image metrics
                image_metrics = calculate_image_metrics(renders)
                print(f"  Resolution: {image_metrics['resolution']}")
                print("")

                # Update full_dict with all metrics
                full_dict[scene_dir][method].update({
                    "SSIM": torch.tensor(ssims).mean().item(),
                    "PSNR": torch.tensor(psnrs).mean().item(),
                    "LPIPS": torch.tensor(lpipss).mean().item(),
                    "num_gaussians": num_points,
                    "model_size_MB": model_size / 1024 / 1024,
                    "resolution": image_metrics['resolution'],
                    "image_count": image_metrics['image_count'],
                    "total_pixels": image_metrics['total_pixels']
                })

                # Update existing metrics with new data
                existing_metrics["scenes"][scene_dir][method].update({
                    "SSIM": torch.tensor(ssims).mean().item(),
                    "PSNR": torch.tensor(psnrs).mean().item(),
                    "LPIPS": torch.tensor(lpipss).mean().item(),
                    "num_gaussians": num_points,
                    "model_size_MB": model_size / 1024 / 1024,
                    "resolution": image_metrics['resolution'],
                    "image_count": image_metrics['image_count'],
                    "total_pixels": image_metrics['total_pixels']
                })

            # Save original format file
            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
        except Exception as e:
            print("Unable to compute metrics for model", scene_dir)
            print("Error:", e)

    # Save all metrics to unified file
    with open(metrics_file, 'w') as fp:
        json.dump(existing_metrics, fp, indent=True)
    # print(f"All metrics saved to: {metrics_file}")

    return metrics_file


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Enhanced metrics evaluation with Gaussian points, model size, and render metrics")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--output_dir', '-o', type=str, default=".", help="Output directory for metrics files")
    parser.add_argument('--verbose', '-v', action='store_true', help="Verbose output")
    args = parser.parse_args()

    # Change to output directory if specified
    if args.output_dir != ".":
        os.makedirs(args.output_dir, exist_ok=True)
        os.chdir(args.output_dir)

    print(f"Enhanced metrics evaluation for {len(args.model_paths)} model(s)")
    if args.verbose:
        print(f"Output directory: {args.output_dir}")

    # Run evaluation
    metrics_file = evaluate(args.model_paths)

    print(f"\nEvaluation complete. All metrics saved to: {metrics_file}")