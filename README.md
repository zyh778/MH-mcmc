# MH-mcmc

## Train
```bash
python train.py -s /data2/tandt/train  --eval   --scale_reg  0.01 --opacity_reg  0.01 --noise_lr  5e5 --cap_max 1100000
```

## Render
```bash
python render.py -m <path to trained model>
```

## Metrics
```bash
python metrics.py -m <path to trained model>
```

## Running

To run the optimizer, simply use:
```bash
python train.py -s <path to COLMAP or NeRF Synthetic dataset>
```

### Command Line Arguments for train.py

| Argument | Description |
|----------|-------------|
| `--source_path` / `-s` | Path to the source directory containing a COLMAP or Synthetic NeRF data set. |
| `--model_path` / `-m` | Path where the trained model should be stored (output/<random> by default). |
| `--images` / `-i` | Alternative subdirectory for COLMAP images (images by default). |
| `--eval` | Add this flag to use a MipNeRF360-style training/test split for evaluation. |
| `--resolution` / `-r` | Specifies resolution of the loaded images before training. If provided 1, 2, 4 or 8, uses original, 1/2, 1/4 or 1/8 resolution, respectively. For all other values, rescales the width to the given number while maintaining image aspect. If not set and input image width exceeds 1.6K pixels, inputs are automatically rescaled to this target. |
| `--data_device` | Specifies where to put the source image data, cuda by default, recommended to use cpu if training on large/high-resolution dataset, will reduce VRAM consumption, but slightly slow down training. Thanks to HrsPythonix. |
| `--white_background` / `-w` | Add this flag to use white background instead of black (default), e.g., for evaluation of NeRF Synthetic dataset. |
| `--sh_degree` | Order of spherical harmonics to be used (no larger than 3). 3 by default. |
| `--convert_SHs_python` | Flag to make pipeline compute forward and backward of SHs with PyTorch instead of ours. |
| `--convert_cov3D_python` | Flag to make pipeline compute forward and backward of the 3D covariance with PyTorch instead of ours. |
| `--debug` | Enables debug mode if you experience erros. If the rasterizer fails, a dump file is created that you may forward to us in an issue so we can take a look. |
| `--debug_from` | Debugging is slow. You may specify an iteration (starting from 0) after which the above debugging becomes active. |
| `--iterations` | Number of total iterations to train for, 30_000 by default. |
| `--ip` | IP to start GUI server on, 127.0.0.1 by default. |
| `--port` | Port to use for GUI server, 6009 by default. |
| `--test_iterations` | Space-separated iterations at which the training script computes L1 and PSNR over test set, 7000 30000 by default. |
| `--save_iterations` | Space-separated iterations at which the training script saves the Gaussian model, 7000 30000 <iterations> by default. |
| `--checkpoint_iterations` | Space-separated iterations at which to store a checkpoint for continuing later, saved in the model directory. |
| `--start_checkpoint` | Path to a saved checkpoint to continue training from. |
| `--quiet` | Flag to omit any text written to standard out pipe. |
| `--feature_lr` | Spherical harmonics features learning rate, 0.0025 by default. |
| `--opacity_lr` | Opacity learning rate, 0.05 by default. |
| `--scaling_lr` | Scaling learning rate, 0.005 by default. |
| `--rotation_lr` | Rotation learning rate, 0.001 by default. |
| `--position_lr_max_steps` | Number of steps (from 0) where position learning rate goes from initial to final. 30_000 by default. |
| `--position_lr_init` | Initial 3D position learning rate, 0.00016 by default. |
| `--position_lr_final` | Final 3D position learning rate, 0.0000016 by default. |
| `--position_lr_delay_mult` | Position learning rate multiplier (cf. Plenoxels), 0.01 by default. |
| `--densify_from_iter` | Iteration where densification starts, 500 by default. |
| `--densify_until_iter` | Iteration where densification stops, 15_000 by default. |
| `--densify_grad_threshold` | Limit that decides if points should be densified based on 2D position gradient, 0.0002 by default. |
| `--densification_interval` | How frequently to densify, 100 (every 100 iterations) by default. |
| `--opacity_reset_interval` | How frequently to reset opacity, 3_000 by default. |
| `--lambda_dssim` | Influence of SSIM on total loss from 0 to 1, 0.2 by default. |
| `--percent_dense` | Percentage of scene extent (0--1) a point must exceed to be forcibly densified, 0.01 by default. |

Note that similar to MipNeRF360, we target images at resolutions in the 1-1.6K pixel range. For convenience, arbitrary-size inputs can be passed and will be automatically resized if their width exceeds 1600 pixels. We recommend to keep this behavior, but you may force training to use your higher-resolution images by setting -r 1.

The MipNeRF360 scenes are hosted by the paper authors [here](link). You can find our SfM data sets for Tanks&Temples and Deep Blending [here](link). If you do not provide an output model directory (-m), trained models are written to folders with randomized unique names inside the output directory. At this point, the trained models may be viewed with the real-time viewer (see further below).

## Evaluation

By default, the trained models use all available images in the dataset. To train them while withholding a test set for evaluation, use the `--eval` flag. This way, you can render training/test sets and produce error metrics as follows:

```bash
python train.py -s <path to COLMAP or NeRF Synthetic dataset> --eval # Train with train/test split
python render.py -m <path to trained model> # Generate renderings
python metrics.py -m <path to trained model> # Compute error metrics on renderings
```

If you want to evaluate our pre-trained models, you will have to download the corresponding source data sets and indicate their location to render.py with an additional --source_path/-s flag. Note: The pre-trained models were created with the release codebase. This code base has been cleaned up and includes bugfixes, hence the metrics you get from evaluating them will differ from those in the paper.

```bash
python render.py -m <path to pre-trained model> -s <path to COLMAP dataset>
python metrics.py -m <path to pre-trained model>
```

### Command Line Arguments for render.py

| Argument | Description |
|----------|-------------|
| `--model_path` / `-m` | Path to the trained model directory you want to create renderings for. |
| `--skip_train` | Flag to skip rendering the training set. |
| `--skip_test` | Flag to skip rendering the test set. |
| `--quiet` | Flag to omit any text written to standard out pipe. |

The below parameters will be read automatically from the model path, based on what was used for training. However, you may override them by providing them explicitly on the command line.

| Argument | Description |
|----------|-------------|
| `--source_path` / `-s` | Path to the source directory containing a COLMAP or Synthetic NeRF data set. |
| `--images` / `-i` | Alternative subdirectory for COLMAP images (images by default). |
| `--eval` | Add this flag to use a MipNeRF360-style training/test split for evaluation. |
| `--resolution` / `-r` | Changes the resolution of the loaded images before training. If provided 1, 2, 4 or 8, uses original, 1/2, 1/4 or 1/8 resolution, respectively. For all other values, rescales the width to the given number while maintaining image aspect. 1 by default. |
| `--white_background` / `-w` | Add this flag to use white background instead of black (default), e.g., for evaluation of NeRF Synthetic dataset. |
| `--convert_SHs_python` | Flag to make pipeline render with computed SHs from PyTorch instead of ours. |
| `--convert_cov3D_python` | Flag to make pipeline render with computed 3D covariance from PyTorch instead of ours. |

### Command Line Arguments for metrics.py

| Argument | Description |
|----------|-------------|
| `--model_paths` / `-m` | Space-separated list of model paths for which metrics should be computed. |

## Full Evaluation

We further provide the full_eval.py script. This script specifies the routine used in our evaluation and demonstrates the use of some additional parameters, e.g., --images (-i) to define alternative image directories within COLMAP data sets. If you have downloaded and extracted all the training data, you can run it like this:

```bash
python full_eval.py -m360 <mipnerf360 folder> -tat <tanks and temples folder> -db <deep blending folder>
```

In the current version, this process takes about 7h on our reference machine containing an A6000. If you want to do the full evaluation on our pre-trained models, you can specify their download location and skip training.

```bash
python full_eval.py -o <directory with pretrained models> --skip_training -m360 <mipnerf360 folder> -tat <tanks and temples folder> -db <deep blending folder>
```

If you want to compute the metrics on our paper's evaluation images, you can also skip rendering. In this case it is not necessary to provide the source datasets. You can compute metrics for multiple image sets at a time.

```bash
python full_eval.py -m <directory with evaluation images>/garden ... --skip_training --skip_rendering
```

### Command Line Arguments for full_eval.py

| Argument | Description |
|----------|-------------|
| `--skip_training` | Flag to skip training stage. |
| `--skip_rendering` | Flag to skip rendering stage. |
| `--skip_metrics` | Flag to skip metrics calculation stage. |
| `--output_path` | Directory to put renderings and results in, ./eval by default, set to pre-trained model location if evaluating them. |
| `--mipnerf360` / `-m360` | Path to MipNeRF360 source datasets, required if training or rendering. |
| `--tanksandtemples` / `-tat` | Path to Tanks&Temples source datasets, required if training or rendering. |
| `--deepblending` / `-db` | Path to Deep Blending source datasets, required if training or rendering. |

## Processing your own Scenes

Our COLMAP loaders expect the following dataset structure in the source path location:

```
<location>
|---images
|   |---<image 0>
|   |---<image 1>
|   |---...
|---sparse
    |---0
        |---cameras.bin
        |---images.bin
        |---points3D.bin
```

For rasterization, the camera models must be either a SIMPLE_PINHOLE or PINHOLE camera. We provide a converter script [convert.py](convert.py), to extract undistorted images and SfM information from input images. Optionally, you can use ImageMagick to resize the undistorted images. This rescaling is similar to MipNeRF360, i.e., it creates images with 1/2, 1/4 and 1/8 the original resolution in corresponding folders.

To use them, please first install a recent version of COLMAP (ideally CUDA-powered) and ImageMagick. Put the images you want to use in a directory `<location>/input`.

```
<location>
|---input
    |---<image 0>
    |---<image 1>
    |---...
```

If you have COLMAP and ImageMagick on your system path, you can simply run:

```bash
python convert.py -s <location> [--resize] #If not resizing, ImageMagick is not needed
```

Alternatively, you can use the optional parameters `--colmap_executable` and `--magick_executable` to point to the respective paths. Please note that on Windows, the executable should point to the COLMAP .bat file that takes care of setting the execution environment.

Once done, `<location>` will contain the expected COLMAP data set structure with undistorted, resized input images, in addition to your original images and some temporary (distorted) data in the directory `distorted`.

If you have your own COLMAP dataset without undistortion (e.g., using OPENCV camera), you can try to just run the last part of the script: Put the images in `input` and the COLMAP info in a subdirectory `distorted`:

```
<location>
|---input
|   |---<image 0>
|   |---<image 1>
|   |---...
|---distorted
    |---database.db
    |---sparse
        |---0
            |---...
```

Then run:

```bash
python convert.py -s <location> --skip_matching [--resize] #If not resizing, ImageMagick is not needed
```

### Command Line Arguments for convert.py

| Argument | Description |
|----------|-------------|
| `--no_gpu` | Flag to avoid using GPU in COLMAP. |
| `--skip_matching` | Flag to indicate that COLMAP info is available for images. |
| `--source_path` / `-s` | Location of the inputs. |
| `--camera` | Which camera model to use for the early matching steps, OPENCV by default. |
| `--resize` | Flag for creating resized versions of input images. |
| `--colmap_executable` | Path to the COLMAP executable (.bat on Windows). |
| `--magick_executable` | Path to the ImageMagick executable. |

## Viewing your Models

We provide two viewers to display your trained Gaussian Splatting models: a network viewer for connecting to running training processes, and a real-time viewer for viewing pre-trained models.

### Network Viewer

After extracting or installing the viewers, you may run the compiled `SIBR_remoteGaussian_app[_config]` app in `<SIBR install dir>/bin`, e.g.:

```bash
./<SIBR install dir>/bin/SIBR_remoteGaussian_app
```

The network viewer allows you to connect to a running training process on the same or a different machine. If you are training on the same machine and OS, no command line parameters should be required: the optimizer communicates the location of the training data to the network viewer. By default, optimizer and network viewer will try to establish a connection on localhost on port 6009. You can change this behavior by providing matching `--ip` and `--port` parameters to both the optimizer and the network viewer.

If for some reason the path used by the optimizer to find the training data is not reachable by the network viewer (e.g., due to them running on different (virtual) machines), you may specify an override location to the viewer by using `-s <source path>`.

#### Primary Command Line Arguments for Network Viewer

| Argument | Description |
|----------|-------------|
| `--path` / `-s` | Argument to override model's path to source dataset. |
| `--ip` | IP to use for connection to a running training script. |
| `--port` | Port to use for connection to a running training script. |
| `--rendering-size` | Takes two space separated numbers to define the resolution at which network rendering occurs, 1200 width by default. Note that to enforce an aspect that differs from the input images, you need `--force-aspect-ratio` too. |
| `--load_images` | Flag to load source dataset images to be displayed in the top view for each camera. |

### Real-Time Viewer

After extracting or installing the viewers, you may run the compiled `SIBR_gaussianViewer_app[_config]` app in `<SIBR install dir>/bin`, e.g.:

```bash
./<SIBR install dir>/bin/SIBR_gaussianViewer_app -m <path to trained model>
```

It should suffice to provide the `-m` parameter pointing to a trained model directory. Alternatively, you can specify an override location for training input data using `-s`. To use a specific resolution other than the auto-chosen one, specify `--rendering-size <width> <height>`. Combine it with `--force-aspect-ratio` if you want the exact resolution and don't mind image distortion.

To unlock the full frame rate, please disable V-Sync on your machine and also in the application (Menu → Display). In a multi-GPU system (e.g., laptop) your OpenGL/Display GPU should be the same as your CUDA GPU (e.g., by setting the application's GPU preference on Windows, see below) for maximum performance.

In addition to the initial point cloud and the splats, you also have the option to visualize the Gaussians by rendering them as ellipsoids from the floating menu. SIBR has many other functionalities, please see the documentation for more details on the viewer, navigation options etc. There is also a Top View (available from the menu) that shows the placement of the input cameras and the original SfM point cloud; please note that Top View slows rendering when enabled. The real-time viewer also uses slightly more aggressive, fast culling, which can be toggled in the floating menu. If you ever encounter an issue that can be solved by turning fast culling off, please let us know.

#### Primary Command Line Arguments for Real-Time Viewer

| Argument | Description |
|----------|-------------|
| `--model-path` / `-m` | Path to trained model. |
| `--iteration` | Specifies which of state to load if multiple are available. Defaults to latest available iteration. |
| `--path` / `-s` | Argument to override model's path to source dataset. |
| `--rendering-size` | Takes two space separated numbers to define the resolution at which real-time rendering occurs, 1200 width by default. Note that to enforce an aspect that differs from the input images, you need `--force-aspect-ratio` too. |
| `--load_images` | Flag to load source dataset images to be displayed in the top view for each camera. |
| `--device` | Index of CUDA device to use for rasterization if multiple are available, 0 by default. |
| `--no_interop` | Disables CUDA/GL interop forcibly. Use on systems that may not behave according to spec (e.g., WSL2 with MESA GL 4.5 software rendering). |
