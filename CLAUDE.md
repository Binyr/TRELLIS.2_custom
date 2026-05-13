# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 特别注意
不允许修改`特别注意`这部分，可以修改其他部分

此机器为一个mac机器，不能跑任何gpu任务；

如果你想跑gpu任务，我已经帮你配置好了一个debug机器，它是/Users/binyanrui/.ssh/config里的 yanruibin-job-debug-gpu

你需要通过sync.sh里的命令将此机器上的代码同步到远程机器，执行的时候别忘了替换一下机器名字

我希望你每次现在本地修改代码，然后同步到远程机器上运行。

不要修改远程的环境，如果环境有问题，请立刻停止并报告给我

你的远程目录为`/local-ssd/xxx/`

远程的环境，通过在远程仓库目录下，`source .venv/bin/activate`激活

远程里只有outputs目录是永存目录，其他目录都是临时目录，运行结果请尽可能保存在outputs里

data里的东西不允许修改

不允许修改此机器，以及远程机器本仓库外的文件

你在运行时的任何中间结果，都存储在本仓库的claude_tmp下面，不要本仓库之外，以避免不断询问权限

我授权给你本仓库的所有权限，包括读写

## Project Overview

TRELLIS.2 is a 4B-parameter 3D generative model for image-to-3D generation using PyTorch. It uses a sparse voxel representation called **O-Voxel** to generate 3D assets with PBR materials. Python 3.8+, PyTorch 2.6.0, CUDA 12.4.

## Setup

```bash
# Full install (creates conda env "trellis2")
. ./setup.sh --new-env --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm

# Individual flags: --new-env, --basic, --flash-attn, --nvdiffrast, --nvdiffrec, --cumesh, --o-voxel, --flexgemm
```

Requires Linux, NVIDIA GPU (24GB+ VRAM), CUDA Toolkit 12.4.

## Running

```bash
# Inference
python example.py                # Image-to-3D (outputs sample.mp4, sample.glb)
python example_texturing.py      # PBR texture generation
python app.py                    # Gradio web UI for image-to-3D
python app_texturing.py          # Gradio web UI for texturing

# Training (single GPU)
python train.py --config configs/scvae/shape_vae_next_dc_f16c32_fp16.json \
  --output_dir results/my_run --data_dir '{"ObjaverseXL_sketchfab": {...}}'

# Distributed training
python train.py --config configs/gen/ss_flow_img_dit_1_3B_64_bf16.json \
  --num_nodes 2 --num_gpus 4 --master_addr hostname --master_port 12355

# Data preparation (see data_toolkit/README.md for full pipeline)
python data_toolkit/build_metadata.py ObjaverseXL --source sketchfab --root datasets/ObjaverseXL_sketchfab
```

There is no formal test suite. Validation is done via the example scripts.

## Architecture

### Three-Stage Generation Pipeline

Image-to-3D generation is a cascade of three flow models:

1. **Sparse Structure (SS)**: Generates low-res sparse voxel grid (~16³) via `SparseStructureFlowModel`
2. **Shape SLat**: Upscales to high-res shape (512³→1024³) via `SLatFlowModel`, conditioned on image + sparse structure
3. **Texture SLat**: Generates PBR textures via `SLatFlowModel`, conditioned on image + shape

Each stage has a corresponding SC-VAE (Sparse Convolutional VAE) that encodes/decodes between voxel space and latent space.

### Key Modules (`trellis2/`)

- **pipelines/**: `Trellis2ImageTo3DPipeline` and `Trellis2TexturingPipeline` — main inference entry points. Load from HuggingFace via `from_pretrained("microsoft/TRELLIS.2-4B")`
- **models/**: All model architectures. Uses lazy-import via `__attributes` dict in `__init__.py` — add new models there to register them
- **datasets/**: Dataset classes, also lazy-imported via `__attributes` dict. Each dataset corresponds to a training stage
- **trainers/**: `Trainer` base class in `basic.py`, with VAE trainers (`vae/`) and flow matching trainers (`flow_matching/`)
- **modules/**: Low-level building blocks — sparse tensors, attention (with RoPE), transformers, normalization
- **renderers/**: `MeshRenderer`, `PBRMeshRenderer`, `VoxelRenderer`, plus `EnvMap` for HDR lighting
- **representations/**: Mesh and voxel data structures
- **utils/**: Distributed training, gradient clipping, loss functions, rendering helpers

### O-Voxel (`o-voxel/`)

Git submodule with its own build system. Core library for mesh↔voxel conversion:
- `convert.py`: Mesh to flexible dual grid (with QEF for sharp features)
- `postprocess.py`: GLB export with UV unwrapping and decimation
- `serialize.py`: Spatial hashing / Z-order encoding
- `io.py`: Compact `.vxz` file format

### Configuration-Driven Training

All training is controlled by JSON configs in `configs/`:
- `configs/scvae/`: VAE training configs (shape and texture, base and fine-tune resolutions)
- `configs/gen/`: Flow model configs (SS, shape, texture, base and fine-tune resolutions)

Configs specify model classes + args, dataset class + args, and trainer class + hyperparameters. The `train.py` script instantiates everything from the config via the lazy-import registries.

### Adding New Components

- **Model**: Implement in `trellis2/models/`, register in `trellis2/models/__init__.py` `__attributes` dict
- **Dataset**: Implement in `trellis2/datasets/`, register in `trellis2/datasets/__init__.py` `__attributes` dict
- **Trainer**: Implement in `trellis2/trainers/`, extend `Trainer` base class, reference in config JSON

## Key Environment Variables

```bash
export OPENCV_IO_ENABLE_OPENEXR=1                          # Required for EXR file support
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"   # Reduces GPU memory fragmentation
export ATTN_BACKEND=xformers                                 # Use xformers instead of flash-attn (for V100 etc.)
```
