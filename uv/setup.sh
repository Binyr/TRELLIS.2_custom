# uv cache directory resolution (evaluated at source time):
#   1. /opt/uv-cache  – pre-populated in the remote machine image (fastest)
#   2. <repo>/.uv-cache – persistent fallback on the network disk
# source uv/setup.sh --new-env --venv-dir /local-ssd/trellis.2-venv
pwd
if [ -d "/opt/uv-cache" ] ; then
    export UV_CACHE_DIR="/opt/uv-cache"
else
    export UV_CACHE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.uv-cache"
fi
echo "[UV_CACHE] Using cache: $UV_CACHE_DIR"

# Read Arguments
TEMP=`getopt -o h --long help,new-env,basic,train,xformers,flash-attn,diffoctreerast,vox2seq,spconv,mipgaussian,kaolin,nvdiffrast,demo,cuda:,venv-dir: -n 'setup_uv.sh' -- "$@"`

eval set -- "$TEMP"

HELP=false
NEW_ENV=false
BASIC=true
TRAIN=false
XFORMERS=true
FLASHATTN=true
DIFFOCTREERAST=true
VOX2SEQ=true
SPCONV=true
ERROR=false
MIPGAUSSIAN=true
KAOLIN=true
NVDIFFRAST=true
DEMO=false
CUDA_ARG="12.8"  # default CUDA version for --new-env
VENV_DIR=""       # if set, create venv here and symlink .venv -> VENV_DIR

if [ "$#" -eq 1 ] ; then
    HELP=true
fi

while true ; do
    case "$1" in
        -h|--help) HELP=true ; shift ;;
        --new-env) NEW_ENV=true ; shift ;;
        --basic) BASIC=true ; shift ;;
        --train) TRAIN=true ; shift ;;
        --xformers) XFORMERS=true ; shift ;;
        --flash-attn) FLASHATTN=true ; shift ;;
        --diffoctreerast) DIFFOCTREERAST=true ; shift ;;
        --vox2seq) VOX2SEQ=true ; shift ;;
        --spconv) SPCONV=true ; shift ;;
        --mipgaussian) MIPGAUSSIAN=true ; shift ;;
        --kaolin) KAOLIN=true ; shift ;;
        --nvdiffrast) NVDIFFRAST=true ; shift ;;
        --demo) DEMO=true ; shift ;;
        --cuda) CUDA_ARG="$2" ; shift 2 ;;
        --venv-dir) VENV_DIR="$2" ; shift 2 ;;
        --) shift ; break ;;
        *) ERROR=true ; break ;;
    esac
done

if [ "$ERROR" = true ] ; then
    echo "Error: Invalid argument"
    HELP=true
fi

if [ "$HELP" = true ] ; then
    echo "Usage: source setup_uv.sh [OPTIONS]"
    echo "Options:"
    echo "  -h, --help              Display this help message"
    echo "  --new-env               Create a new uv virtual environment (.venv)"
    echo "  --cuda <version>        CUDA version for PyTorch install (default: 12.8, options: 11.8, 12.1, 12.4, 12.8, rocm6.1)"
  echo "  --venv-dir <path>       Create the venv at <path> and symlink .venv -> <path> (e.g. /local-ssd/ss4d-venv)"
    echo "  --basic                 Install basic dependencies"
    echo "  --train                 Install training dependencies"
    echo "  --xformers              Install xformers"
    echo "  --flash-attn            Install flash-attn"
    echo "  --diffoctreerast        Install diffoctreerast"
    echo "  --vox2seq               Install vox2seq"
    echo "  --spconv                Install spconv"
    echo "  --mipgaussian           Install mip-splatting"
    echo "  --kaolin                Install kaolin"
    echo "  --nvdiffrast            Install nvdiffrast"
    echo "  --demo                  Install all dependencies for demo"
    return 2>/dev/null || exit 0
fi

if [ "$NEW_ENV" = true ] ; then
    # Map CUDA_ARG to PyTorch index URL tag
    case $CUDA_ARG in
        11.8)   TORCH_INDEX_TAG="cu118" ;;
        12.1)   TORCH_INDEX_TAG="cu121" ;;
        12.4)   TORCH_INDEX_TAG="cu124" ;;
        12.8)   TORCH_INDEX_TAG="cu128" ;;  # native cu128 wheels available since PyTorch 2.8.0
        rocm6.1) TORCH_INDEX_TAG="rocm6.1" ;;
        *)
            echo "[NEW-ENV] Unsupported --cuda version: $CUDA_ARG (choose 11.8, 12.1, 12.4, 12.8, rocm6.1)"
            return 2>/dev/null || exit 1
            ;;
    esac

    echo "[NEW-ENV] Creating uv virtual environment (Python 3.10) ..."
    # Determine where the actual venv directory lives
    if [ -n "$VENV_DIR" ] ; then
        ACTUAL_VENV="$VENV_DIR"
    else
        ACTUAL_VENV="$(pwd)/.venv"
    fi
    uv venv "$ACTUAL_VENV" --python 3.12
    # Symlink .venv -> ACTUAL_VENV when using an external venv dir
    if [ -n "$VENV_DIR" ] ; then
        rm -rf "$(pwd)/.venv"
        ln -s "$ACTUAL_VENV" "$(pwd)/.venv"
        echo "[NEW-ENV] Symlinked .venv -> $ACTUAL_VENV"
    fi
    source "$ACTUAL_VENV/bin/activate"

    echo "[NEW-ENV] Installing PyTorch 2.8.0 + torchvision 0.23.0 (index: $TORCH_INDEX_TAG) ..."
    if [ "$TORCH_INDEX_TAG" = "rocm6.1" ] ; then
        uv pip install torch==2.4.1 torchvision==0.19.1 \
            --index-url https://download.pytorch.org/whl/rocm6.1
        # amd_smi needed for ROCm
        mkdir -p /tmp/extensions
        sudo cp /opt/rocm/share/amd_smi /tmp/extensions/amd_smi -r
        cd /tmp/extensions/amd_smi
        sudo chmod -R 777 .
        uv pip install .
        cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
    else
        uv pip install torch==2.8.0 torchvision==0.23.0 \
            --index-url "https://download.pytorch.org/whl/${TORCH_INDEX_TAG}"
    fi
fi

# Activate venv if it exists and is not already active
if [ -z "$VIRTUAL_ENV" ] && [ -f ".venv/bin/activate" ] ; then
    source .venv/bin/activate
fi

# Get system information
WORKDIR=$(pwd)
PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
PLATFORM=$(python -c "import torch; print(('cuda' if torch.version.cuda else ('hip' if torch.version.hip else 'unknown')) if torch.cuda.is_available() else 'cpu')")
case $PLATFORM in
    cuda)
        CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)")
        CUDA_MAJOR_VERSION=$(echo $CUDA_VERSION | cut -d'.' -f1)
        CUDA_MINOR_VERSION=$(echo $CUDA_VERSION | cut -d'.' -f2)
        echo "[SYSTEM] PyTorch Version: $PYTORCH_VERSION, CUDA Version: $CUDA_VERSION"
        ;;
    hip)
        HIP_VERSION=$(python -c "import torch; print(torch.version.hip)")
        HIP_MAJOR_VERSION=$(echo $HIP_VERSION | cut -d'.' -f1)
        HIP_MINOR_VERSION=$(echo $HIP_VERSION | cut -d'.' -f2)
        if [ "$PYTORCH_VERSION" != "2.4.1+rocm6.1" ] ; then
            echo "[SYSTEM] Installing PyTorch 2.4.1 for HIP ($PYTORCH_VERSION -> 2.4.1+rocm6.1)"
            uv pip install torch==2.4.1 torchvision==0.19.1 \
                --index-url https://download.pytorch.org/whl/rocm6.1
            mkdir -p /tmp/extensions
            sudo cp /opt/rocm/share/amd_smi /tmp/extensions/amd_smi -r
            cd /tmp/extensions/amd_smi
            sudo chmod -R 777 .
            uv pip install .
            cd $WORKDIR
            PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
        fi
        echo "[SYSTEM] PyTorch Version: $PYTORCH_VERSION, HIP Version: $HIP_VERSION"
        ;;
    *)
        ;;
esac


# torchsparse
echo "[INSTALL] torchsparse..."
apt update
apt install -y libsparsehash-dev ninja-build
apt install -y  libegl1
# x=$(pwd)
# cp -r  /efs/yanruibin/projects/video_pixal3d_train/third_party/torchsparse third_party/
# cd third_party/torchsparse
# uv pip install . --no-build-isolation
# cd $x
uv pip install /efs/yanruibin/packages/torchsparse-2.1.0-cp312-cp312-linux_x86_64.whl

echo "[INSTALL] requirements_v2.txt..."
apt install -y python3.12-dev # Triton 在初始化时需要编译一个小的 C 文件，要用到 Python 头文件，但系统没装 Python dev 包
uv pip install -r requirements_v2.txt
echo "[INSTALL] direct3ds2_train (editable)..."
# uv pip install -e .

# echo "[INSTALL] third_party/voxelize..."
# uv pip install third_party/voxelize/ --no-build-isolation

uv pip install pytorch_lightning
uv pip install wandb
uv pip install lightning
uv pip install /efs/yanruibin/packages/natten-0.21.1+torch280cu128-cp312-cp312-linux_x86_64.whl
uv pip install /efs/yanruibin/packages/utils3d-1.7-py3-none-any.whl
uv pip install tensorboard
uv pip install flash-attn-4==4.0.0b8

if [ "$BASIC" = true ] ; then
    echo "[INSTALL] basic dependencies..."
    uv pip install pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless scipy ninja rembg onnxruntime trimesh open3d xatlas pyvista pymeshfix igraph transformers
fi

if [ "$TRAIN" = true ] ; then
    echo "[INSTALL] training dependencies..."
    uv pip install tensorboard pandas lpips
    uv pip uninstall -y pillow
    sudo apt install -y libjpeg-dev
    uv pip install pillow-simd
fi

# echo "[INSTALL] xformers..."
# uv pip install -U xformers --index-url https://download.pytorch.org/whl/cu128

if [ "$FLASHATTN" = true ] ; then
    if [ "$PLATFORM" = "cuda" ] ; then
        echo "[INSTALL] flash-attn..."
        uv pip install flash-attn --no-build-isolation
    elif [ "$PLATFORM" = "hip" ] ; then
        echo "[INSTALL] flash-attn (ROCm, building from source)..."
        mkdir -p /tmp/extensions
        git clone --recursive https://github.com/ROCm/flash-attention.git /tmp/extensions/flash-attention
        cd /tmp/extensions/flash-attention
        git checkout tags/v2.6.3-cktile
        GPU_ARCHS=gfx942 python setup.py install  # MI300 series
        cd $WORKDIR
    else
        echo "[FLASHATTN] Unsupported platform: $PLATFORM"
    fi
fi

echo "[INSTALL] kaolin..."
uv pip install kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu128.html

if [ "$SPCONV" = true ] ; then
    if [ "$PLATFORM" = "cuda" ] ; then
        echo "[INSTALL] spconv..."
        case $CUDA_MAJOR_VERSION in
            11) uv pip install spconv-cu118 ;;
            12)
                case $CUDA_MINOR_VERSION in
                    0|1|2|3|4|5) uv pip install spconv-cu120 ;;
                    6|7)          uv pip install spconv-cu126 ;;
                    *)            uv pip install spconv-cu126 ;;  # 12.8+: use cu126 until spconv-cu128 is available
                esac
                ;;
            *) echo "[SPCONV] Unsupported PyTorch CUDA version: $CUDA_MAJOR_VERSION" ;;
        esac
    else
        echo "[SPCONV] Unsupported platform: $PLATFORM"
    fi
fi

# ln -s /efs/yanruibin/projects/video_pixal3d_train/data
# ln -s /efs/yanruibin/projects/video_pixal3d_train/outputs/

uv pip uninstall flash-attn-4
uv pip install flash-attn-4==4.0.0b8

uv pip install pyrender
uv pip install "pyglet<2"
uv pip install fast-simplification
uv pip install --extra-index-url https://miropsota.github.io/torch_packages_builder "pytorch3d==0.7.8+pt2.8.0cu128"
uv pip install Kornia
# apt install -y zsh
# sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
# source .venv/bin/activate

# uv pip install git+https://github.com/ashawkey/cubvh.git --no-build-isolation
# uv pip install diso --no-build-isolation


uv pip install objaverse

ln -s /threed-code/yanruibin/trellis.2_data/
ln -s /efs/yanruibin/projects/video_pixal3d_train/data
# ln -s /threed-code/yanruibin/efs/4D_video_data_process/data2/ data
# HF_TOKEN should be set in your environment (e.g. ~/.bashrc or secrets manager)
# export HF_TOKEN=your_token_here

# ===== TRELLIS.2 data pipeline dependencies =====
# Ensure eigen submodule is available for o-voxel build
if [ ! -f "o-voxel/third_party/eigen/Eigen/Dense" ] ; then
    echo "[INSTALL] Cloning Eigen for o-voxel build..."
    rm -rf o-voxel/third_party/eigen
    git clone --depth 1 https://gitlab.com/libeigen/eigen.git o-voxel/third_party/eigen
fi

# nvdiffrast (required by o_voxel)
echo "[INSTALL] nvdiffrast..."
mkdir -p /tmp/extensions
if [ ! -d "/tmp/extensions/nvdiffrast" ] ; then
    git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast
fi
uv pip install /tmp/extensions/nvdiffrast --no-build-isolation

# o-voxel and its compiled dependencies (pre-built wheels to avoid high memory usage)
echo "[INSTALL] cumesh, flex_gemm, o-voxel (pre-built wheels)..."
uv pip install /efs/yanruibin/packages/cumesh-0.0.1-cp312-cp312-linux_x86_64.whl
uv pip install /efs/yanruibin/packages/flex_gemm-1.0.0-cp312-cp312-linux_x86_64.whl
uv pip install /efs/yanruibin/packages/o_voxel-0.0.1-cp312-cp312-linux_x86_64.whl --no-deps

# Blender (for dump_pbr_4d.py)
echo "[INSTALL] Blender 4.5.1..."
BLENDER_PATH="/tmp/blender-4.5.1-linux-x64/blender"
if [ ! -f "$BLENDER_PATH" ] ; then
    wget -q https://ftp.halifax.rwth-aachen.de/blender/release/Blender4.5/blender-4.5.1-linux-x64.tar.xz -O /tmp/blender-4.5.1-linux-x64.tar.xz
    tar -xf /tmp/blender-4.5.1-linux-x64.tar.xz -C /tmp/
fi
/tmp/blender-4.5.1-linux-x64/4.5/python/bin/python3.11 -m pip install -q pillow
echo "[DONE] Blender ready at: $BLENDER_PATH"
