#!/usr/bin/env bash

# ============================================================
# Gemma 4 Full Fine-Tuning on Nano5
#
# Submit from repository root:
#
# sbatch --account=GOV113008 --partition=taide \
#   scripts/full_ft_nano5.sbatch
# ============================================================


#SBATCH --job-name=gemma4-full-ft
#SBATCH --partition=taide

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32

#SBATCH --time=2-00:00:00

#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err


set -euo pipefail


# ============================================================
# 1. CUDA
# ============================================================

ml load gcc/12.5.0
ml load cuda/12.6

echo
echo "============================================================"
echo "Compiler"
echo "============================================================"

echo "gcc:"
which gcc
gcc --version

echo
echo "g++:"
which g++
g++ --version

echo
echo "c++:"
which c++
c++ --version


# ============================================================
# 2. Conda
# ============================================================

# Use Nano5 Miniconda directly.
#
# Do not rely on:
#
#   conda activate
#   command -v conda
#
# inside the Slurm batch environment.

CONDA_ROOT="/work/HPC_software/LMOD/miniconda3/miniconda3_app/24.11.1"

CONDA_BIN="${CONDA_ROOT}/bin/conda"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-gemma4-ft}"


if [[ ! -x "$CONDA_BIN" ]]; then

    echo "ERROR: Conda executable does not exist:" >&2
    echo "  ${CONDA_BIN}" >&2

    exit 2

fi


# ============================================================
# 3. Repository root
# ============================================================

# IMPORTANT:
#
# Submit this job from the gemma4-FT repository root.
#
# Example:
#
# cd /work/$USER/gemma4-FT
#
# sbatch --account=GOV113008 --partition=taide \
#   scripts/full_ft_nano5.sbatch


REPO_ROOT="${SLURM_SUBMIT_DIR}"


if [[ -z "${REPO_ROOT:-}" ]]; then

    echo "ERROR: SLURM_SUBMIT_DIR is not set." >&2

    exit 2

fi


cd "$REPO_ROOT"


echo
echo "Repository root:"
echo "  ${REPO_ROOT}"


# ============================================================
# 4. Validate repository
# ============================================================

# Training parameters are defined in:
#
#   config/full_ft.py
#
# No .env file is required.


if [[ ! -f "config/full_ft.py" ]]; then

    echo "ERROR: config/full_ft.py not found." >&2

    exit 2

fi


if [[ ! -f "scripts/full_ft.sh" ]]; then

    echo "ERROR: scripts/full_ft.sh not found." >&2

    exit 2

fi


# ============================================================
# 5. Validate dataset paths
# ============================================================

# These paths must match config/full_ft.py.

TRAIN_JSON="./dataset/ucf11/annotations/train.json"

VAL_JSON="./dataset/ucf11/annotations/val.json"

MEDIA_DIR="./dataset/ucf11"

PLE_MODE="${PLE_MODE:-on}"
SEED="${SEED:-42}"
case "$PLE_MODE" in
    on) PLE_ENABLED=True; PLE_TRAINABLE=True ;;
    frozen) PLE_ENABLED=True; PLE_TRAINABLE=False ;;
    off) PLE_ENABLED=False; PLE_TRAINABLE=False ;;
    *) echo "PLE_MODE must be on, frozen, or off" >&2; exit 2 ;;
esac
OUTPUT_DIR="./output/ucf11_ple_${PLE_MODE}_seed${SEED}"
RUN_NAME="ucf11-ple-${PLE_MODE}-seed${SEED}"
export PLE_ENABLED PLE_TRAINABLE SEED OUTPUT_DIR RUN_NAME


if [[ ! -f "$TRAIN_JSON" ]]; then

    echo "ERROR: Training JSON not found:" >&2
    echo "  ${REPO_ROOT}/${TRAIN_JSON#./}" >&2

    exit 2

fi


if [[ ! -f "$VAL_JSON" ]]; then

    echo "ERROR: Validation JSON not found:" >&2
    echo "  ${REPO_ROOT}/${VAL_JSON#./}" >&2

    exit 2

fi


if [[ ! -d "$MEDIA_DIR" ]]; then

    echo "ERROR: Media directory not found:" >&2
    echo "  ${REPO_ROOT}/${MEDIA_DIR#./}" >&2

    exit 2

fi


mkdir -p "$OUTPUT_DIR"


# ============================================================
# 6. Cache / Runtime settings
# ============================================================

# Put large Hugging Face / Torch / Triton caches in /work
# instead of consuming /home quota.

export HF_HOME="${HF_HOME:-/work/${USER}/.cache/huggingface}"

export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/work/${USER}/.cache/torch_extensions}"

export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/work/${USER}/.cache/triton}"


# CPU threads used by each process.

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"


# Hugging Face tokenizer settings.

export TOKENIZERS_PARALLELISM=false


# NCCL debugging / error handling.

export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

mkdir -p \
    "$HF_HOME" \
    "$TORCH_EXTENSIONS_DIR" \
    "$TRITON_CACHE_DIR"


# Repository-provided uv shim.

if [[ -f "scripts/conda-bin/uv" ]]; then

    chmod u+x scripts/conda-bin/uv

fi


# ============================================================
# 7. GPU configuration
# ============================================================

# Must match:
#
#   #SBATCH --gpus-per-node=2
#
# and:
#
#   config/full_ft.py
#   "num_gpus": 4

export NUM_GPUS=4


# ============================================================
# 8. Job information
# ============================================================

echo
echo "============================================================"
echo "Slurm Job Information"
echo "============================================================"

echo "Job ID:             ${SLURM_JOB_ID}"
echo "Node:               $(hostname)"
echo "Submit directory:   ${SLURM_SUBMIT_DIR}"
echo "Repository root:    ${REPO_ROOT}"

echo
echo "Slurm:"
echo "  GPUs requested:   ${NUM_GPUS}"
echo "  CPUs per task:    ${SLURM_CPUS_PER_TASK:-unknown}"

echo
echo "Conda:"
echo "  Root:              ${CONDA_ROOT}"
echo "  Executable:        ${CONDA_BIN}"
echo "  Environment:       ${CONDA_ENV_NAME}"

echo
echo "CUDA:"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

echo
echo "Training config:"
echo "  config/full_ft.py"

echo
echo "Training JSON:"
echo "  ${TRAIN_JSON}"

echo
echo "Validation JSON:"
echo "  ${VAL_JSON}"

echo
echo "Media directory:"
echo "  ${MEDIA_DIR}"

echo
echo "Output directory:"
echo "  ${OUTPUT_DIR}"

echo
echo "Cache:"
echo "  HF_HOME=${HF_HOME}"
echo "  TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR}"
echo "  TRITON_CACHE_DIR=${TRITON_CACHE_DIR}"


# ============================================================
# 9. NVIDIA information
# ============================================================

echo
echo "============================================================"
echo "nvidia-smi"
echo "============================================================"

nvidia-smi

echo
echo "============================================================"
echo "nvidia-smi -L"
echo "============================================================"

nvidia-smi -L


# ============================================================
# 10. Check Conda environment
# ============================================================

echo
echo "============================================================"
echo "Checking Conda environment"
echo "============================================================"


"$CONDA_BIN" env list


if ! "$CONDA_BIN" env list \
    | awk '{print $1}' \
    | grep -qx "$CONDA_ENV_NAME"; then

    echo
    echo "ERROR: Conda environment '${CONDA_ENV_NAME}' was not found." >&2

    echo
    echo "Available Conda environments:" >&2

    "$CONDA_BIN" env list >&2

    exit 2

fi


# ============================================================
# 11. Run inside gemma4-ft environment
# ============================================================

echo
echo "============================================================"
echo "Entering Conda environment: ${CONDA_ENV_NAME}"
echo "============================================================"


"$CONDA_BIN" run \
    --no-capture-output \
    -n "$CONDA_ENV_NAME" \
    bash -c '

set -euo pipefail


REPO_ROOT="$1"


cd "$REPO_ROOT"


# ============================================================
# Repository tools
# ============================================================

if [[ -d "$REPO_ROOT/scripts/conda-bin" ]]; then

    export PATH="$REPO_ROOT/scripts/conda-bin:$PATH"

fi


# ============================================================
# Python environment
# ============================================================

echo
echo "============================================================"
echo "Python Environment"
echo "============================================================"


echo
echo "Python executable:"

which python3


echo
echo "Python version:"

python3 --version


# ============================================================
# Package versions
# ============================================================

echo
echo "============================================================"
echo "Package Versions"
echo "============================================================"


python3 - <<'"'"'PY'"'"'

import torch
import transformers
import deepspeed


print("torch =", torch.__version__)

print("torch CUDA =", torch.version.cuda)

print("CUDA available =", torch.cuda.is_available())

print("GPU count =", torch.cuda.device_count())

print("transformers =", transformers.__version__)

print("deepspeed =", deepspeed.__version__)


for i in range(torch.cuda.device_count()):

    print(
        f"GPU {i}:",
        torch.cuda.get_device_name(i),
    )

PY


# ============================================================
# Validate CUDA
# ============================================================

CUDA_AVAILABLE="$(
    python3 -c "import torch; print(int(torch.cuda.is_available()))"
)"


if [[ "$CUDA_AVAILABLE" != "1" ]]; then

    echo
    echo "ERROR: PyTorch reports CUDA is not available." >&2

    exit 3

fi


# ============================================================
# Validate GPU count
# ============================================================

VISIBLE_GPUS="$(
    python3 -c "import torch; print(torch.cuda.device_count())"
)"


echo
echo "============================================================"
echo "GPU Check"
echo "============================================================"

echo "Expected GPUs: ${NUM_GPUS}"

echo "PyTorch visible GPUs: ${VISIBLE_GPUS}"


if [[ "$VISIBLE_GPUS" != "$NUM_GPUS" ]]; then

    echo
    echo "ERROR: PyTorch GPU count mismatch." >&2

    echo "Expected GPUs: ${NUM_GPUS}" >&2

    echo "torch.cuda.device_count(): ${VISIBLE_GPUS}" >&2

    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}" >&2

    exit 3

fi


# ============================================================
# Start training
# ============================================================

echo
echo "============================================================"
echo "Starting Gemma 4 Full Fine-Tuning"
echo "============================================================"

echo
echo "Start time: $(date)"

echo


bash scripts/full_ft.sh


echo
echo "============================================================"
echo "Training Finished"
echo "============================================================"

echo
echo "End time: $(date)"

' bash "$REPO_ROOT"
