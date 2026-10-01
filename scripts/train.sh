#!/usr/bin/env bash

set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FLASH_ATTENTION_FORCE_DISABLE=1
export TRANSFORMERS_NO_FLASH_ATTENTION=1

export WANDB_PROJECT="${WANDB_PROJECT:-gemma4-e4b-FT}"
export WANDB_LOG_MODEL="${WANDB_LOG_MODEL:-false}"


# ============================================================
# 1. Training profile
# ============================================================

PROFILE="${1:?Usage: bash scripts/train.sh <full_ft>}"


# ============================================================
# 2. Repository root
# ============================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

cd "$REPO_ROOT"

echo
echo "============================================================"
echo "Training configuration"
echo "============================================================"
echo "Profile:   ${PROFILE}"
echo "Repo root: ${REPO_ROOT}"


# ============================================================
# 3. Load configuration
# ============================================================

# config.entry provides training parameters from:
#
#   config/common.py
#   config/full_ft.py
#
# Do NOT source .env here.

eval "$(python3 -m config.entry "$PROFILE")"


# ============================================================
# 4. Normalize repository-relative paths
# ============================================================

# config/full_ft.py contains paths such as:
#
#   ./dataset/...
#   ./output/...
#
# Always resolve those relative to REPO_ROOT.

normalize_path() {

    local path="$1"

    if [[ "$path" == ./* ]]; then
        echo "${REPO_ROOT}/${path#./}"
    else
        echo "$path"
    fi
}


DATA_PATH="$(normalize_path "$DATA_PATH")"

if [[ -n "${EVAL_DATA_PATH:-}" ]]; then
    EVAL_DATA_PATH="$(normalize_path "$EVAL_DATA_PATH")"
fi

IMAGE_FOLDER="$(normalize_path "$IMAGE_FOLDER")"

OUTPUT_DIR="$(normalize_path "$OUTPUT_DIR")"


# ============================================================
# 5. Validate paths
# ============================================================

echo
echo "Resolved paths:"
echo "  DATA_PATH=${DATA_PATH}"
echo "  EVAL_DATA_PATH=${EVAL_DATA_PATH:-}"
echo "  IMAGE_FOLDER=${IMAGE_FOLDER}"
echo "  OUTPUT_DIR=${OUTPUT_DIR}"


if [[ ! -f "$DATA_PATH" ]]; then
    echo
    echo "ERROR: Training JSON not found:" >&2
    echo "  ${DATA_PATH}" >&2
    exit 2
fi


if [[ -n "${EVAL_DATA_PATH:-}" && ! -f "$EVAL_DATA_PATH" ]]; then
    echo
    echo "ERROR: Validation JSON not found:" >&2
    echo "  ${EVAL_DATA_PATH}" >&2
    exit 2
fi


if [[ ! -d "$IMAGE_FOLDER" ]]; then
    echo
    echo "ERROR: Image/video directory not found:" >&2
    echo "  ${IMAGE_FOLDER}" >&2
    exit 2
fi


mkdir -p "$OUTPUT_DIR"


# ============================================================
# 6. Show effective training configuration
# ============================================================

echo
echo "============================================================"
echo "Effective Training Settings"
echo "============================================================"

echo "MODEL_NAME=${MODEL_NAME}"
echo "MODEL_REVISION=${MODEL_REVISION:-main}"
echo "PLE_ENABLED=${PLE_ENABLED}"
echo "PLE_TRAINABLE=${PLE_TRAINABLE}"
echo "SEED=${SEED}"
echo "NUM_GPUS=${NUM_GPUS}"

echo
echo "DATA_PATH=${DATA_PATH}"
echo "EVAL_DATA_PATH=${EVAL_DATA_PATH:-}"
echo "IMAGE_FOLDER=${IMAGE_FOLDER}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

echo
echo "RUN_NAME=${RUN_NAME}"

echo
echo "BF16=${BF16}"
echo "FP16=${FP16}"

echo
echo "LEARNING_RATE=${LEARNING_RATE}"
echo "IMAGE_ENCODER_LR=${IMAGE_ENCODER_LR}"
echo "PROJECTOR_LR=${PROJECTOR_LR}"

echo
echo "PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE}"
echo "GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}"

echo
echo "============================================================"


# ============================================================
# 7. Build training arguments
# ============================================================

ARGS=(

    --deepspeed "$DEEPSPEED_CONFIG"

    --model_id "$MODEL_NAME"

    --data_path "$DATA_PATH"

    --image_folder "$IMAGE_FOLDER"

    --output_dir "$OUTPUT_DIR"

    --run_name "$RUN_NAME"

    --ple_enabled "$PLE_ENABLED"

    --ple_trainable "$PLE_TRAINABLE"

    --seed "$SEED"

    --bf16 "$BF16"

    --fp16 "$FP16"

    --num_train_epochs "$NUM_TRAIN_EPOCHS"

    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE"

    --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE"

    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"

    --optim "$OPTIM"

    --learning_rate "$LEARNING_RATE"

    --image_encoder_lr "$IMAGE_ENCODER_LR"

    --projector_lr "$PROJECTOR_LR"

    --weight_decay "$WEIGHT_DECAY"

    --warmup_ratio "$WARMUP_RATIO"

    --lr_scheduler_type "$LR_SCHEDULER_TYPE"

    --eval_strategy "$EVAL_STRATEGY"

    --load_best_model_at_end "$LOAD_BEST_MODEL_AT_END"

    --metric_for_best_model "$METRIC_FOR_BEST_MODEL"

    --save_strategy "$SAVE_STRATEGY"

    --save_total_limit "$SAVE_TOTAL_LIMIT"

    --gradient_checkpointing "$GRADIENT_CHECKPOINTING"

    --logging_steps "$LOGGING_STEPS"

    --dataloader_num_workers "$DATALOADER_NUM_WORKERS"

    --max_seq_length "$MAX_SEQ_LENGTH"

    --report_to "$REPORT_TO"
)

if [[ -n "${MODEL_REVISION:-}" ]]; then
    ARGS+=(--model_revision "$MODEL_REVISION")
fi

if [[ -n "${SAVE_STEPS:-}" ]]; then
    ARGS+=(--save_steps "$SAVE_STEPS")
fi


# ============================================================
# 8. Optional evaluation arguments
# ============================================================

if [[ -n "${EVAL_DATA_PATH:-}" ]]; then

    ARGS+=(
        --eval_data_path "$EVAL_DATA_PATH"
    )

fi


if [[ -n "${EVAL_STEPS:-}" ]]; then

    ARGS+=(
        --eval_steps "$EVAL_STEPS"
    )

fi


# ============================================================
# 9. Find DeepSpeed launcher
# ============================================================

if command -v deepspeed >/dev/null 2>&1; then
    DEEPSPEED_LAUNCHER=(deepspeed)
elif command -v uv >/dev/null 2>&1; then
    DEEPSPEED_LAUNCHER=(uv run deepspeed)
else
    echo "ERROR: deepspeed and uv are both unavailable." >&2
    exit 4
fi

echo "DeepSpeed launcher: ${DEEPSPEED_LAUNCHER[*]}"


# ============================================================
# 10. Start DeepSpeed
# ============================================================

echo
echo "============================================================"
echo "Launching DeepSpeed"
echo "============================================================"

echo "GPUs:        ${NUM_GPUS}"
echo "Master port: ${MASTER_PORT}"

echo


"${DEEPSPEED_LAUNCHER[@]}" \
    --num_gpus "$NUM_GPUS" \
    --master_port "$MASTER_PORT" \
    src/train.py \
    "${ARGS[@]}"
