#!/usr/bin/env bash
# PLE ablation: train or evaluate the on / frozen / off arms.
#
#   on      PLE active, weights updated
#   frozen  PLE active, weights frozen
#   off     PLE input zeroed, weights frozen
#
# Usage: bash scripts/run_ple_ablation.sh <train|eval> <on|frozen|off|all> [seed ...]
#   bash scripts/run_ple_ablation.sh train frozen 42
#   bash scripts/run_ple_ablation.sh train all 42 43 44
#   bash scripts/run_ple_ablation.sh eval all 42
set -euo pipefail

USAGE="Usage: bash scripts/run_ple_ablation.sh <train|eval> <on|frozen|off|all> [seed ...]"
ACTION="${1:?$USAGE}"
ARM="${2:?$USAGE}"
shift 2
SEEDS=("${@:-42}")

case "$ACTION" in
    train|eval) ;;
    *) echo "Action must be train or eval" >&2; exit 2 ;;
esac
case "$ARM" in
    all) PLE_MODES=(on frozen off) ;;
    on|frozen|off) PLE_MODES=("$ARM") ;;
    *) echo "PLE mode must be on, frozen, off, or all" >&2; exit 2 ;;
esac

train_arm() {
    local ple_mode="$1" seed="$2"
    case "$ple_mode" in
        on) PLE_ENABLED=True; PLE_TRAINABLE=True ;;
        frozen) PLE_ENABLED=True; PLE_TRAINABLE=False ;;
        off) PLE_ENABLED=False; PLE_TRAINABLE=False ;;
    esac
    SEED="$seed"
    OUTPUT_DIR="./output/ucf11_ple_${ple_mode}_seed${seed}"
    RUN_NAME="ucf11-ple-${ple_mode}-seed${seed}"
    export PLE_ENABLED PLE_TRAINABLE SEED OUTPUT_DIR RUN_NAME
    bash scripts/full_ft.sh
}

eval_arm() {
    local ple_mode="$1" seed="$2"
    python3 src/test_set.py \
        --model_id "output/ucf11_ple_${ple_mode}_seed${seed}" \
        --data_path dataset/ucf11/annotations/test.json \
        --image_folder dataset/ucf11 \
        --candidate_labels dataset/ucf11/annotations/labels.txt \
        --max_seq_length 3072 --skip_bertscore --max_new_tokens 32
}

for seed in "${SEEDS[@]}"; do
    for ple_mode in "${PLE_MODES[@]}"; do
        "${ACTION}_arm" "$ple_mode" "$seed"
    done
done
