#!/usr/bin/env bash
# Phase 2 재학습 런 실행 (클라우드 8 GPU 가정). 이번 세션에서는 실행하지 않음 — 초안.
#   bash scripts/phase2_launch.sh configs/phase2_matrix/groot_20k.yaml [seed] [num_gpus] [extra accelerate/config overrides...]
# 사전 조건 (클라우드 노드):
#   1) scripts/01_clone_third_party.sh, scripts/02_create_envs.sh starvla  (starVLA env)
#   2) hf download StarVLA/Qwen3-VL-4B-Instruct-Action --local-dir playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action
#   3) scripts/07_prepare_datasets.sh  (bridge + fractal LeRobot, modality.json 배치, 데이터로더 검증)
#   4) export WANDB_API_KEY=...  (또는 WANDB_MODE=disabled)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib/env.sh"
cfg="$(realpath "$1")"; seed="${2:-0}"; ngpu="${3:-8}"; shift $(( $# >= 3 ? 3 : $# ))
run_id="$(grep -E '^run_id:' "$cfg" | awk '{print $2}' | sed "s/_s0$/_s${seed}/")"
cd "$STARVLA_DIR"
mkdir -p "results/Checkpoints/$run_id"; cp "$cfg" "results/Checkpoints/$run_id/launch_config.yaml"; cp "$0" "results/Checkpoints/$run_id/"
export NCCL_ASYNC_ERROR_HANDLING=1
"$CONDA_BASE/envs/starVLA/bin/accelerate" launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "$ngpu" \
  starVLA/training/train_starvla.py \
  --config_yaml "$cfg" \
  --seed "$seed" \
  --run_id "$run_id" \
  "$@"
