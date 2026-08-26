#!/usr/bin/env bash
# WidowX 과제 1개 평가 (simpler_env env). 인자는 원본 start_simpler_env.sh 와 동일한 값 사용.
#   run_task.sh <ckpt.pt> <port> <task: spoon|carrot|stack|eggplant> <ep_start> <ep_end> <video_dir> <logfile>
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
ckpt="$1"; port="$2"; task="$3"; ep_start="$4"; ep_end="$5"; video_dir="$6"; log="$7"
IFS='|' read -r env_name scene robot overlay rx ry <<< "${TASKS[$task]}"
mkdir -p "$(dirname "$log")" "$video_dir"
cd "$STARVLA_DIR"
export DISPLAY=""
export PATH="$(dirname "$SIM_PY"):$PATH"   # mediapy 가 PATH 에서 ffmpeg 를 찾음 (simpler_env 에 conda 로 설치)
export CUDA_VISIBLE_DEVICES="${gpu_id:-0}"
"$SIM_PY" examples/SimplerEnv/eval_files/start_simpler_env.py \
  --ckpt-path "$ckpt" --port "$port" \
  --robot "$robot" --policy-setup widowx_bridge \
  --control-freq 5 --sim-freq 500 --max-episode-steps 120 \
  --env-name "$env_name" --scene-name "$scene" \
  --rgb-overlay-path "$SIMPLER_DIR/$overlay" \
  --robot-init-x "$rx" "$rx" 1 --robot-init-y "$ry" "$ry" 1 \
  --obj-variation-mode episode --obj-episode-range "$ep_start" "$ep_end" \
  --robot-init-rot-quat-center 0 0 0 1 \
  --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
  --logging-dir "$video_dir" > "$log" 2>&1
grep -E "Average success" "$log" | tail -1
