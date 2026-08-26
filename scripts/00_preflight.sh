#!/usr/bin/env bash
# Step 0: 사전 점검. 결과를 stdout + logs/preflight_<date>.log 에 기록.
set -u
WS="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$WS/logs/preflight_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG") 2>&1
echo "== preflight @ $(date) =="
echo "-- GPU --"; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv 2>&1
nvidia-smi | grep -o "CUDA Version: [0-9.]*"
echo "-- disk --"; df -h "$WS" "$HOME" | sed 1d
echo "-- conda --"; conda --version 2>&1; conda env list 2>&1 | grep -v '^#'
echo "-- huggingface --"; (command -v huggingface-cli || command -v hf || echo "hf cli: not installed") 2>&1
ls "$HOME/.cache/huggingface/token" 2>/dev/null && echo "HF token: present" || echo "HF token: absent (public repos only)"
echo "-- internet --"; for u in https://github.com https://huggingface.co; do printf "%s -> " "$u"; curl -s -o /dev/null -w "%{http_code}\n" -m 10 "$u" 2>/dev/null; done
echo "-- vulkan --"; ldconfig -p | grep -E "libvulkan.so.1" || echo "libvulkan.so.1: MISSING"
ls /usr/share/vulkan/icd.d/ 2>/dev/null
echo "-- misc --"; command -v tmux >/dev/null && tmux -V || echo "tmux: not installed"
echo "cpu: $(nproc)  ram: $(free -g | awk '/Mem/{print $2}')G"; lsb_release -ds 2>/dev/null
echo "log: $LOG"
