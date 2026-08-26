#!/usr/bin/env bash
# Step 5: 2터미널 워크플로(정책 서버 + 시뮬 클라이언트)를 단일 스크립트로 자동화 (tmux 미설치 → nohup 기반).
#   scripts/eval_simplerenv.sh <ckpt.pt> <port> [tasks=spoon,carrot,stack,eggplant] [ep_end=24] [parallel=1]
#   예) 스모크: scripts/eval_simplerenv.sh $CKPT 6678 spoon 2
# 결과: logs/eval/<run>/ (server.log, <task>.log, videos/), stdout 에 과제별 성공률 요약
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib/env.sh"
ckpt="$1"; port="${2:-6678}"; tasks="${3:-spoon,carrot,stack,eggplant}"; ep_end="${4:-24}"; parallel="${5:-1}"
run="$(date +%Y%m%d_%H%M%S)_$(basename "$(dirname "$(dirname "$ckpt")")")_$(basename "$ckpt" .pt)"
out="$WS/logs/eval/$run"; mkdir -p "$out"
echo "[eval] run=$run ckpt=$ckpt port=$port tasks=$tasks episodes=0..$ep_end parallel=$parallel"
trap 'bash "$WS/scripts/lib/policy_server.sh" stop "$port"' EXIT
t0=$(date +%s)
bash "$WS/scripts/lib/policy_server.sh" start "$ckpt" "$port" "$out/server.log"
t_srv=$(date +%s)
IFS=',' read -r -a TLIST <<< "$tasks"
declare -A T_START T_PID
for t in "${TLIST[@]}"; do
  T_START[$t]=$(date +%s)
  if [ "$parallel" = "1" ]; then
    bash "$WS/scripts/lib/run_task.sh" "$ckpt" "$port" "$t" 0 "$ep_end" "$out/videos" "$out/$t.log" > "$out/$t.summary" 2>&1 &
    T_PID[$t]=$!; sleep 6
  else
    bash "$WS/scripts/lib/run_task.sh" "$ckpt" "$port" "$t" 0 "$ep_end" "$out/videos" "$out/$t.log" > "$out/$t.summary" 2>&1 || true
    echo "$t wall_s=$(( $(date +%s) - T_START[$t] ))" >> "$out/timing.txt"
  fi
done
if [ "$parallel" = "1" ]; then
  for t in "${TLIST[@]}"; do wait "${T_PID[$t]}" || true; echo "$t wall_s=$(( $(date +%s) - T_START[$t] ))" >> "$out/timing.txt"; done
fi
t1=$(date +%s)
echo "== results ($run) =="
sum=0; n=0
for t in "${TLIST[@]}"; do
  sr=$( (grep -oE "Average success [0-9.]+" "$out/$t.log" || true) | tail -1 | awk '{print $3}')
  [ -z "$sr" ] && { echo "  $t: FAILED (see $out/$t.log)"; tail -5 "$out/$t.log"; continue; }
  printf "  %-9s %6.1f%%  (%s)\n" "$t" "$(echo "$sr*100" | bc -l)" "$(grep "^$t " "$out/timing.txt" || true)"
  sum=$(echo "$sum+$sr" | bc -l); n=$((n+1))
done
[ "$n" -gt 0 ] && printf "  %-9s %6.1f%%\n" "mean" "$(echo "$sum/$n*100" | bc -l)"
echo "  server_load_s=$((t_srv-t0)) eval_wall_s=$((t1-t_srv)) total_s=$((t1-t0))"
echo "  logs: $out"
