#!/usr/bin/env bash
# 정책 서버 제어 (starVLA env, nohup 백그라운드).
#   policy_server.sh start <ckpt.pt> <port> <logfile>   # 포트가 열릴 때까지 대기 (최대 600s)
#   policy_server.sh stop  <port>
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cmd="$1"; shift
case "$cmd" in
  start)
    ckpt="$1"; port="$2"; log="$3"
    [ -f "$ckpt" ] || { echo "ckpt not found: $ckpt" >&2; exit 1; }
    mkdir -p "$(dirname "$log")"
    cd "$STARVLA_DIR"
    CUDA_VISIBLE_DEVICES="${gpu_id:-0}" nohup "$STARVLA_PY" deployment/model_server/server_policy.py \
      --ckpt_path "$ckpt" --port "$port" --use_bf16 --idle_timeout -1 > "$log" 2>&1 &
    pid=$!; echo "$pid" > "$log.pid"
    echo "[server] pid=$pid port=$port log=$log"
    for i in $(seq 1 300); do
      if ! kill -0 "$pid" 2>/dev/null; then echo "[server] died; tail:" >&2; tail -30 "$log" >&2; exit 1; fi
      if (exec 3<>/dev/tcp/127.0.0.1/"$port") 2>/dev/null; then echo "[server] ready after ${i}x2s"; exit 0; fi
      sleep 2
    done
    echo "[server] timeout waiting for port $port" >&2; exit 1 ;;
  stop)
    port="$1"
    pids=$(pgrep -f "server_policy.py --ckpt_path .* --port $port( |$)" || true)
    [ -n "$pids" ] && { echo "[server] stopping $pids"; kill $pids 2>/dev/null || true; sleep 3; kill -9 $pids 2>/dev/null || true; } || echo "[server] none on port $port" ;;
  *) echo "usage: $0 start <ckpt> <port> <log> | stop <port>" >&2; exit 1 ;;
esac
