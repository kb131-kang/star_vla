#!/usr/bin/env bash
# Step 3: 공식 검증 (1) SimplerEnv 빌드 (2) 프레임워크 스모크 2종. 데이터로더 검증(3)은 Step 7 데이터 수령 후 07 스크립트에서 수행.
#   bash scripts/03_smoke_tests.sh [sim|groot|oft|all]
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib/env.sh"
what="${1:-all}"; mkdir -p "$WS/logs/smoke"; cd "$STARVLA_DIR"; rc=0
run_sim() {
  echo "== [1] SimplerEnv env build test (simpler_env)"
  DISPLAY="" "$SIM_PY" examples/SimplerEnv/eval_files/test_your_simplerEnv.py > "$WS/logs/smoke/sim_env.log" 2>&1
  grep -q "Env built successfully" "$WS/logs/smoke/sim_env.log" && echo "   ✅ Env built successfully" || { echo "   ❌ FAIL (logs/smoke/sim_env.log)"; tail -20 "$WS/logs/smoke/sim_env.log"; rc=1; }
}
run_fw() {  # $1 = QwenGR00T | QwenOFT
  echo "== [2] framework smoke: $1 (starVLA)"
  "$STARVLA_PY" "starVLA/model/framework/VLM4A/$1.py" --config_yaml "$WS/configs/smoke_oxe_qwen3vl.yaml" > "$WS/logs/smoke/$1.log" 2>&1
  if grep -q "^Finished" "$WS/logs/smoke/$1.log"; then
    echo "   ✅ $1: $(grep -E 'Action Loss|Predicted Action shape' "$WS/logs/smoke/$1.log" | head -2 | tr '\n' ' ')"
  else echo "   ❌ $1 FAIL (logs/smoke/$1.log)"; tail -20 "$WS/logs/smoke/$1.log"; rc=1; fi
}
case "$what" in
  sim) run_sim ;; groot) run_fw QwenGR00T ;; oft) run_fw QwenOFT ;;
  all) run_sim; run_fw QwenGR00T; run_fw QwenOFT ;;
esac
exit $rc
