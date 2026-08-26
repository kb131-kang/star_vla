#!/usr/bin/env bash
# Step 7-1/7-2: 학습 데이터 수령 + modality.json 배치 + 데이터로더 검증.
#   ⚠️ 사용자 승인 후 실행 (총 ~8.8 GB: bridge 5.1 GB / fractal 3.7 GB, 각 ~10만 파일; 옵션 vlm 2.6 GB)
#   bash scripts/07_prepare_datasets.sh [bridge|fractal|vlm|check|all]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib/env.sh"
HF_CLI="${HF_CLI:-$CONDA_BASE/envs/starVLA/bin/hf}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
ROOT="$WS/playground/Datasets/OXE_LEROBOT_DATASET"   # config 의 data_root_dir
TF="$STARVLA_DIR/examples/SimplerEnv/train_files"
what="${1:-all}"
dl_bridge() {  # data_mix bridge_rt_1 이 기대하는 디렉토리 이름으로 저장 (HF 리포명과 다름)
  "$HF_CLI" download IPEC-COMMUNITY/bridge_orig_lerobot --repo-type dataset --local-dir "$ROOT/bridge_orig_1.0.0_lerobot"
  cp "$TF/modality.json" "$ROOT/bridge_orig_1.0.0_lerobot/meta/modality.json"; echo "bridge: modality.json placed"
}
dl_fractal() {
  "$HF_CLI" download IPEC-COMMUNITY/fractal20220817_data_lerobot --repo-type dataset --local-dir "$ROOT/fractal20220817_data_0.1.0_lerobot"
  cp "$TF/fractal_modality.json" "$ROOT/fractal20220817_data_0.1.0_lerobot/meta/modality.json"; echo "fractal: modality.json placed"
}
dl_vlm() {  # Phase 3 공동학습용 (Phase 2 에는 불필요)
  "$HF_CLI" download StarVLA/LLaVA-OneVision-COCO --repo-type dataset --local-dir "$WS/playground/Datasets/LLaVA-OneVision-COCO"
}
check() {  # Step 3-3 데이터로더 검증 (starVLA env)
  cd "$STARVLA_DIR"
  "$STARVLA_PY" starVLA/dataloader/lerobot_datasets.py --config_yaml "$WS/configs/phase2_matrix/groot_20k.yaml" 2>&1 | tee "$WS/logs/smoke/dataloader.log" | tail -20
}
mkdir -p "$ROOT" "$WS/logs/smoke"
case "$what" in
  bridge) dl_bridge ;; fractal) dl_fractal ;; vlm) dl_vlm ;; check) check ;;
  all) dl_bridge; dl_fractal; check ;;
esac
