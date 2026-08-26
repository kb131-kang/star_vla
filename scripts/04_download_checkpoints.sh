#!/usr/bin/env bash
# Step 4: 기준 체크포인트 2종 + base VLM 다운로드, 체크포인트 config.yaml 의 base_vlm 경로를 로컬 절대경로로 교정,
#         configs/checkpoints.yaml 생성. 재실행 가능 (hf download 는 이어받기/스킵).
#   HF_CLI=<hf 실행파일>  (기본: PATH 의 hf; starVLA env 에 설치됨)
set -euo pipefail
WS="$(cd "$(dirname "$0")/.." && pwd)"
HF_CLI="${HF_CLI:-hf}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

PRETRAINED="$WS/playground/Pretrained_models"
CKPT_ROOT="$WS/checkpoints"
mkdir -p "$PRETRAINED" "$CKPT_ROOT" "$WS/playground/Datasets"

# starVLA 원본 리포는 상대경로 ./playground/... 를 참조하므로 워크스페이스 playground 로 심볼릭 링크
if [ ! -e "$WS/third_party/starVLA/playground" ]; then
  ln -s "$WS/playground" "$WS/third_party/starVLA/playground"
fi

echo "== base VLM: Qwen/Qwen3-VL-4B-Instruct (~8.9 GB)"
"$HF_CLI" download Qwen/Qwen3-VL-4B-Instruct --local-dir "$PRETRAINED/Qwen3-VL-4B-Instruct"

declare -A REPOS=(
  [groot]=StarVLA/Qwen3VL-GR00T-Bridge-RT-1
  [oft]=StarVLA/Qwen3VL-OFT-Bridge-RT-1
)
YAML="$WS/configs/checkpoints.yaml"
echo "# 자동 생성: scripts/04_download_checkpoints.sh ($(date +%F))" > "$YAML"
echo "# 실제 .pt 파일명은 ls 로 확인한 값 (문서의 steps_XXXXX 플레이스홀더 아님)" >> "$YAML"
for key in groot oft; do
  repo="${REPOS[$key]}"; name="${repo#*/}"; dst="$CKPT_ROOT/$name"
  echo "== $repo (~10 GB)"
  "$HF_CLI" download "$repo" --local-dir "$dst"
  pt=$(ls "$dst"/checkpoints/steps_*_pytorch_model.pt | head -1)
  # config.yaml 의 base_vlm 은 원저자 환경 경로(상대경로 또는 /mnt/petrelfs/...)이므로 로컬 절대경로로 교정 (원본은 .orig 로 보존)
  [ -f "$dst/config.yaml.orig" ] || cp "$dst/config.yaml" "$dst/config.yaml.orig"
  python3 - "$dst/config.yaml" "$PRETRAINED/Qwen3-VL-4B-Instruct" <<'PY'
import re, sys
p, vlm = sys.argv[1], sys.argv[2]
s = open(p).read()
s2 = re.sub(r"^(\s*base_vlm:\s*).*$", lambda m: m.group(1) + vlm, s, count=1, flags=re.M)
open(p, "w").write(s2)
print("  base_vlm ->", vlm)
PY
  steps=$(basename "$pt" | sed -E 's/steps_([0-9]+)_.*/\1/')
  printf "%s:\n  repo: %s\n  ckpt: %s\n  steps: %s\n  framework: %s\n" \
    "$key" "$repo" "$pt" "$steps" "$( [ $key = groot ] && echo QwenGR00T || echo QwenOFT )" >> "$YAML"
done
echo "== configs/checkpoints.yaml"; cat "$YAML"
