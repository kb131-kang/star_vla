#!/usr/bin/env bash
# Step 2: conda 환경 2개 생성 (starVLA / simpler_env). 재실행 가능 (이미 있으면 재사용).
#   bash scripts/02_create_envs.sh [starvla|simpler|all]
set -euo pipefail
WS="$(cd "$(dirname "$0")/.." && pwd)"
source "$(conda info --base)/etc/profile.d/conda.sh"
TARGET="${1:-all}"
FLASH_ATTN_WHL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

create_starvla() {
  echo "=== [starVLA env] ==="
  conda env list | grep -qE "^starVLA\s" || conda create -n starVLA python=3.10 -y
  conda activate starVLA
  cd "$WS/third_party/starVLA"
  # torchvision==0.21.0 이 torch==2.6.0(+cu124) 을 함께 설치함
  pip install -r requirements.txt
  # flash-attn: 로컬에 nvcc 가 없으므로 공식 prebuilt wheel 사용 (torch2.6 / cu12 / py3.10 / cxx11abi=FALSE)
  pip install "$FLASH_ATTN_WHL"
  pip install -e .
  pip install "huggingface_hub[cli]"
  conda deactivate
}

create_simpler() {
  echo "=== [simpler_env env] ==="
  conda env list | grep -qE "^simpler_env\s" || conda create -n simpler_env python=3.10 -y
  conda activate simpler_env
  pip install numpy==1.24.4
  cd "$WS/third_party/SimplerEnv/ManiSkill2_real2sim" && pip install -e .
  cd "$WS/third_party/SimplerEnv" && pip install -e .
  # starVLA SimplerEnv 예제 클라이언트 의존성
  pip install tyro matplotlib mediapy websockets msgpack
  # 위 설치가 numpy 를 올렸을 수 있으므로 반드시 재고정 (시뮬 IK/pinocchio 호환)
  pip install numpy==1.24.4
  conda deactivate
}

case "$TARGET" in
  starvla) create_starvla ;;
  simpler) create_simpler ;;
  all) create_starvla; create_simpler ;;
  *) echo "usage: $0 [starvla|simpler|all]"; exit 1 ;;
esac
echo "=== done ($TARGET) ==="
