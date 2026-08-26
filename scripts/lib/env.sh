#!/usr/bin/env bash
# 공통 경로/인터프리터 정의. 다른 스크립트에서 `source scripts/lib/env.sh`.
WS="${WS:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CONDA_BASE="${CONDA_BASE:-$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")}"
STARVLA_PY="${STARVLA_PY:-$CONDA_BASE/envs/starVLA/bin/python}"
SIM_PY="${SIM_PY:-$CONDA_BASE/envs/simpler_env/bin/python}"
STARVLA_DIR="${STARVLA_DIR:-$WS/third_party/starVLA}"
SIMPLER_DIR="${SIMPLER_DIR:-$WS/third_party/SimplerEnv}"
export WS CONDA_BASE STARVLA_PY SIM_PY STARVLA_DIR SIMPLER_DIR
export PYTHONNOUSERSITE=1   # ~/.local 사용자 패키지 차단
export PYTHONPATH="$STARVLA_DIR:${PYTHONPATH:-}"

# WidowX 4과제 정의 (third_party/starVLA/examples/SimplerEnv/eval_files/start_simpler_env.sh 의 값 그대로)
# name -> env_name|scene|robot|overlay(SimplerEnv 상대경로)|robot_init_x|robot_init_y
declare -Ag TASKS=(
  [spoon]="PutSpoonOnTableClothInScene-v0|bridge_table_1_v1|widowx|ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png|0.147|0.028"
  [carrot]="PutCarrotOnPlateInScene-v0|bridge_table_1_v1|widowx|ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png|0.147|0.028"
  [stack]="StackGreenCubeOnYellowCubeBakedTexInScene-v0|bridge_table_1_v1|widowx|ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png|0.147|0.028"
  [eggplant]="PutEggplantInBasketScene-v0|bridge_table_1_v2|widowx_sink_camera_setup|ManiSkill2_real2sim/data/real_inpainting/bridge_sink.png|0.127|0.06"
)
TASK_ORDER=(spoon carrot stack eggplant)
