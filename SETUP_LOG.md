# SETUP_LOG — StarVLA 환경 구축 (Phase 0 → Phase 2 준비)

- 머신: RTX 4090 24GB / Ubuntu 24.04.4 / 32 CPU / 61 GB RAM
- 워크스페이스: `/media/ys/data2/starvla` (5.5 TB 디스크, 여유 4.9 TB).
  프롬프트의 `~/vla-workspace/` 는 이 경로로의 심볼릭 링크로 제공 (`ln -sfn /media/ys/data2/starvla ~/vla-workspace`).
  이유: 홈(`/`)은 188 GB 여유뿐이라 체크포인트(각 ~10 GB)+데이터를 두기엔 data2 가 적합하고, 사용자가 이미 이 경로에 git repo 를 만들어 둔 상태였음.
- 작업일: 2026-08-26

---

## Step 0. 사전 점검  (`scripts/00_preflight.sh`, 로그 `logs/preflight_20260826_192450.log`)

| 항목 | 결과 |
|---|---|
| GPU | NVIDIA GeForce RTX 4090, 24564 MiB |
| 드라이버 / CUDA | 580.126.09 / CUDA 13.0 (nvidia-smi 기준). **시스템 nvcc 없음** → flash-attn 은 prebuilt wheel 사용 예정 |
| 디스크 | `/media/ys/data2` 5.5T 중 **4.9T 여유** (체크포인트+평가 50GB 조건 충족, 학습 데이터 전체 수령도 가능) / `/` 188G 여유 |
| conda | 26.1.1 (`/home/ys/miniconda3`), 기존 env: base, gmr, teleop_operator, twist2 (건드리지 않음) |
| huggingface-cli | 미설치·미로그인 (`~/.cache/huggingface/token` 없음). 필요한 리포는 모두 public 이라 로그인 불필요. CLI 는 starVLA env 에 설치 |
| 인터넷 | github.com 200 / huggingface.co 200 |
| Vulkan | `libvulkan.so.1` 존재 (`/lib/x86_64-linux-gnu/libvulkan.so.1`), `/usr/share/vulkan/icd.d/nvidia_icd.json` 존재 (api 1.4.312) |
| tmux | **미설치** (sudo 필요). 원칙 5(b)에 따라 설치하지 않고, 평가 자동화 스크립트는 tmux 대신 `nohup` 백그라운드 프로세스 + 로그 파일 방식으로 작성 |
| 기타 | `~/.bashrc` 에 `export CUDA_HOME=$CONDA_PREFIX` 가 있음 — 소스 빌드가 필요한 패키지에서 잘못된 CUDA_HOME 을 볼 수 있으니 주의 (이번엔 prebuilt wheel 로 회피) |

참고 (사전 조사, HF API):
- `StarVLA/Qwen3VL-GR00T-Bridge-RT-1`: `checkpoints/steps_20000_pytorch_model.pt` 9.98 GB
- `StarVLA/Qwen3VL-OFT-Bridge-RT-1`: `checkpoints/steps_5000_pytorch_model.pt` 9.79 GB (OFT 공개본은 **5K 스텝** — 문서의 "미완성 학습" 설명과 일치)
- `Qwen/Qwen3-VL-4B-Instruct` (체크포인트 config.yaml 이 base_vlm 으로 참조) 8.9 GB
- `IPEC-COMMUNITY/bridge_orig_lerobot` 5.1 GB (99,673 파일) / `fractal20220817_data_lerobot` 3.7 GB (99,733 파일) / `StarVLA/LLaVA-OneVision-COCO` 2.6 GB

## Step 1. 워크스페이스 구성  (`scripts/01_clone_third_party.sh`)

- 디렉토리: `scripts/ configs/ logs/ reports/ patches/ third_party/ docs/`
- `third_party/starVLA` ← `https://github.com/starVLA/starVLA` **branch `starVLA`** (안정 브랜치, shallow clone)
- `third_party/SimplerEnv` ← `https://github.com/simpler-env/SimplerEnv` + submodule `ManiSkill2_real2sim`
- `.gitignore`: third_party/, playground/, checkpoints/, datasets/, *.pt, wandb/, logs/*.log
- 계획서·프롬프트 사본은 `docs/` 에 위치

리포 구조에서 확인한, 프롬프트/계획서 경로와 다른 점 (안정 브랜치 기준):
- 프레임워크 파일 위치: `starVLA/model/framework/VLM4A/QwenGR00T.py`, `.../VLM4A/QwenOFT.py` (프롬프트의 `starVLA/model/framework/QwenXXX.py` 가 아님)
- SimplerEnv 검증/실행 스크립트 위치: `examples/SimplerEnv/eval_files/test_your_simplerEnv.py`, `examples/SimplerEnv/eval_files/start_simpler_env.sh`
- `start_simpler_env.sh` 는 안정 브랜치에서 WidowX 3과제가 주석 처리되어 Eggplant 만 실행되고, `CUDA_VISIBLE_DEVICES` 미설정 시 `set -u` 로 즉시 종료됨 → 원본은 수정하지 않고 워크스페이스 `scripts/` 에 자체 드라이버를 작성
- `data_mix: bridge_rt_1` 이 기대하는 데이터 디렉토리 이름: `bridge_orig_1.0.0_lerobot`, `fractal20220817_data_0.1.0_lerobot` (HF 리포명과 다름 — Step 7 에서 이 이름으로 배치)

## Step 2. conda 환경 2개 구축  (`scripts/02_create_envs.sh [starvla|simpler|all]`, 로그 `logs/02_env_all.log`)

| env | 내용 | 검증 |
|---|---|---|
| `starVLA` | py3.10, `pip install -r requirements.txt` (torch **2.6.0+cu124**, torchvision 0.21.0, transformers 4.57.0, numpy 1.26.4, deepspeed 0.16.9), flash-attn 2.7.4.post1 (**prebuilt wheel**: cu12/torch2.6/cp310/cxx11abiFALSE — 로컬에 nvcc 없음), `pip install -e third_party/starVLA`, `huggingface_hub[cli]` | `torch.cuda.is_available()==True`, device RTX 4090, `import starVLA, flash_attn` OK |
| `simpler_env` | py3.10, numpy==1.24.4 → `ManiSkill2_real2sim` -e → `SimplerEnv` -e → tyro/matplotlib/mediapy/websockets/msgpack → `opencv-python<5`, `setuptools<81`, numpy==1.24.4 재고정 | `import simpler_env` OK (25 envs), sapien 2.2.2, gymnasium 0.29.1, **numpy 1.24.4**, `pip check` 이상 없음 |

발생한 문제와 해결:
1. **conda 캐시 경합**: 두 `conda create` 를 동시에 실행하자 `pkgs/*.conda.partial` 경합으로 둘 다 실패 → 순차 실행으로 재시도 (스크립트 `all` 모드는 순차).
2. **사용자 site-packages 오염**: `~/.local/lib/python3.10/site-packages` 에 cv2(4.11)/h5py/tqdm 이 있어 conda py3.10 env 에서 이를 먼저 import 하고 pip 도 "already satisfied" 로 건너뜀 → 모든 스크립트에서 `PYTHONNOUSERSITE=1` 강제 (`scripts/lib/env.sh`, `02_create_envs.sh`) 후 재설치.
3. **opencv-python 5.0.0** 이 설치되어 numpy>=2 요구 → `opencv-python<5` (4.11.0.86) 로 고정.
4. **sapien 2.2.2 가 `pkg_resources` import** → setuptools 81+ 에서 제거됨 → `setuptools<81` (80.10.2) 로 고정.
5. CUDA/torch 불일치 경고: 없음 (드라이버 CUDA 13.0 ≥ 런타임 12.4, 정상). 단 `QWen3.py` 는 `attn_implementation="sdpa"` 를 하드코딩하므로 flash-attn 은 Qwen3-VL 경로에서 실제로 사용되지 않음 (설치는 무해, 다른 백본용).

## Step 3. 공식 검증 3종  (`scripts/03_smoke_tests.sh [sim|groot|oft|all]`, 로그 `logs/smoke/`)

| # | 검증 | 결과 |
|---|---|---|
| 1 | `examples/SimplerEnv/eval_files/test_your_simplerEnv.py` (simpler_env) | ✅ `Env built successfully` — Vulkan 에러 없음 (`libvulkan.so.1` + nvidia ICD 기존재). `GLFW error: X11: Failed to open display` 는 `DISPLAY=""` 헤드리스 실행 시 나오는 무해한 경고 |
| 2a | `starVLA/model/framework/VLM4A/QwenGR00T.py --config_yaml configs/smoke_oxe_qwen3vl.yaml` (starVLA) | ✅ forward `Action Loss: 1.441`, predict_action 성공, `Finished` |
| 2b | `starVLA/model/framework/VLM4A/QwenOFT.py --config_yaml configs/smoke_oxe_qwen3vl.yaml` | ✅ `Action Loss (with state): 0.751`, `Predicted Action shape: (1, 16, 7)`, `Finished` |
| 3 | `starVLA/dataloader/lerobot_datasets.py` 데이터로더 검증 | ⏸ **연기** — 학습 데이터(bridge/fractal LeRobot) 필요. Step 7 데이터 수령 후 `scripts/07_prepare_datasets.sh` 에서 수행 |

메모:
- `configs/smoke_oxe_qwen3vl.yaml` = 원본 `examples/SimplerEnv/train_files/starvla_cotrain_oxe.yaml` 에서 `base_vlm` 만 로컬 `playground/Pretrained_models/Qwen3-VL-4B-Instruct` 절대경로로 바꾼 사본 (원본 yaml 은 Qwen2.5-VL-3B 상대경로를 가리켜 그대로는 실행 불가).
- base VLM `Qwen/Qwen3-VL-4B-Instruct` 는 `playground/Pretrained_models/` 에 두고, 원본 리포의 `third_party/starVLA/playground` → 워크스페이스 `playground/` 심볼릭 링크로 연결 (원본 리포 무수정).
- DINOv2 (`dinov2_vits14`) 는 torch.hub 에서 자동 다운로드됨 (`~/.cache/torch/hub`).
- "NotImplementedError: Framework ... is not implemented" 는 발생하지 않음 (안정 브랜치는 `build_framework()` 에서 VLM4A/ 하위 모듈을 자동 import).

## Step 4. 기준 체크포인트 2종 다운로드  (`scripts/04_download_checkpoints.sh`, 로그 `logs/04_download.log`)

| 항목 | 경로 | 크기 |
|---|---|---|
| base VLM `Qwen/Qwen3-VL-4B-Instruct` | `playground/Pretrained_models/Qwen3-VL-4B-Instruct/` | 8.9 GB |
| `StarVLA/Qwen3VL-GR00T-Bridge-RT-1` | `checkpoints/Qwen3VL-GR00T-Bridge-RT-1/checkpoints/steps_20000_pytorch_model.pt` | 9.98 GB |
| `StarVLA/Qwen3VL-OFT-Bridge-RT-1` | `checkpoints/Qwen3VL-OFT-Bridge-RT-1/checkpoints/steps_5000_pytorch_model.pt` | 9.79 GB |

- 실제 파일명은 `ls` 로 확인: GR00T = **steps_20000**, OFT = **steps_5000** (문서의 `steps_XXXXX`/`steps_50000` 은 플레이스홀더). `configs/checkpoints.yaml` 에 절대경로로 기록.
- 두 체크포인트의 `config.yaml` 은 `framework.qwenvl.base_vlm` 이 원저자 환경 경로(GR00T: 상대경로 `./playground/...`, OFT: 절대경로 `/mnt/petrelfs/yejinhui/...`)를 가리켜 그대로는 로드 불가
  → 스크립트가 로컬 절대경로 `playground/Pretrained_models/Qwen3-VL-4B-Instruct` 로 교정 (`config.yaml.orig` 로 원본 보존). 가중치는 `.pt` 에서 전부 로드되므로 base_vlm 은 아키텍처/프로세서 로딩용.
- `dataset_statistics.json` 의 unnorm 키: `oxe_bridge`, `oxe_rt1` (WidowX 평가는 `oxe_bridge` 사용, 클라이언트 기본값).
- HF 로그인 없이(public) 다운로드. `hf_transfer` 는 최신 `hf` CLI 에서 무시됨(Xet 기반) — 경고만 출력.
- 다운로드는 starVLA env 구축과 병행하기 위해 scratchpad 의 임시 venv(`huggingface_hub[cli]`)로 실행 (`HF_CLI=` 환경변수로 지정). 재현 시엔 starVLA env 의 `hf` 사용.
