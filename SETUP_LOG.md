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

## Step 5. 평가 파이프라인 가동 및 기준선 재현  (`scripts/eval_simplerenv.sh <ckpt> <port> [tasks] [ep_end] [parallel]`)

### 5-0. 파이프라인 구조 (tmux 미설치 → nohup)
- `scripts/lib/policy_server.sh start|stop`: starVLA env 에서 `deployment/model_server/server_policy.py --use_bf16` 를 nohup 으로 띄우고 포트가 열릴 때까지 대기 (체크포인트 10 GB 로드 ≈ 10–14 s, 페이지캐시 덕분).
- `scripts/lib/run_task.sh`: simpler_env env 에서 `examples/SimplerEnv/eval_files/start_simpler_env.py` 를 원본 `start_simpler_env.sh` 와 **동일한 인자**(scene/robot/overlay/robot_init/24 episodes/120 steps/5 Hz)로 실행. `SimplerEnv_PATH` 역할은 `lib/env.sh` 의 `SIMPLER_DIR` (third_party/SimplerEnv 절대경로).
- 원본 `start_simpler_env.sh` 를 쓰지 않은 이유: 안정 브랜치에서 Spoon/Carrot/Stack 이 주석 처리되어 있고 `CUDA_VISIBLE_DEVICES` 미설정 시 `set -u` 로 종료됨 (원본 무수정 원칙에 따라 워크스페이스 드라이버로 대체).

### 5-1. 축소 스모크 평가 (GR00T, spoon × 2 episodes, 순차)
- 1차: 에피소드 1 이 120 스텝 완주(success) 후 영상 저장 단계에서 `RuntimeError: Program 'ffmpeg' is not found` → simpler_env 에 `conda install -c conda-forge ffmpeg` (sudo 불필요) + `run_task.sh` 가 env bin 을 PATH 앞에 추가. `02_create_envs.sh` 에 반영.
- 2차: ✅ spoon 2/2 성공, 서버 로드 12 s, 2 에피소드 25 s (**≈ 12–14 s/episode**, 120 스텝 기준 ≈ 110 ms/step: 서버 추론 + 시뮬 스텝 + 렌더).
- 부수 수정: `eval_simplerenv.sh` 요약 단계에서 `grep` 무매치 + `pipefail` 로 스크립트가 조용히 종료되던 버그 수정 (`|| true`).

### 5-2. 공식 기준 수치의 실체 (체크포인트에 동봉된 `checkpoints/*_infer_<task>.log.run{1..4}` 분석)
공개 수치(65.3 / 42.7)는 **과제당 24 episodes × 4 runs 평균**이다. 동봉 로그에서 읽은 run 별 성공률:

| 체크포인트 | 과제 | run1 | run2 | run3 | run4 | 4-run 평균 |
|---|---|---|---|---|---|---|
| GR00T 20K | spoon | 75.0 | 83.3 | 70.8 | 70.8 | **75.0** |
| | carrot | 62.5 | 54.2 | 58.3 | 62.5 | **59.4** |
| | stack | 20.8 | 20.8 | 12.5 | 20.8 | **18.8** |
| | eggplant | 100 | 100 | 100 | 100 | **100** |
| | 평균 | 64.6 | 64.6 | 60.4 | 63.5 | **63.3** (문서 65.3) |
| OFT 5K | spoon | 33.3 | 33.3 | 25.0 | 33.3 | **31.3** |
| | carrot | 50.0 | 50.0 | 54.2 | 50.0 | **51.0** |
| | stack | 0 | 0 | 0 | 0 | **0** |
| | eggplant | 100 | 54.2 | 100 | 100 | **88.5** |
| | 평균 | 45.8 | 34.4 | 44.8 | 45.8 | **42.7** (문서 42.7 과 일치) |

→ 단일 24-episode 런의 과제별 표준편차는 ~8–10%p, 평균 기준 ~±4%p. 따라서 **±3%p 합격 판정은 4-run 평균으로 수행**하는 것이 공식 프로토콜과 정합. `run_ci.py --runs 4` 로 구현.

### 5-3. GR00T 전체 평가 (WidowX 4과제 × 24 episodes) — 단일 런 2회
| 런 | 모드 | spoon | carrot | stack | eggplant | 평균 | 벽시계 |
|---|---|---|---|---|---|---|---|
| 1 | 4과제 병렬 | 62.5 | 45.8 | 20.8 | 100 | **57.3** | 과제당 ~658 s, 총 688 s (서버 로드 10 s) |
| 2 | 순차 | 79.2 | 54.2 | 12.5 | 95.8 | **60.4** | spoon 304 / carrot 297 / stack 287 / eggplant 299 s, 총 1199 s |
| 공식 4-run 평균 | | 75.0 | 59.4 | 18.8 | 100 | 63.3 (문서 65.3) | |

- 런 2 의 과제별 수치는 모두 공식 run1–4 범위 안. 런 1 은 spoon/carrot 이 공식 최소치보다 낮았으나 두 런 모두 stack/eggplant 는 정상 → 파이프라인 결함이 아니라 24-episode 표본 분산으로 판단. 리스크 표 점검: numpy 1.24.4 ✅, 체크포인트 파일명 steps_20000 ✅, Vulkan ✅.
- 벽시계: 병렬 11.5 min vs 순차 20 min (1.75×). 병렬은 원본 `start_simpler_env.sh` 도 사용하는 방식(v1 3과제 백그라운드 동시 실행)이므로 CI 기본값으로 채택.
- 정식 판정은 5-4 의 `run_ci.py --runs 4`(공식 프로토콜) 결과로 수행.

### 5-4. GR00T 공식 프로토콜 평가 (`python3 scripts/run_ci.py --ckpt <groot.pt> --runs 4`)  → `reports/20260826_201104_QwenGR00T_20000.json`
| 과제 | run1 | run2 | run3 | run4 | **4-run 평균** | 공식 동봉 로그 평균 | 커뮤니티 재현 (#191 / #424) |
|---|---|---|---|---|---|---|---|
| spoon | 83.3 | 79.2 | 75.0 | 66.7 | **76.0** | 75.0 | 77.9 / – |
| carrot | 45.8 | 54.2 | 58.3 | 50.0 | **52.1** | 59.4 | 54.2 / – |
| stack | 29.2 | 16.7 | 16.7 | 20.8 | **20.8** | 18.8 | 25.0 / – |
| eggplant | 100 | 100 | 100 | 91.7 | **97.9** | 100 | 100 / – |
| 평균 | | | | | **61.7** | 63.3 | 64.3 / 60.2 |

- 추론 지연 (`scripts/latency_probe.py`, 224×224 단일 이미지, 16-step 청크): **53.7 ms/chunk** (p50 53.6, p95 54.6), bf16, 4090.
- 벽시계: 서버 로드 18 s, 4 runs × 4과제 병렬 = **2720 s (45 min)**; 1 run(24×4) ≈ 680 s.
- 판정: 문서값 65.3 ± 3 → [62.3, 68.3] 에 **0.6%p 미달** (carrot −7%p 가 주원인). 체크포인트 동봉 공식 로그 평균 63.3 ± 3 → [60.3, 66.3] 에는 **포함**. 커뮤니티 재현 60.2 (#424, A100) / 64.3 (#191) 과도 정합 → 파이프라인은 정상 범위로 판단.
- 리스크 표 점검 결과: numpy 1.24.4 ✅ / 파일명 steps_20000 ✅ / Vulkan ✅. GitHub Issues 검색 결과:
  - **#424** (2026-07~08, open): 커뮤니티가 동일 안정 브랜치(3422b9f)+SimplerEnv(06accac)로 재현 — Qwen3VL-GR00T 60.16%, **Qwen3VL-OFT(5K) 26.04%** (spoon 20.8 / carrot 12.5 / stack 8.3 / eggplant 62.5). 저자(2026-08-07): "업로드된 OFT 체크포인트가 잘못된 것이었음, 재학습 후 재업로드 예정" → **OFT 공개본 42.7 은 재현 불가 가능성 높음** (본 세션 결과와 함께 판단).
  - **#191** (closed): HF GR00T 체크포인트 평가값 64.27 (96 eps). 자체 학습본 재현 실패(그리퍼 회전 고정) 논의 — Phase 2 학습 시 참고 (transformers 4.57.0 고정, 그리퍼 임계값 정합).
  - **#324**: OFT 5K 업로드 오류 지적 (위와 동일 맥락).

### 5-5. OFT 공식 프로토콜 평가 (`run_ci.py --ckpt <oft.pt> --runs 4`) → `reports/20260826_205713_QwenOFT_5000.json`
| 과제 | run1–4 (모두 동일) | **평균** | 공식 동봉 로그 평균 | 커뮤니티 재현 (#424) |
|---|---|---|---|---|
| spoon | 12.5 ×4 | **12.5** | 31.3 | 20.8 |
| carrot | 8.3 ×4 | **8.3** | 51.0 | 12.5 |
| stack | 0 ×4 | **0.0** | 0 | 8.3 |
| eggplant | 58.3 ×4 | **58.3** | 88.5 | 62.5 |
| 평균 | | **19.8** | 42.7 | 26.0 |

- 추론 지연 **34.6 ms/chunk** (OFT 1-pass MLP 헤드; GR00T 53.7 ms 대비 0.64×). 벽시계 1798 s (4 runs), 1 run ≈ 450 s (에피소드가 빨리 실패해 GR00T 보다 짧음).
- **4 run 이 완전히 동일** → OFT 는 결정론적 (샘플링 없는 회귀 헤드 + 결정론적 시뮬). 반면 GR00T 는 flow-matching 노이즈로 run 간 분산 존재. OFT 는 `--runs 1` 로 충분.
- 판정: 목표 42.7 ± 3 에서 **−22.9%p 로 크게 이탈**. 리스크 표 순서 점검(numpy/파일명/Vulkan) 모두 정상이고 같은 파이프라인에서 GR00T 가 정상 재현되므로 파이프라인 문제가 아님. GitHub #424 의 커뮤니티 재현(26.0%)과 저자 답변(2026-08-07: "업로드된 OFT 체크포인트가 잘못됨, 재학습 후 재업로드 예정")과 정합 → **공개 OFT 체크포인트 자체의 문제**로 결론. 사용자 보고 사항.
- Phase 2 에의 함의: (a) "출발점 선정" 목적의 OFT 공개본 수치는 신뢰 불가 → 비교표에 "공개본 결함(#424)" 명기. (b) 패러다임 비교는 어차피 스텝 통제 재학습(oft_65k.yaml)으로 수행하므로 계획 변경 불필요. (c) HF 리포에 재학습본이 올라오면 `scripts/04_download_checkpoints.sh` 재실행 + `run_ci.py` 로 재평가.

**Step 5 합격 판정**: 두 체크포인트 리포트가 `reports/` 에 저장됨 ✅. GR00T 61.7 (동봉 공식 로그 63.3 ± 3 안 / 문서 65.3 ± 3 에 0.6%p 미달) — 경계선 합격, OFT 19.8 — 공개본 결함으로 목표 미달 (외부 원인).
