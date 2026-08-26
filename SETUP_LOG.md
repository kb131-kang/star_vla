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
