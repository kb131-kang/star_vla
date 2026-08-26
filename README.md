# star_vla — StarVLA 기반 VLA 개발 워크스페이스

계획서: [docs/starvla_개발환경_및_프로세스.md](docs/starvla_개발환경_및_프로세스.md) · 실행 프롬프트: [docs/claude_code_프롬프트_phase2_준비.md](docs/claude_code_프롬프트_phase2_준비.md) · 구축 기록: [SETUP_LOG.md](SETUP_LOG.md)

현재 상태: **Phase 0(인프라) + Phase 1(SimplerEnv CI) 완료, Phase 2(OFT vs GR00T) 시작 가능** — 자세한 수치는 SETUP_LOG 및 `reports/`.

| 기준 체크포인트 (WidowX, 4 runs × 24 eps) | spoon | carrot | stack | eggplant | 평균 | 문서값 | 지연/청크 |
|---|---|---|---|---|---|---|---|
| Qwen3VL-GR00T-Bridge-RT-1 (20K) | 76.0 | 52.1 | 20.8 | 97.9 | **61.7** | 65.3 (동봉 로그 63.3) | 53.7 ms |
| Qwen3VL-OFT-Bridge-RT-1 (5K) | 12.5 | 8.3 | 0.0 | 58.3 | **19.8** | 42.7 | 34.6 ms |

OFT 공개본은 starVLA#424 에서 저자가 결함을 인정(재학습 중)한 상태 — 수치는 체크포인트 문제이며 파이프라인 문제가 아님(GR00T 는 커뮤니티 재현 60–64 와 정합).

## 디렉토리

```
scripts/          재현 가능한 절차 (번호 순서대로 실행)
  00_preflight.sh            GPU/디스크/conda/Vulkan 점검
  01_clone_third_party.sh    starVLA(안정 브랜치 `starVLA`) + SimplerEnv 클론
  02_create_envs.sh          conda env 2개 (starVLA / simpler_env) — 절대 합치지 말 것
  03_smoke_tests.sh          SimplerEnv 빌드 + QwenGR00T/QwenOFT 프레임워크 스모크
  04_download_checkpoints.sh base VLM + 기준 체크포인트 2종 → configs/checkpoints.yaml
  eval_simplerenv.sh         정책 서버 + WidowX 평가 원클릭 (Step 5)
  run_ci.py                  CI: 서버 기동→지연 측정→4과제→reports/{run}.json + summary.csv (Step 6)
  07_prepare_datasets.sh     학습 데이터(bridge/fractal) 수령 + modality.json + 데이터로더 검증 (승인 후)
  phase2_launch.sh           Phase 2 클라우드 학습 런처 (초안)
  lib/                       env.sh(경로/과제 정의), policy_server.sh, run_task.sh
configs/          checkpoints.yaml, smoke_oxe_qwen3vl.yaml, phase2_matrix/{groot_20k,oft_65k}.yaml
reports/          CI 결과 (JSON per run + summary.csv 누적)
logs/             실행 로그 (eval/, ci/, smoke/) — git 제외
third_party/      starVLA, SimplerEnv 원본 (무수정, git 제외)
playground/       Pretrained_models/, Datasets/ (git 제외; third_party/starVLA/playground → 여기로 심볼릭 링크)
checkpoints/      HF 체크포인트 (git 제외)
patches/          원본 수정이 불가피할 때의 패치 (현재 없음)
```

## 처음부터 재현하기 (RTX 4090, Ubuntu 24.04, conda)

```bash
git clone <this repo> && cd star_vla
bash scripts/00_preflight.sh                 # 요구: NVIDIA 드라이버(CUDA≥12.4), libvulkan.so.1, conda
bash scripts/01_clone_third_party.sh
bash scripts/02_create_envs.sh all           # ~15분 (torch 2.6+cu124, flash-attn prebuilt, sapien 2.2.2)
bash scripts/04_download_checkpoints.sh      # ~29 GB (base VLM 8.9 + 체크포인트 2×10)
bash scripts/03_smoke_tests.sh all           # ✅ Env built / Finished ×2
# 축소 스모크 평가 → 전체 평가
CKPT=$(sed -n 's/^  ckpt: //p' configs/checkpoints.yaml | head -1)   # groot 항목
bash scripts/eval_simplerenv.sh "$CKPT" 6678 spoon 2 0
python3 scripts/run_ci.py --ckpt "$CKPT" --notes "baseline"
```

sudo 가 필요한 항목은 없다 (tmux 대신 nohup, ffmpeg 는 conda-forge 로 env 내부 설치).

## 평가 프로토콜 (WidowX Visual Matching, 원본 `start_simpler_env.sh` 값 그대로)

| 과제 | env | 성격 |
|---|---|---|
| spoon | PutSpoonOnTableClothInScene-v0 | 관대 |
| carrot | PutCarrotOnPlateInScene-v0 | 파지 다양성 |
| stack | StackGreenCubeOnYellowCubeBakedTexInScene-v0 | 정밀 정렬 (참고 지표 — 열쇠/밸브 정밀 과제의 프록시가 아님) |
| eggplant | PutEggplantInBasketScene-v0 (sink 카메라) | 관대 |

에피소드 24개/과제(`--obj-episode-range 0 24`), control 5 Hz, max 120 steps, 액션 앙상블 horizon 7. 4과제를 병렬 실행(기본)하면 GPU 1장에서 서버 1개를 공유한다.
`reports/summary.csv` 는 과제별 성공률을 항상 분리 기록하며, 추론 지연은 `scripts/latency_probe.py` 로 청크(16 스텝)당 ms 를 별도 측정한다.

## 한계 (계획서 1-3)

SimplerEnv 는 시각 도메인 갭·범용 조작 지표다. 정밀 정렬(열쇠 삽입 프록시)은 측정 범위 밖이며 Stack 수치는 참고용으로만 기록한다.
