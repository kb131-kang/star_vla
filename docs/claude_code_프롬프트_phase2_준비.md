# [Claude Code 프롬프트] StarVLA 환경 구축 — Phase 2 시작 가능 상태까지

아래 내용을 첨부 파일 `starvla_개발환경_및_프로세스.md`와 함께 Claude Code에 붙여넣기.

---

첨부한 `starvla_개발환경_및_프로세스.md`가 전체 개발 계획서다. 이번 세션의 목표는 이 중
**Phase 0(공통 인프라)과 Phase 1(SimplerEnv CI)을 이 머신(RTX 4090, Ubuntu)에서 완료**하고,
**Phase 2(OFT vs GR00T 공정 비교)를 즉시 시작할 수 있는 상태**를 만드는 것이다.
Phase 2의 클라우드 학습 실행 자체는 이번 범위가 아니다 (config 초안 작성까지만).

## 작업 원칙 (전 단계 공통)

1. **단계별 진행 + 검증 게이트**: 아래 각 Step은 명시된 [합격 기준]을 통과해야만 다음 Step으로
   진행한다. 실패 시 원인 분석 → 수정 → 재검증하고, 같은 단계에서 3회 실패하면 멈추고
   상황을 나에게 보고하라.
2. **git 커밋 체크포인트**: 워크스페이스 repo를 만들어 각 Step 완료 시마다 커밋한다.
   메시지 형식: `phase0-2: conda envs created (starVLA, simpler_env)`.
   문제가 생기면 마지막 정상 커밋과의 diff로 디버깅한다.
3. **원본 불변**: starVLA·SimplerEnv 원본 리포는 third_party/ 아래 두고 직접 수정하지 않는다.
   수정이 불가피하면 변경분을 patches/*.patch로 뽑아 워크스페이스에 커밋하고 SETUP_LOG에 사유 기록.
4. **기록**: `SETUP_LOG.md`에 각 Step의 실행 명령, 결과, 발생 에러와 해결책을 기록하고 커밋에 포함.
5. **확인 후 실행**: (a) 10GB 이상 다운로드, (b) sudo 필요 작업, (c) 기존 conda env 삭제·변경은
   실행 전 반드시 나에게 확인받는다.
6. **재실행 가능성**: 일회성 셸 명령으로 끝내지 말고 모든 절차를 scripts/ 아래 스크립트로 남긴다.
7. **보고 형식**: 각 Step 완료 시 4줄 요약으로 보고 — (완료 단계 / 합격 근거 / 다음 단계 / 특이사항).

## Step 0. 사전 점검

다음을 확인해 SETUP_LOG.md에 기록하라:
- nvidia-smi로 GPU/드라이버/CUDA 버전 (RTX 4090 기대)
- 디스크 여유 공간 — 체크포인트+평가용 최소 50GB, 학습 데이터까지 받으려면 수백 GB.
  현재 여유 공간을 보고하고 데이터 전략(로컬 수령 vs 클라우드 연기)은 Step 7에서 내가 결정한다.
- conda 설치 여부, huggingface-cli 로그인 상태, 인터넷 접속
- libvulkan.so.1 존재 여부 (SimplerEnv 렌더링 의존성)

[합격 기준] 위 항목이 모두 SETUP_LOG.md에 기록됨.
[커밋] `phase0-0: preflight check`

## Step 1. 워크스페이스 구성

- `~/vla-workspace/` 생성 후 git init
- 디렉토리: `scripts/ configs/ logs/ reports/ patches/ third_party/`
- third_party/에 starVLA 클론 (**안정 브랜치 `starVLA` 사용** — starVLA_dev는 불안정 브랜치이므로 금지)
- .gitignore: third_party/, 체크포인트(*.pt), 데이터셋, wandb/ 제외
- 계획서 md와 이 프롬프트 사본을 docs/에 배치

[합격 기준] 구조 생성 완료, `git log`에 초기 커밋 존재.
[커밋] `phase0-1: workspace scaffold`

## Step 2. conda 환경 2개 구축

계획서 Phase 0-2를 그대로 따른다. **두 환경을 절대 하나로 합치지 말 것**
(모델 추론과 시뮬레이션의 패키지 버전이 충돌하는 것이 공식 문서에 명시된 사항이다):

- env `starVLA`: python 3.10 → third_party/starVLA의 requirements.txt 설치
- env `simpler_env`: 공식 SimplerEnv 리포를 third_party/SimplerEnv에 클론·설치 후
  `pip install tyro matplotlib mediapy websockets msgpack` 및 `pip install numpy==1.24.4`
  (numpy 다운그레이드는 시뮬 호환성 필수 사항 — 건너뛰지 말 것)
- 설치 로그에서 CUDA/torch 버전 불일치 경고가 나오면 SETUP_LOG에 기록하고 해결 후 진행

[합격 기준] 각 env에서 핵심 패키지 import 성공
(starVLA: torch+CUDA available / simpler_env: simpler_env, numpy==1.24.4 확인).
[커밋] `phase0-2: conda envs created`

## Step 3. 공식 검증 3종 (순서 고정)

1. simpler_env 환경: `python examples/SimplerEnv/test_your_simplerEnv.py`
   → "✅ Env built successfully" 확인. Vulkan 에러 발생 시 ManiSkill 설치 가이드의
   Vulkan 섹션 절차를 적용하고 해결 과정을 SETUP_LOG에 기록.
2. starVLA 환경: 프레임워크 스모크 테스트 2종
   `python starVLA/model/framework/QwenGR00T.py --config_yaml examples/SimplerEnv/train_files/starvla_cotrain_oxe.yaml`
   `python starVLA/model/framework/QwenOFT.py --config_yaml examples/SimplerEnv/train_files/starvla_cotrain_oxe.yaml`
   ("NotImplementedError: Framework ... is not implemented" 에러는 알려진 이슈 —
   해당 프레임워크 파일 단독 실행으로 등록을 트리거한 뒤 재시도)
3. 데이터로더 검증(`starVLA/dataloader/lerobot_datasets.py`)은 학습 데이터가 필요하므로
   Step 7에서 데이터 수령 후 수행한다. 이 연기를 SETUP_LOG에 명시.

[합격 기준] 1, 2 통과 (스모크 테스트가 fake 데이터로 forward/predict_action 성공).
[커밋] `phase0-3: smoke tests passed`

## Step 4. 기준 체크포인트 2종 다운로드 (Phase 2의 비교 대상)

- `StarVLA/Qwen3VL-GR00T-Bridge-RT-1`
- `StarVLA/Qwen3VL-OFT-Bridge-RT-1`
- huggingface-cli download로 받고, 실제 `checkpoints/steps_*.pt` 파일명을 확인해
  `configs/checkpoints.yaml`에 절대경로로 기록 (문서 예시의 steps_XXXXX는 플레이스홀더이므로
  ls로 실제 파일명을 확인할 것)

[합격 기준] 두 .pt 파일 존재 + configs/checkpoints.yaml에 경로 기록.
[커밋] `phase0-4: baseline checkpoints downloaded`

## Step 5. 평가 파이프라인 가동 및 기준선 재현 (Phase 1-1)

- 2터미널 워크플로를 tmux 기반 단일 스크립트로 자동화:
  `scripts/eval_simplerenv.sh <ckpt_path> <port>`
  - 세션 A: starVLA env에서 run_policy_server.sh (ckpt 경로·포트 파라미터화)
  - 세션 B: simpler_env env에서 start_simpler_env.sh (SimplerEnv_PATH 변수를
    third_party/SimplerEnv 절대경로로 설정했는지 확인)
- **반드시 축소 스모크 평가 먼저**: 과제 1개 × 소수 에피소드로 서버-클라이언트 관통 확인.
  성공 후 전체 평가로 진행.
- GR00T 체크포인트 전체 평가 (WidowX 4과제): 목표 평균 65.3 ± 3%p
- OFT 체크포인트 전체 평가: 목표 평균 ~42.7 ± 3%p
  (이 수치가 낮은 것은 정상이다 — 학습량이 미통제된 공개본이며, 이 사실이 Phase 2의 존재 이유다)
- 전체 평가의 벽시계 소요 시간을 과제별로 logs/에 기록 (이후 CI 주기 설계의 근거)
- 목표 범위 이탈 시: 계획서 Phase 5 리스크 표의 순서(numpy 버전 → 체크포인트 파일명 →
  Vulkan)로 점검하고, 해결 안 되면 starVLA GitHub Issues 검색 후 나에게 보고

[합격 기준] 두 체크포인트의 평가 리포트가 reports/에 저장되고 목표 범위 내.
[커밋] `phase1-1: baselines reproduced (groot=6X.X, oft=4X.X)`

## Step 6. CI 스크립트화 (Phase 1-2)

- `scripts/run_ci.py` 작성: 체크포인트 경로를 입력받아 → 서버 기동 → WidowX 4과제 평가 →
  결과를 `reports/{run_id}.json` + 누적 `reports/summary.csv`에 적재 후 서버 종료
- 기록 필수 항목: **과제별 성공률(Spoon/Carrot/Stack/Eggplant 분리)**, 평균, 청크당 추론 지연(ms),
  체크포인트 식별자, 평가 일시. 평균만 기록하는 구현은 불합격.
- 두 체크포인트에 대해 run_ci.py를 재실행하여 Step 5와 동일한 수치가 재현되는지 확인

[합격 기준] run_ci.py 2회 실행 결과가 summary.csv에 정상 누적, Step 5 수치 재현.
[커밋] `phase1-2: CI script operational`

## Step 7. Phase 2 준비 패키지

1. **학습 데이터**: `IPEC-COMMUNITY/bridge_orig_lerobot`과 `fractal20220817_data_lerobot`의
   용량을 먼저 조회해 보고하고, 내 승인 후 다운로드 (디스크 부족 시 옵션 제시:
   bridge만 로컬 / 전체는 클라우드 스토리지로 연기). 수령 시 각 데이터셋 meta/에
   공식 modality.json 배치 (bridge용·fractal용이 다름 — 리포 제공 파일 사용).
2. 데이터 수령 완료 시 Step 3-3의 데이터로더 검증 수행:
   `python starVLA/dataloader/lerobot_datasets.py --config_yaml ...` 배치 로딩 성공 확인
3. `configs/phase2_matrix/`에 재학습 config 초안 2개 작성 (실행은 하지 않음):
   - 공통: base_vlm=`StarVLA/Qwen3-VL-4B-Instruct-Action`, data_mix=`bridge_rt_1`,
     학습률 그룹(qwen_vl_interface 1e-5 / action_model 1e-4), bf16,
     클라우드 8GPU 가정의 accelerate+deepspeed zero2 설정
   - groot_20k.yaml: QwenGR00T, 중간 평가 10K/20K/40K
   - oft_65k.yaml: QwenOFT, 중간 평가 20K/40K/65K
4. SETUP_LOG.md 최종 정리 + 워크스페이스 README.md (제3자가 재현 가능한 절차 요약)

[합격 기준] git log에 단계별 커밋 이력 완비, run_ci.py 동작, phase2 config 2종 존재,
(데이터 수령 시) 데이터로더 검증 통과.
[최종 커밋] `phase2-ready`

## 완료 보고

마지막에 다음을 정리해 보고하라:
- 최종 상태 요약 (합격한 게이트 목록)
- 기준선 재현 수치 (과제별 표)
- 4090에서의 전체 평가 소요 시간 → 권장 CI 실행 주기 제안
- 미해결/연기 항목과 사유 (예: 데이터 클라우드 연기)
- Phase 2 시작을 위해 사람이 결정해야 할 사항 (클라우드 인스턴스 선택 등)
