# StarVLA 기반 VLA 개발 환경 구축 및 개발 프로세스

작성일: 2026-08-26 · 대상: 강기봉 (Application Engineer)
목표: Generalist AI(GEN 계열)급 범용 성능을 지향하는 자체 VLA 개발의 1차 사이클 구축

---

## 0. 전략 요약

| 트랙 | 목적 | 산출물 |
|---|---|---|
| ① SimplerEnv CI | 베이스 모델 일반 역량의 회귀 테스트 장비 | 체크포인트별 자동 평가 리포트 |
| ② OFT vs GR00T 비교 | 액션 헤드 패러다임의 공정 비교 | 스텝 통제 비교표 + 헤드 선정 근거 |
| ③ 자체 학습 | Qwen3-VL-4B-Action 기반 자체 정책 | 공동학습 레시피 v1 체크포인트 |

핵심 원칙 (논문 분석 결론 반영):
- **공동학습이 기본값** — 행동 단독 SFT는 20K 스텝 내 지각 붕괴 (StarVLA 기술보고서 §6.2)
- ②는 목적을 분리: 출발점 선정(공개 체크포인트 평가)과 패러다임 비교(스텝 통제 재학습)를 섞지 않음
- SimplerEnv는 일반 역량 CI로만 사용. 정밀 과제(열쇠/밸브)는 별도 평가 트랙 필요 (Stack류 정밀 정렬은 전 헤드 19~30% 수준)

---

## Phase 0. 공통 인프라 (1주차)

### 0-1. 하드웨어 배치

| 자원 | 역할 |
|---|---|
| 로컬 RTX 4090 | 스모크 테스트, 평가 서버(정책 서빙), 소규모 디버그 학습 |
| 클라우드 8×A100 80GB (또는 8×H100) | 본 학습 런. 스팟/온디맨드, 시간당 ~$8–15 |
| Jetson AGX Orin | (후속) 실기 배포 검증용 — 이번 사이클 범위 외 |

비용 감각 (기술보고서 Table 10, 8×A100 실측 기준):
- GPU당 배치 8 → 100K 스텝 ≈ 31.5h → **20K 스텝 ≈ 6.3h ≈ $60–100/런**
- OFT 65K 스텝 ≈ 20.5h ≈ $200–300/런
- 공동학습은 스텝당 forward/backward 2회 → 위 수치의 약 1.5–2배

### 0-2. 소프트웨어 환경 (로컬·클라우드 동일 구성)

conda 환경 **2개 분리** (공식 워크플로 — 의존성 충돌 방지):

```bash
# env 1: starVLA (모델/학습/정책 서버)
git clone https://github.com/starVLA/starVLA && cd starVLA
git checkout starVLA        # 안정 브랜치 (starVLA_dev는 실험용)
conda create -n starVLA python=3.10 && conda activate starVLA
pip install -r requirements.txt

# env 2: simpler_env (시뮬레이션 평가)
# 공식 SimplerEnv 리포 설치 후:
conda activate simpler_env
pip install tyro matplotlib mediapy websockets msgpack
pip install numpy==1.24.4   # 시뮬 호환성 필수
```

검증 절차 (순서 고정):
```bash
# 1. 시뮬 환경 검증
python examples/SimplerEnv/test_your_simplerEnv.py   # "✅ Env built successfully"
# 2. 프레임워크 스모크 테스트 (모듈 등록 트리거 — 생략 시 NotImplementedError 발생 사례 있음)
python starVLA/model/framework/QwenGR00T.py --config_yaml examples/SimplerEnv/train_files/starvla_cotrain_oxe.yaml
python starVLA/model/framework/QwenOFT.py   --config_yaml examples/SimplerEnv/train_files/starvla_cotrain_oxe.yaml
# 3. 데이터로더 단독 검증
python starVLA/dataloader/lerobot_datasets.py --config_yaml examples/SimplerEnv/train_files/starvla_cotrain_oxe.yaml
```

### 0-3. 데이터 준비 (클라우드 스토리지에 1회 구축, 로컬은 서브셋)

| 데이터 | 용도 | 비고 |
|---|---|---|
| `IPEC-COMMUNITY/bridge_orig_lerobot` | ②③ 학습 | WidowX 실기 데이터 |
| `IPEC-COMMUNITY/fractal20220817_data_lerobot` | ②③ 학습 | Google RT-1 로봇 데이터 |
| `StarVLA/LLaVA-OneVision-COCO` | ③ 공동학습 VLM 배치 | 지각 보존용 이미지-텍스트 |
| (선택) RefCOCO-g 평가 스플릿 | ③ 지각 회귀 프로브 | 그라운딩 붕괴 감시 |

각 LeRobot 데이터셋의 `meta/`에 공식 `modality.json` 배치 (bridge용/fractal용 각각, 리포 제공 파일 사용).
config: `data_mix: bridge_rt_1`

### 0-4. 실험 관리

- W&B 프로젝트 1개, 명명 규칙: `{head}-{backbone}-{data}-{steps}-{seed}` (예: `groot-qwen3vla-bridgert1-20k-s0`)
- 체크포인트는 클라우드 오브젝트 스토리지에 즉시 백업 (optimizer state는 미저장이 프레임워크 기본)
- 학습 곡선 + 평가 결과를 단일 대시보드로: 이후 벤더 평가표와 같은 양식 유지

---

## Phase 1. 트랙 ① — SimplerEnv CI 구축 (1~2주차)

### 1-1. 기준선 재현 (신뢰성 캘리브레이션)

목표: 공개 체크포인트로 공식 수치를 재현해 "우리 평가 파이프라인이 올바르다"를 먼저 증명.

```bash
# Terminal A (starVLA env): 정책 서버
huggingface-cli download StarVLA/Qwen3VL-GR00T-Bridge-RT-1 --local-dir ./results/Checkpoints/...
bash examples/SimplerEnv/eval_files/run_policy_server.sh   # your_ckpt 경로 수정

# Terminal B (simpler_env env): 평가 실행
bash examples/SimplerEnv/start_simpler_env.sh ${MODEL_PATH}
```

합격 기준: WidowX 평균 65.3 ± 3%p 내 재현 (시뮬 평가의 시드 분산 감안).
불일치 시: numpy 버전, 체크포인트 스텝 파일명, Vulkan 설정 순으로 점검.

### 1-2. CI화

- 스크립트화: 체크포인트 경로를 인자로 받아 WidowX 4과제 + Google Robot(VM/VA) 전 과제 실행 → JSON/CSV 리포트
- **과제별 수치를 항상 분리 기록** (평균만 보지 않기): Spoon/Eggplant(관대한 과제) vs Carrot(파지) vs Stack(정밀 정렬)의 분해가 모델 특성 진단의 핵심
- 리포트 표준 항목: 평균, 과제별 성공률, 추론 지연(청크당 ms), 학습 스텝 수, 데이터 믹스
- 운영 규칙: ②③의 모든 신규 체크포인트는 이 CI를 통과한 수치로만 비교·보고

### 1-3. 한계 명시 (CI 문서에 병기)

- SimplerEnv는 시각 도메인 갭·범용 조작의 지표. **정밀 정렬(열쇠 삽입 프록시)은 측정 범위 밖** — Stack 과제 수치를 참고 지표로만 기록
- 정밀 과제 평가는 후속 Phase에서 자체 환경(Isaac 시뮬 or UR10e 실기)으로 별도 구축

---

## Phase 2. 트랙 ② — OFT vs GR00T 공정 비교 (2~4주차)

### 2-1. 이중 목적의 분리

| 목적 | 방법 | 비용 |
|---|---|---|
| (a) 출발점 선정 | 공개 체크포인트 2종을 CI로 평가만 | GPU 학습 0 |
| (b) 패러다임 비교 | 동일 조건 재학습 매트릭스 | 클라우드 ~$400–600 |

(a)의 예상 결과: GR00T 65.3 vs OFT 42.7 — 단 이는 **학습량 미통제 수치**(OFT 공개본은 미완성 학습 상태, 문서상 65K 스텝 시 64.6)임을 보고서에 반드시 명기.

### 2-2. (b) 스텝 통제 재학습 매트릭스

통제 변수: 백본 = `StarVLA/Qwen3-VL-4B-Instruct-Action` 고정, 데이터 = `bridge_rt_1` 고정, 시드 ≥ 2

| 런 | 헤드 | 평가 스텝 | 예상 시간 (8×A100) |
|---|---|---|---|
| 1 | QwenGR00T | 10K / 20K / 40K 중간 평가 | ~13h (40K) |
| 2 | QwenOFT | 20K / 40K / 65K 중간 평가 | ~20h (65K) |
| 3 (선택) | QwenPI | 20K / 40K | ~13h |

측정 지표:
1. **스텝-성능 곡선** (수렴 효율 — 문서상 GR00T 20K=65.3 vs OFT 65K=64.6의 재확인)
2. 과제별 분해 (Carrot=파지 다양성, Stack=정밀 정렬 — 헤드 특성의 실질 차이)
3. 추론 지연 (OFT 1-pass vs GR00T 디노이징 수회 — 실기 50Hz 관점)
4. 동일 최종 성능 도달 시의 총 GPU-시간 (비용 효율)

의사결정 규칙 (사전 등록):
- 최종 성능 동률(±2%p) 시 → 수렴 효율·정밀 과제(Stack/Carrot) 우수 헤드 채택
- 실기 데이터(자체 텔레옵)는 변동성이 크므로, 동률 시 flow 계열(분포 표현력) 우선

### 2-3. 주의사항

- StarVLA-π는 π0 원본(공유 어텐션 MoT + KI 절연)이 아닌 근사(layer-wise cross-DiT)임 — 결과를 "π0의 성능"으로 일반화하지 않기
- KI식 stop-gradient는 프레임워크 미탑재. 백본 보호는 공동학습(③)으로 해결하는 것이 이 생태계의 검증된 경로

---

## Phase 3. 트랙 ③ — Qwen3-VL-4B-Action 자체 학습, 공동학습 기본값 (4~8주차)

### 3-1. 레시피 구조 (기술보고서 §3.1.2 + §6.2 근거)

**공동학습 = 매 스텝마다 두 배치를 함께 학습:**

```
스텝 t:
  [VLA 배치]  로봇 데이터 (bridge_rt_1 → 이후 자체 데이터)
              → forward → 행동 손실 → backward
  [VLM 배치]  LLaVA-OneVision-COCO (이미지-텍스트)
              → forward → 언어 손실 → backward
  → optimizer step (두 그래디언트 합산 적용)
```

진입점: `starVLA/training/train_starvla_cotrain.py`
근거 수치 (§6.2, WidowX): 행동 단독 54.7 → +공동학습 61.1 → +공간 유도 67.4 → +공간 사전학습 **73.2**

### 3-2. 단계적 적용 로드맵

| 단계 | 내용 | 기대 효과 |
|---|---|---|
| v1 (필수) | vanilla 공동학습: VLA + LLaVA-OneVision-COCO | 지각 붕괴 방지, +6%p급 |
| v2 (권장) | 공간 데이터 추가: 그라운딩(RefCOCO류) 배치를 VLM 로더에 편입 | 그라운딩-조작 정렬 |
| v3 (연구) | ST4VLA식 공간 사전학습: 행동 학습 전 백본을 공간 과제로 선행 단련 + 공간 프롬프팅 | +18%p급 상한 (ST4VLA 논문 정독 후 적용) |

### 3-3. 설정 예시 (config.yaml 골자)

```yaml
framework:
  name: QwenGR00T            # ②의 결론에 따라 확정
  qwenvl:
    base_vlm: StarVLA/Qwen3-VL-4B-Instruct-Action
datasets:
  vla_data:
    dataset_py: lerobot_datasets
    data_mix: bridge_rt_1     # → 자체 데이터 전환 시 교체
  vlm_data:
    dataset_py: qwenvl_jsonl  # LLaVA-OneVision-COCO
trainer:
  learning_rate:
    base: 1e-05
    qwen_vl_interface: 1.0e-05   # 백본: 저학습률
    action_model: 1.0e-04        # 헤드: 고학습률
  # freeze_modules: "..."        # 필요 시 비전 인코더 등 선택 동결
  # bf16 autocast, grad accumulation, cosine schedule — 기본 지원
```

### 3-4. 감시 지표 (§6.2의 붕괴를 조기 탐지)

- **지각 프로브**: 5K 스텝마다 RefCOCO-g(또는 소형 그라운딩 셋) IoU@0.5 측정 — 하락 추세 시 VLM 배치 비율 상향
- 행동 지표: SimplerEnv CI (트랙 ①) 10K 스텝 주기 실행
- 학습 안정성: 두 손실의 진동 여부 (vanilla 공동학습의 알려진 증상)

### 3-5. 자체 데이터 전환 (실기 트랙 연결)

1. 텔레옵 파이프라인 출력 → **LeRobot v3 포맷** (docs "Use Your Own LeRobot Dataset" 가이드 준수)
2. `meta/modality.json` 작성: UR10e 6DoF + DG-5F 관절 스키마 정의 — 상태/행동 차원, 카메라 키 명시
3. 1차 검증: 공개 데이터로 완주한 파이프라인에 `data_mix`만 교체 (디버깅 기준점 확보 후 전환)
4. 혼합 비율 주의: 공개 데이터 + 자체 데이터 혼합은 비율 ablation 필수 (Bridge 단독 71.4 > Bridge+RT-1 63.6 사례 — 소규모 혼합은 부정 전이 가능)
5. sim 병행 시: Isaac 계열에서 UR10e+DG-5F 합성 데이터 생성 → 동일 LeRobot 포맷으로 통일 (데이터 스키마가 자산이라는 원칙 유지)

---

## Phase 4. 마일스톤 요약

| 주차 | 마일스톤 | 판정 기준 |
|---|---|---|
| 1 | 인프라 + 스모크 테스트 통과 | 검증 3종 스크립트 성공 |
| 2 | ① CI 가동 + 기준선 재현 | GR00T 65.3 ± 3%p |
| 3–4 | ② 비교 매트릭스 완료 | 헤드 확정 보고서 (곡선+과제별+지연) |
| 5–6 | ③ v1 공동학습 런 | 지각 프로브 유지 + CI 수치 ≥ 공개 체크포인트 |
| 7–8 | ③ v2 + 자체 데이터 1차 통합 | 자체 LeRobot 셋 로딩·학습 완주 |

## Phase 5. 리스크 및 대응

| 리스크 | 징후 | 대응 |
|---|---|---|
| 평가 재현 실패 | 기준선 ±3%p 초과 이탈 | numpy/Vulkan/체크포인트 스텝 점검, 커뮤니티 이슈 검색 |
| 지각 붕괴 | 그라운딩 프로브 급락 | VLM 배치 비중↑, 백본 LR↓ 또는 부분 동결 |
| 부정 전이 | 데이터 추가 후 성능 하락 | 혼합 비율 ablation, 단독 학습 대조군 유지 |
| 정밀 과제 한계 | Stack류 정체 | 예상된 한계 — 본가 GR00T N1.x 트랙과 병행 (기존 2-트랙 전략 유지) |
| 클라우드 비용 초과 | 런당 예산 상회 | 스팟 인스턴스, 중간 평가로 조기 중단 규칙 |

---

## 부록: 이번 사이클에서 하지 않는 것 (범위 통제)

- KI stop-gradient 자체 구현 (③ v3 이후 검토)
- 열쇠/밸브 실기 평가 (별도 Phase — 정밀 과제 평가 환경 설계 필요)
- Jetson 배포 최적화 (헤드·레시피 확정 후)
- Generalist(멀티벤치마크) 공동학습 (v1 파이프라인 안정화 후, 공개 예제 릴리스 추적)
