# Phase 2 매트릭스 — OFT vs GR00T 스텝 통제 비교 (config 초안)

| 파일 | 헤드 | max_steps | save_interval | 평가 지점 (run_ci.py) | 8×A100 예상 |
|---|---|---|---|---|---|
| `groot_20k.yaml` | QwenGR00T | 40,000 | 10,000 | 10K / 20K / 40K | ~13h |
| `oft_65k.yaml` | QwenOFT | 65,000 | 5,000 | 20K / 40K / 65K | ~20h |

공통 통제 변수: `base_vlm=StarVLA/Qwen3-VL-4B-Instruct-Action`, `data_mix=bridge_rt_1`, per-GPU batch 16, LR 그룹
(qwen_vl_interface 1e-5 / action_model 1e-4), bf16, accelerate+DeepSpeed ZeRO-2 (`starVLA/config/deepseeds/`), 시드 0·1.

실행(클라우드): `bash scripts/phase2_launch.sh configs/phase2_matrix/groot_20k.yaml 0 8`
→ `third_party/starVLA/results/Checkpoints/<run_id>/checkpoints/steps_N_pytorch_model.pt` 를 로컬로 가져와
`python3 scripts/run_ci.py --ckpt ... --notes "phase2 groot s0 20k"`.

주의:
- 공개 체크포인트 레시피(`starvla_cotrain_oxe.yaml` + `train_starvla.py`)를 그대로 따른다. `train_starvla.py` 는 `vla_data` 만 사용하고
  `vlm_data` 블록은 무시한다(공동학습은 `train_starvla_cotrain.py`, Phase 3 범위). 따라서 Phase 2 는 행동 단독 학습이며,
  두 헤드에 동일하게 적용되므로 비교는 공정하다.
- `base_vlm` 은 로컬 경로(`playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action`)로 두었다. 문자열에 `Qwen3-VL` 이 포함되어야
  `starVLA/model/modules/vlm/__init__.py` 가 Qwen3 인터페이스로 분기한다.
- `wandb_entity` 는 TODO. W&B 명명 규칙: `{head}-qwen3vla-bridgert1-{steps}-s{seed}`.
