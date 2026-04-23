# douri 프로젝트 구조 및 GOUB 실험 가이드

OGBench 기반 오프라인 실험 코드 모음입니다. 현재 메인 경로는 **GOUB dynamics + DQC critic + SPI actor**의 joint 학습이며, 본 문서는 그 기준으로 정리되어 있습니다.

GOUB 구현은 특정 논문을 그대로 복제한 버전이 아니라, `utils/goub.py`와 `agents/goub_dynamics.py`에서 브리지 스케줄과 경계 \(n=N\) 처리를 실험 목적에 맞게 단순화한 변형입니다. 비교 시 이 점을 전제로 보세요.

## 프로젝트 구조 한눈에

작업 디렉터리는 항상 저장소 루트(`douri`)이며, 실행 전 `PYTHONPATH=.`을 설정합니다.

| 경로 | 역할 |
|------|------|
| `agents/` | Flax/JAX 에이전트 정의 (GOUB dynamics, DQC critic, SPI actor) |
| `config/` | 실험별 YAML 설정 |
| `utils/` | 데이터셋, 환경 로더, 네트워크, 로깅, GOUB 수학 헬퍼, 공용 run/checkpoint I/O |
| `rollout/` | 체크포인트 기반 평가/시각화 패키지 (`subgoal`, `idm`, `actor` CLI + 공용 env/plot/value-field 헬퍼) |
| `main.py` | joint 학습 단일 엔트리포인트 |
| `eval_joint_checkpoint.py` | 학습된 체크포인트 평가 |

### GOUB 핵심 구현 위치

| 경로 | 역할 |
|------|------|
| `utils/goub.py` | `bridge_sample`, `posterior_mean`, `model_mean` 등 GOUB 수학 헬퍼 |
| `agents/goub_dynamics.py` | `GOUBDynamicsAgent` source of truth (loss, planning, IDM head 포함) |
| `agents/critic.py` | DQC critic 구현 (chunk + partial critic, expectile/quantile backup) + 네트워크/헬퍼/기본 config |
| `agents/actor.py` | `JointActorAgent` (SPI W₂² proximal actor) |
| `main.py` | joint 학습 공식 엔트리포인트 |

## 설정 파일

YAML은 **에이전트 default 위에 덮어쓰는 override만** 적는 슬림 구조입니다. 기본값은 다음 두 곳에서 옵니다.

- `agents/goub_dynamics.get_dynamics_config()`
- `agents/critic.get_config()` (DQC critic) / `agents/actor.get_actor_config()` (SPI actor)

대표 설정:

| 파일 | 환경 |
|------|------|
| `config/antmaze_large_navigate.yaml` | `antmaze-large-navigate-v0` |
| `config/antmaze_medium_navigate.yaml` | `antmaze-medium-navigate-v0` |

YAML top-level에서 자주 만지는 값:
- `train_epochs`, `eval_freq`, `eval_max_chunks`, `eval_task_ids`
- `batch_size`, `joint_horizon` (= `goub_N` = `subgoal_steps` = `full_chunk_horizon`)
- `plan_candidates`, `plan_noise_scale` (`plan_candidates > 1`이면 stochastic GOUB sampling)

## 학습 실행

```bash
cd /path/to/douri
export PYTHONPATH=.
python main.py --run_config=config/antmaze_large_navigate.yaml
```

`env_name`, `seed`, `train_epochs` 등은 YAML이나 CLI flag로 override할 수 있습니다.

### resume

```bash
python main.py \
  --run_config=config/antmaze_large_navigate.yaml \
  --resume_run_dir=runs/<기존_run_dir> \
  --resume_epoch=200
```

resume 시에는 `run_resume_from<E>_<ts>.log`라는 별도 로그 파일이 새로 생성됩니다.

### critic / actor 구조

- critic은 **DQC 단일**입니다. critic을 고르는 옵션(`--critic`, YAML `critic:`)은 더 이상 존재하지 않으며, `agents/critic`은 DQC를 직접 노출합니다.
- SPI actor는 **항상 함께 학습**됩니다(끄는 옵션은 더 이상 없음). actor는 GOUB가 만든 후보 chunk를 DQC critic으로 랭킹해서 학습됩니다.
- critic / actor / goub은 각자 별도 state·체크포인트를 가집니다.

### SPI actor 손실 (참고)

GOUB가 만든 후보 chunk `μ_k`(`proposal_partial_chunks`)와 critic 점수 `s_k = Q(s, g, μ_k)`로
SPI target 분포 `ρ_k = softmax(β · s_k)`를 만들고, actor `π(s, g)`(Dirac)에 대해 **W₂² proximal**을 적용합니다.

```
ρ_k    = softmax(spi_beta · score_k)
prox   = Σ_k ρ_k · ||π(s,g) - μ_k||²        # W₂²(δ_{π(s,g)}, Σ_k ρ_k δ_{μ_k})
Q̂      = Q(s, g, π(s,g)) / (mean|Q| + eps)   # batch-mean Q normalize (TD3+BC식 분모)
loss   = mean( -Q̂ + prox / (2 · spi_tau) )
```

- `prox`는 raw squared L2(차원 정규화 없음). actor 출력은 deterministic chunk이므로 Dirac → 이산 분포 W₂²의 closed form.
- `spi_tau`로 W₂² 강도, `spi_beta`로 후보 softmax 온도, `spi_q_norm_eps`로 분모 안정화.

## 런 디렉터리 레이아웃

```text
runs/<YYYYMMDD_HHMMSS>_joint_dqc_seed<seed>_<env_name>/
  config_used.yaml
  flags.json
  train.csv
  run.log                       # 최초 run
  run_resume_from<E>_<ts>.log   # resume마다 별도 파일
  checkpoints/
    goub/params_<epoch>.pkl
    critic/params_<epoch>.pkl
    actor/params_<epoch>.pkl
```

- 에포크당 스텝 수 = `ceil(dataset_size / batch_size)`.
- `global_step = epoch * steps_per_epoch`.
- 체크포인트 파일명의 숫자는 optimizer step이 아니라 **epoch 번호**입니다.

## 손실 / 로깅 요약

`GOUBDynamicsAgent.update`는 다음을 한 번에 갱신합니다.

- **GOUB 항**: ε 네트워크 가중 L1 매칭
- **path 항**: step-aligned reverse-mean 손실
- **rollout consistency 항**: short rollout 누적 오차
- **subgoal 항**: `MSE(subgoal_net(s,g), target) − α·E[V(subgoal, g)]`
- **IDM 항**: `idm_net(s_t, s_{t+1}) → a_t` MSE

주요 로깅 키:

- `phase1/loss`, `phase1/loss_goub`, `phase1/loss_path_step`, `phase1/loss_roll`
- `phase1/loss_subgoal`, `phase1/loss_idm`
- `phase1/eps_norm`, `phase1/mu_true_norm`, `phase1/mu_pred_norm`, `phase1/xN_minus_1_norm`
- `phase1/first_step_l1`, `phase1/first_step_xy_l2`, `phase1/roll_h_l1`

DQC critic / SPI actor / coupling 관련:

- `action_critic/value_loss`, `action_critic/distill_loss`, `chunk_critic/critic_loss`
- `spi_actor/actor_loss`, `spi_actor/q_mean`, `spi_actor/prox_mean`
- `coupling/critic_score_*`: critic이 GOUB 후보 chunk에 매기는 평균 점수

에포크 집계는 `train.csv` / `run.log` (옵션으로 W&B)에 기록됩니다. eval 키는 `eval/...`(actor)와 `eval_idm/...`(IDM 폴백) 두 계열이 있습니다.

## 평가 / 롤아웃 스크립트

### 학습된 체크포인트 평가

```bash
python eval_joint_checkpoint.py \
  --run_dir=runs/<run_dir> \
  --epoch=200 \
  --eval_task_ids="1,2,3,4,5" \
  --eval_episodes_per_task=10
```

### 서브골 plot (정적 PNG, value heatmap 포함 가능)

```bash
python -m rollout.subgoal \
  --run_dir=runs/<run_dir> \
  --checkpoint_epoch=-1 \
  --task_id=1 \
  --max_steps=50 \
  --out_path=rollout_state.png
```

### IDM 기반 환경 롤아웃 (mp4)

```bash
python -m rollout.idm \
  --run_dir=runs/<run_dir> \
  --checkpoint_epoch=200 \
  --task_id=1 \
  --action_chunk_horizon=5 \
  --navigator=snap \
  --out_mp4=rollout_inv_task1.mp4
```

### Actor 기반 환경 롤아웃 (mp4)

```bash
python -m rollout.actor \
  --run_dir=runs/<run_dir> \
  --checkpoint_epoch=200 \
  --task_id=1 \
  --out_mp4=rollout_actor_task1.mp4
```

## AntMaze Large 실험 메모

`runs/antmaze_large_200plus_summary.{md,csv}` 기준 현재까지 해석:

- `action_chunk_h=5`가 `10`보다 안정적.
- `plan_candidates=8`은 조합에 따라 분산이 크지만, 현재까지 best run에서는 유효.
- `subgoal_value_alpha=0.1`이 가장 유망.
- DQC default 정렬(`discount=0.999`, `quantile` backup, `kappa_b=kappa_d=0.7`, `value_geom_sample=false`)을 기본값으로 채택. 더 큰 `value_hidden_dims=(512,512,512)`도 default 반영됨.
- 대표 베스트 run: `runs/20260423_175301_joint_dqc_seed0_antmaze-large-navigate-v0`.

## 스모크 테스트

긴 학습 전에 GOUB dynamics 경로를 빠르게 점검:

```bash
python smoke_test_goub.py
```

확인 항목:

1. 실제 배치에서 `trajectory_segment` 정합성
2. reverse index mapping
3. 경계 step supervision의 finite / non-zero gradient / tiny overfit 감소
4. rollout consistency 감소
5. synthetic held-out prefix fidelity 점검

## 의존성

`requirements.txt` 기준 핵심 패키지:

- `ogbench`, `jax[cuda12]`, `flax`, `ml_collections`, `pyyaml`, `wandb`, `absl-py`, `tqdm`, `matplotlib`

JAX CUDA12 빌드 사용 → 실행 환경의 GPU/CUDA 버전과 맞춰야 합니다. GPU 메모리가 부족하면:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

## 주의사항

- 저장소 루트에서 `PYTHONPATH=.` 없이 실행하면 상대 import가 깨질 수 있습니다.
- `config_used.yaml`의 `env_name`이 일부 옛 large run에서 medium으로 잘못 저장되어 있으니, large/medium 구분은 run 디렉터리명과 `run.log`로 확인하세요.
- 옛 문서가 가리키는 `agents/goub_phase1.py`는 더 이상 존재하지 않습니다 → 최신은 `agents/goub_dynamics.py`.
