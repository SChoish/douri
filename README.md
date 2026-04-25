# DOURI Dynamics Experiment Guide

OGBench 기반 오프라인 제어 실험 코드입니다. 현재 메인 경로는 **linear-SDE dynamics + DQC critic + SPI actor**의 joint 학습입니다.

이전 브리지 변형 선택지는 제거했고, dynamics는 단일 exact linear-SDE bridge만 사용합니다. `bridge_gamma_inv: 0.0`이면 hard endpoint bridge이며, 현재 sweep은 이 설정을 기준으로 deterministic point subgoal을 학습합니다.

## 프로젝트 구조

작업 디렉터리는 항상 저장소 루트(`douri`)이며, 실행 전 `PYTHONPATH=.`을 설정합니다.

| 경로 | 역할 |
|------|------|
| `main.py` | joint 학습 엔트리포인트 |
| `agents/goub_dynamics.py` | dynamics agent 구현. 파일명은 레거시 이름이지만 현재는 linear-SDE dynamics 경로입니다. |
| `utils/goub.py` | bridge schedule, sampling, posterior/model mean 헬퍼. 파일명은 레거시 이름입니다. |
| `agents/critic.py` | DQC critic |
| `agents/actor.py` | SPI actor |
| `rollout/` | checkpoint 기반 `subgoal`, `idm`, `actor` rollout/시각화 |
| `config/` | 실험 YAML |
| `scripts/` | sweep, eval summary, heatmap 스크립트 |

## 설정

YAML은 agent default 위에 필요한 override만 적습니다. 기본값은 다음에서 옵니다.

- `agents/goub_dynamics.get_dynamics_config()`
- `agents/critic.get_config()`
- `agents/actor.get_actor_config()`

대표 top-level 옵션:

- `env_name`, `seed`, `train_epochs`
- `batch_size`, `joint_horizon`
- `plan_candidates`, `plan_noise_scale`
- `eval_freq`, `eval_task_ids`, `eval_episodes_per_task`, `eval_max_chunks`

Dynamics override는 아직 레거시 YAML 키인 `goub:` 아래에 둡니다.

```yaml
goub:
  bridge_gamma_inv: 0.0
  subgoal_distribution: deterministic
  subgoal_value_alpha: 0.3
```

핵심 옵션:

| 키 | 의미 |
|----|------|
| `bridge_gamma_inv` | bridge denominator offset. `0.0`이면 hard endpoint bridge |
| `subgoal_distribution` | 현재 기본 sweep은 `deterministic` 사용 |
| `subgoal_value_alpha` | subgoal loss의 critic value bonus 계수 |
| `clip_path_to_goal` | 가까운 goal에서 endpoint를 실제 goal로 clip/pad할지 여부 |
| `planner_type` | 기본 `reverse_score`; ablation으로 `forward_bridge`, `forward_bridge_residual` 지원 |

`bridge_gamma_inv=0.0`인 hard endpoint 실험에서는 deterministic point subgoal을 쓰는 것이 기본입니다. Distributional subgoal은 별도 ablation이 필요할 때만 명시적으로 켭니다.

## 학습 실행

```bash
cd /path/to/douri
export PYTHONPATH=.
python main.py --run_config=config/antmaze_large_navigate.yaml
```

Resume:

```bash
python main.py \
  --run_config=config/antmaze_large_navigate.yaml \
  --resume_run_dir=runs/<run_dir> \
  --resume_epoch=200
```

Resume 로그는 `run_resume_from<E>_<timestamp>.log`로 따로 저장됩니다.

## 현재 Sweep

AntMaze large dynamics sweep:

- 환경: `antmaze-large-navigate-v0`
- epoch: `300`
- `subgoal_value_alpha ∈ {0.1, 0.3, 0.5}`
- `actor.spi_tau ∈ {5, 10}`
- `alpha=0.3, tau=10`은 기존 run이 있어서 skip
- `bridge_gamma_inv: 0.0`
- `subgoal_distribution: deterministic`

실행 스크립트:

```bash
scripts/launch_dynamics_tau_alpha_sweep_large_300ep.sh
```

## Run Directory

```text
runs/<YYYYMMDD_HHMMSS>_joint_dqc_seed<seed>_<env_name>/
  config_used.yaml
  flags.json
  train.csv
  run.log
  run_resume_from<E>_<timestamp>.log
  checkpoints/
    goub/params_<epoch>.pkl
    critic/params_<epoch>.pkl
    actor/params_<epoch>.pkl
```

Checkpoint 하위 폴더 `goub/`도 레거시 이름입니다. 현재 저장되는 agent는 linear-SDE dynamics agent입니다.

## 학습 구성

Dynamics agent는 다음 손실을 함께 학습합니다.

- `phase1/loss_goub`: reverse mean matching. 로그 키 이름은 레거시입니다.
- `phase1/loss_path_step`: dataset segment와 step-aligned path loss
- `phase1/loss_roll`: short rollout consistency
- `phase1/loss_subgoal`: deterministic subgoal MSE와 value bonus
- `phase1/loss_idm`: embedded inverse dynamics MSE

DQC critic과 SPI actor:

- DQC critic은 후보 action chunk를 평가합니다.
- SPI actor는 critic score로 만든 soft target distribution에 대해 W2-style proximal loss를 씁니다.
- `spi_tau`가 작을수록 후보 chunk 쪽으로 더 강하게 당깁니다.
- `spi_conditioned: subgoal`이면 actor와 critic은 predicted subgoal에 condition됩니다.

주요 dynamics 로그:

- `dynamics/bridge_gamma_inv`
- `dynamics/gamma_inv`
- `phase1/mu_true_norm`
- `phase1/mu_pred_norm`
- `phase1/bridge_step_mean`

## 평가

Checkpoint eval:

```bash
PYTHONPATH=. MUJOCO_GL=egl python eval_joint_checkpoint.py \
  --run_dir=runs/<run_dir> \
  --epoch=300 \
  --eval_task_ids="1,2,3,4,5" \
  --eval_episodes_per_task=10
```

Rollout 3종:

```bash
# state-space subgoal rollout
PYTHONPATH=. MUJOCO_GL=egl python rollout/subgoal.py \
  --run_dir=runs/<run_dir> \
  --checkpoint_epoch=300 \
  --task_id=1 \
  --max_steps=1000 \
  --out_path=runs/<run_dir>/rollouts/task1_subgoal.png

# IDM real-env rollout
PYTHONPATH=. MUJOCO_GL=egl python rollout/idm.py \
  --run_dir=runs/<run_dir> \
  --checkpoint_epoch=300 \
  --task_id=1 \
  --max_steps=1000 \
  --action_chunk_horizon=5 \
  --out_mp4=runs/<run_dir>/rollouts/task1_idm.mp4

# Actor real-env rollout
PYTHONPATH=. MUJOCO_GL=egl python rollout/actor.py \
  --run_dir=runs/<run_dir> \
  --checkpoint_epoch=300 \
  --task_id=1 \
  --max_chunks=1000 \
  --out_mp4=runs/<run_dir>/rollouts/task1_actor.mp4
```

## 주의사항

- 저장소 루트에서 `PYTHONPATH=.` 없이 실행하면 import가 깨질 수 있습니다.
- `MUJOCO_GL=egl`을 설정하면 headless rollout/영상 생성이 안정적입니다.
- 일부 파일명과 checkpoint 폴더명에 남은 `goub`는 레거시 이름입니다. 새 실험 의미는 linear-SDE dynamics입니다.
- 새 hard bridge sweep에서는 `bridge_gamma_inv: 0.0`과 `subgoal_distribution: deterministic`을 같이 둡니다.
