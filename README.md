# PathBridger Experiment Guide

OGBench 기반 오프라인 제어 실험 코드입니다. 메인 경로는 **linear-SDE dynamics + critic + SPI actor**의 동시 학습입니다.

- Dynamics 부분은 GOUB[^goub] 계열의 bridge 아이디어에서 시작했지만, 현재 메인 경로는 `forward_bridge_residual`입니다. Closed-form forward bridge mean 위에 endpoint-preserving `PathResidualNet` residual을 얹고, `bridge_gamma_inv: 0.0`이면 hard endpoint bridge입니다. Subgoal은 `diag_gaussian` + `subgoal_stochastic_loss: nll`, `subgoal_target_mode`, `residual_target_mode`, state normalization 등의 ablation을 지원합니다.
- Critic 부분은 DQC[^dqc]의 chunk/action critic 구조와 SPI actor rescoring 경로를 유지합니다. 추가로 `direct_chunk_trl` critic mode는 action chunk goal-conditioned critic `Q_H(s, A_H, g)`를 직접 학습하고, value는 해당 chunk critic의 expectile readout으로만 학습합니다.

[^goub]: Generalized Ornstein-Uhlenbeck Bridge.
[^dqc]: Decoupled Q Chunking.

## 프로젝트 구조

작업 디렉터리는 항상 저장소 루트(`Pathbridger`)이며, 실행 전 `PYTHONPATH=.`을 설정합니다.

| 경로 | 역할 |
|------|------|
| `main.py` | 학습 엔트리포인트 (dynamics + critic + actor 동시 학습) |
| `agents/dynamics.py` | linear-SDE dynamics agent (`forward_bridge_residual`, subgoal/IDM/path 손실, planner API) |
| `utils/dynamics.py` | bridge 스케줄, sampling, posterior/model mean, exact-residual 헬퍼 |
| `utils/theta_schedules.py` | linear-SDE θ 스케줄 (`linear_beta`, `prefix_progress`) |
| `agents/critic.py` | DQC/IQL/direct chunk TRL critic |
| `agents/actor.py` | SPI actor |
| `eval_checkpoint.py` | 체크포인트에서 환경 평가만 재실행 |
| `rollout/` | checkpoint 기반 `subgoal`, `idm`, `actor` rollout/시각화 |
| `tests/` | 핵심 수치/분기 회귀 테스트 (`test_*.py`) |
| `config/` | 실험 YAML |
| `scripts/` | sweep / eval summary / heatmap 스크립트 |

## JAX · GPU (CUDA 12)

GPU 학습은 **CUDA용 JAX**와 **pip로 깔린 `nvidia-*-cu12` 공유 라이브러리 경로**가 잡혀 있어야 합니다.

1. 환경에서 설치: `pip install -U "jax[cuda12]"` (`requirements.txt`의 `jax[cuda12]`와 동일 계열).
2. `import jax` 전에 `site-packages/nvidia/*/lib`들이 `LD_LIBRARY_PATH` 앞쪽에 있어야 합니다. 그렇지 않으면 `cuSPARSE`를 못 찾거나 CPU로 폴백할 수 있습니다.
3. **conda 예시** (`offrl` 등): `$CONDA_PREFIX/etc/conda/activate.d/jax_cuda_ld.sh`에서 위 경로를 붙이고, `deactivate.d/jax_cuda_ld.sh`에서 복원합니다. 스크립트는 **`chmod +x`** 해 두는 것이 안전합니다.
4. 확인: `conda activate <env>` 후 `python -c "import jax; print(jax.default_backend(), jax.devices())"` → `gpu` / `CudaDevice`가 나와야 합니다.

## 설정

YAML은 agent default 위에 필요한 override만 적습니다. 기본값은 다음에서 옵니다.

- `agents/dynamics.get_dynamics_config()`
- `agents/critic.get_config()`
- `agents/actor.get_actor_config()`

### Top-level 옵션 (run/eval 공통)

| 키 | 기본 | 설명 |
|----|------|------|
| `env_name`, `seed`, `train_epochs` | — | OGBench 환경, 시드, 학습 epoch 수 |
| `batch_size` | `1024` | 학습 배치 크기 |
| `horizon` | `25` | dataset segment 길이 (`dynamics_N` / `subgoal_steps`와 정합) |
| `plan_candidates` | `1` | subgoal endpoint당 bridge/action 후보 수 `N`. SPI 전에 critic best-of-N으로 endpoint별 1개만 남김 |
| `plan_noise_scale` | `1.0` | plan sampling 시 추가 노이즈 scale. 실제 sampling 여부는 planner/config의 후보 생성 경로에 따름 |
| `eval_freq` | `100` | env 평가 주기(epoch) |
| `eval_task_ids` | `"1,2,3,4,5"` | OGBench task id 목록 |
| `eval_episodes_per_task` | `10` | task 당 epoch 평가 episode 수 |
| `final_eval_episodes_per_task` | `50` | `>0`이면 **마지막 epoch**에서만 episode 수를 이 값으로 override |
| `eval_max_chunks` | `200` | episode 당 최대 action chunk 수 |
| `async_prefetch` | `true` | host-side 배치 샘플링을 GPU 학습과 오버랩 |

### `dynamics:` 핵심 옵션

YAML의 `dynamics:` 키 아래에 둡니다. 모든 옵션은 `get_dynamics_config()`에 default가 잡혀 있어 명시 안 해도 됩니다.

```yaml
dynamics:
  # 모든 dynamics 키는 default가 잡혀 있어 yaml에 적지 않으면
  # agents/dynamics.get_dynamics_config()의 값이 쓰입니다.
  # 예시: 새 default와 다른 값만 명시.
  bridge_gamma_inv: 1.0e-6      # default 0.0 (hard endpoint bridge)
  subgoal_distribution: diag_gaussian   # default deterministic
```

#### Bridge / 스케줄

| 키 | 기본 | 설명 |
|----|------|------|
| `dynamics_N` | `25` | linear-SDE 단계 수 (= subgoal까지의 forward step 수) |
| `dynamics_beta_min`, `dynamics_beta_max` | `0.1`, `20.0` | `linear_beta` 스케줄의 β 범위 |
| `dynamics_lambda` | `1.0` | OU stationary scale λ |
| `bridge_gamma_inv` | `0.0` | bridge denominator offset. `0.0`이면 hard endpoint bridge |
| `theta_schedule` | `prefix_progress` | `prefix_progress` 또는 `linear_beta` (아래 §Prefix-progress 스케줄 참조) |
| `theta_total` | `1.0` | `prefix_progress` 모드의 누적 rate $\Theta_K$ |
| `progress_alpha` | `0.8` | `prefix_progress` 모드의 진행 곡률, $c_i = (i/K)^\alpha$ |
| `use_time_embedding` | `true` | `PathResidualNet` 시간 조건. `true`면 sinusoidal embedding, `false`면 raw scalar `i/K`를 그대로 concat |
| `state_normalization` | `false` | dynamics 내부 state normalization 활성화. 외부 API는 항상 env-scale absolute state를 소비/반환 |
| `path_loss_normalized` | `true` | path-supervised loss를 planner-normalized frame에서 계산. raw/norm diagnostics를 모두 로깅 |

`state_normalization: true`일 때 `state_mean`, `state_std`는 full train dataset에서 계산되어야 합니다. `main.py`는 학습 dataset 로딩 직후 이를 자동으로 채워 넣고, `DynamicsAgent.create()`는 missing stats를 허용하지 않습니다. Absolute state는 `(x - mean) / std`, delta/displacement는 `d / std`를 사용하며, `plan()`, `sample_plan()`, `predict_subgoal()`, `infer_subgoal_distribution()`, `sample_subgoal_candidates()`, `build_actor_proposals()`는 모두 env-scale absolute state contract를 유지합니다.

#### Subgoal 손실

| 키 | 기본 | 설명 |
|----|------|------|
| `subgoal_distribution` | `deterministic` | 기본은 point subgoal. 분포 학습 ablation에서는 `diag_gaussian`으로 `(mu, log_std)`를 예측 |
| `subgoal_stochastic_loss` | `mse` | `diag_gaussian` 학습 loss. `mse`는 PDF Eq. (51)에 가까운 reparameterized sample-MSE, `nll`은 일부 실험에서 의도적으로 쓰는 value-gap-weighted Gaussian NLL |
| `subgoal_target_mode` | `absolute` | `absolute`: raw subgoal output/teacher가 $s_{t+K}$. `displacement`: raw output/teacher가 $\Delta=s_{t+K}-s_t$이고 bridge는 local frame에서 학습/계획 |
| `subgoal_loss_weight` | `1.0` | subgoal regression/value loss 가중치 |
| `subgoal_value_alpha` | `0.5` | subgoal loss의 $V(\hat s_{t+K}, g)$ critic value bonus 계수. `0`이면 비활성화 |
| `subgoal_value_style` | `exponential` | subgoal regression/NLL value-gap weight 방식. `exponential`은 $\exp(c\Delta V)$, `expectile`은 $\Delta V>0$이면 `subgoal_value_expectile`, 아니면 `1-subgoal_value_expectile` |
| `subgoal_value_expectile` | `0.7` | `subgoal_value_style=expectile`일 때 양의 value gap에 주는 weight |
| `subgoal_value_gap_scale` | `1.0` | $\Delta V=V(s_{t+K}^{D}, g)-V(s_t, g)$에 곱하는 scale $c$ |
| `subgoal_num_samples` | `1` | `diag_gaussian` planning에서 뽑을 subgoal endpoint 수 `U` |
| `subgoal_log_std_min`, `subgoal_log_std_max` | `-5.0`, `1.0` | `diag_gaussian`의 log-std clamp 범위 |
| `subgoal_temperature` | `1.0` | `diag_gaussian` subgoal sampling std multiplier |
| `subgoal_steps` | `25` | subgoal 추정에 사용할 future horizon |
| `clip_path_to_goal` | `true` | 가까운 goal에서 endpoint를 실제 goal로 clip/pad해 "도착 후 머물기"를 학습 |

#### Displacement target mode

`subgoal_target_mode: displacement`는 `docs/linear_displacement.pdf`의 translated displacement chart와 맞춘 모드입니다. 내부적으로는 현재 상태를 원점으로 빼서 `z0=0`, `zK=s_{t+K}-s_t`인 bridge/residual chain을 학습하고, subgoal net도 raw `Delta`를 예측합니다. 하지만 public API(`predict_subgoal`, `infer_subgoal_distribution`, `sample_subgoal_candidates`, `plan`, actor proposal 생성)는 항상 absolute state endpoint를 반환/소비합니다. 즉 downstream IDM/critic/actor는 기존처럼 absolute state를 보고, displacement frame은 dynamics 내부 표현으로만 쓰입니다.

Stochastic subgoal의 경우 PDF는 sample-MSE 형태로 쓰여 있지만, 이 저장소는 실험 편의를 위해 `subgoal_stochastic_loss: nll`도 제공합니다. 이 차이는 의도된 구현 차이이며, NLL 모드에서도 value bonus는 reconstructed absolute sample `s_t + Delta`에 대해 계산합니다.

#### Planner / model 모드

| 키 | 기본 | 설명 |
|----|------|------|
| `planner_type` | `forward_bridge_residual` | closed-form forward bridge mean + endpoint-preserving 학습 residual `PathResidualNet` |
| `forward_bridge_mode` | `mean` | `forward_bridge*` 모드의 inference (`mean` / `sample`) |
| `forward_bridge_use_path_loss` | `true` | path-step 손실 활성화 |
| `forward_bridge_path_loss_horizon` | `0` | `>0`이면 prefix path loss만 평가해 update 비용을 줄임 |
| `dynamics_model_type` | `exact_residual` | 현재는 `forward_bridge_residual` 경로의 compatibility key |

#### Prefix-progress θ 스케줄 (default)

`linear_beta` 스케줄은 K=25 bridge의 초기 prefix를 매우 천천히 움직이지만, actor / IDM / SPI 재평가는 보통 첫 `rollout_horizon=5` proposal state만 소비하므로 짧은 prefix가 subgoal 변위의 의미 있는 비율을 차지하도록 보정할 필요가 있습니다. 그래서 default는 `prefix_progress`입니다.

`theta_schedule: prefix_progress`는 desired progress curve $c_i = (i/K)^\alpha$를 두고

$$\Theta_i = \mathrm{asinh}(c_i\, \sinh\Theta_K), \quad \theta_i = \Theta_{i+1} - \Theta_i$$

로 역산해 hard bridge marginal interpolation $\beta_i = \sinh(\Theta_i)/\sinh(\Theta_K)$가 정확히 $c_i$가 되게 합니다. 예: `K=25, alpha=0.8`이면 $c_5 \approx 0.276$ (5스텝이면 약 28% 진행).

- `prefix_progress`로 학습한 run은 `dynamics/theta_schedule_id`, `dynamics/prefix_progress_target_5`, `dynamics/prefix_progress_actual_5` 등 진단 로그가 찍힙니다.
- diffusion-style 비교 ablation을 돌릴 때만 `theta_schedule: linear_beta`를 yaml에 명시합니다.

### `critic_agent:` / `actor:` override

Critic 옵션은 **환경별로 yaml에서 명시**합니다 (action_chunk_horizon, discount, kappa_b, kappa_d). actor는 default가 권장 baseline이라 보통 `spi_beta`만 명시합니다.

```yaml
critic_agent:
  action_chunk_horizon: 5    # default 10. 모든 환경에서 5
  discount: 0.995            # default 0.99 (long-horizon 환경)
  kappa_b: 0.93              # default 0.7 (manip cube 계열)
  kappa_d: 0.8               # default 0.7

actor:
  spi_beta: 1.0              # default 10.0
  # spi_tau: 10.0 (default), spi_actor_layer_norm: true (default)
```

#### Critic modes

| `critic_type` | 설명 |
|---------------|------|
| `dqc` | 기본값. full chunk critic + action chunk critic + value를 함께 학습 |
| `iql` | DQC chunk critic 없이 action chunk critic + value만 사용 |
| `direct_chunk_trl` / `chunk_trl` / `trl` | direct chunk-level Transitive RL. primary head는 `Q_H(s, A_H, g)`이며, `V(s,g)`는 in-sample expectile readout |

Direct chunk TRL의 기본 target은 Q-only transitive target입니다.

```yaml
critic_agent:
  critic_type: direct_chunk_trl
  algorithm: direct_chunk_trl
  use_chunk_critic: false
  tau_q: 0.7
  tau_v: 0.9
  lambda_q_base: 1.0
  lambda_q_tri: 1.0
  lambda_v: 1.0
  use_v_in_q_target: false   # 기본값. true일 때만 Q_H * V target 사용
  target_tau: 0.005
  q_value_eps: 1.0e-6
```

Sampler는 같은 trajectory에서만 `i, j, k`를 뽑고, transitive split은 반드시 `k >= i + H`, `k < j`, `k + H <= T`를 만족해야 합니다. valid split이 없으면 `trl_valid_mask=0`으로 Q-tri loss에서 제외합니다.

## 학습 실행

```bash
cd /path/to/Pathbridger
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

Resume 로그는 `run_resume_from<E>_<timestamp>.log`로 따로 저장됩니다. `flags.json`이 있으면 hyperparameter는 자동으로 그 스냅샷에서 불러옵니다. Resume 시 새 config로 override하면 epoch별 평가 episode 수 같은 값을 갈아끼울 수 있습니다 (예: `eval_episodes_per_task=10`, `final_eval_episodes_per_task=50`).

## 현재 Config 레이아웃

환경마다 baseline config 1개씩 운영합니다. 알고리즘 default(`forward_bridge_residual` + `prefix_progress` + `spi_tau` baseline)는 yaml에서 생략하고, **환경별 critic 차이만 yaml에 명시**합니다.

| Config | 환경 | 비고 |
|--------|------|------|
| `antmaze_medium_navigate.yaml` | antmaze-medium-navigate-v0 | `train_epochs=200` |
| `antmaze_large_navigate.yaml` | antmaze-large-navigate-v0 | |
| `antmaze_giant_navigate.yaml` | antmaze-giant-navigate-v0 | `discount=0.995` |
| `antmaze_teleport_navigate.yaml` | antmaze-teleport-navigate-v0 | `discount=0.995` |
| `humanoidmaze_giant_navigate.yaml` | humanoidmaze-giant-navigate-v0 | 100M shards, `kappa_b=0.5`, `kappa_d=0.8` |
| `cube_single.yaml` | cube-single-play-v0 | `kappa_b=0.93`, `kappa_d=0.8` |
| `cube_double.yaml` | cube-double-play-v0 | 동일 |
| `cube_triple.yaml` | cube-triple-play-v0 | 100M shards |
| `cube_quadruple.yaml` | cube-quadruple-play-v0 | 100M shards |
| `cube_octuple.yaml` | cube-octuple-play-v0 | 100M shards, `kappa_d=0.5` |
| `puzzle_3x3.yaml` | puzzle-3x3-play-v0 | `kappa_b=0.9`, `kappa_d=0.5` |
| `puzzle_4x4.yaml` | puzzle-4x4-play-v0 | 동일 |
| `puzzle_4x5.yaml` | puzzle-4x5-play-v0 | 100M shards |
| `puzzle_4x6.yaml` | puzzle-4x6-play-v0 | 100M shards, `kappa_b=0.7` |

알고리즘 ablation을 돌리려면 위 baseline yaml을 복제해 dynamics/actor 키를 override하세요.

## Run Directory

```text
runs/<YYYYMMDD_HHMMSS>_seed<seed>_<env_name>/
  config_used.yaml
  flags.json
  train.csv
  run.log
  run_resume_from<E>_<timestamp>.log
  checkpoints/
    dynamics/params_<epoch>.pkl
    critic/params_<epoch>.pkl
    actor/params_<epoch>.pkl
```

> 과거 학습된 디렉토리들은 `..._joint_dqc_seed<seed>_<env_name>/` prefix를 그대로 사용합니다 (data 호환). eval / heatmap glob은 두 prefix를 모두 매치합니다.

## 학습 구성

Dynamics agent는 다음 손실을 함께 학습합니다.

- `phase1/loss_dynamics`: reverse mean matching (또는 `forward_bridge*` 모드의 forward bridge mean matching)
- `phase1/loss_path_step`: dataset segment와 step-aligned path loss
- `phase1/loss_roll`: short rollout consistency
- `phase1/loss_subgoal`: deterministic은 target-value-gap-weighted point MSE와 critic value bonus. `diag_gaussian`은 `subgoal_stochastic_loss=mse`이면 reparameterized sample에 대해 `stopgrad(w(Delta V)) * MSE(sample, target) - alpha * V(sample_abs, g)`를 쓰고, `subgoal_stochastic_loss=nll`이면 `stopgrad(w(Delta V)) * NLL(target | mu, log_std) - alpha * V(sample_abs, g)`를 씁니다. `nll`은 PDF의 stochastic subgoal 식과 다른 의도적 구현 옵션입니다. `subgoal_value_style=exponential`이면 `w(Delta V)=exp(c * (V(target_abs, g) - V(s, g)))`, `expectile`이면 gap이 양수일 때 `subgoal_value_expectile`, 그 외에는 `1-subgoal_value_expectile`
- `phase1/loss_idm`: embedded inverse dynamics MSE

Critic + SPI actor:

- Critic은 후보 action chunk를 평가합니다. DQC는 full chunk/action heads를 함께 쓰고, direct chunk TRL은 action chunk goal-conditioned critic `Q_H(s,A_H,g)`를 직접 학습합니다.
- SPI actor는 critic score로 만든 soft target distribution에 대해 W2-style proximal loss를 씁니다.
- `spi_tau`가 작을수록 후보 chunk 쪽으로 더 강하게 당깁니다.
- Actor와 critic SPI 경로는 항상 dynamics가 예측한 subgoal(`spi_goals`)에 condition됩니다.

주요 dynamics 로그:

- `dynamics/bridge_gamma_inv`, `dynamics/gamma_inv`, `dynamics/subgoal_target_mode` (`0=absolute`, `1=displacement`)
- `dynamics/theta_schedule_id`, `dynamics/theta_total`, `dynamics/progress_alpha`
- `dynamics/use_time_embedding`, `dynamics/state_normalization`, `dynamics/path_loss_normalized`
- `dynamics/prefix_progress_actual_5`, `dynamics/prefix_progress_target_5` (target은 `prefix_progress` 모드에서만)
- `phase1/mu_true_norm`, `phase1/mu_pred_norm`, `phase1/bridge_step_mean`

## 평가

Checkpoint eval:

```bash
PYTHONPATH=. MUJOCO_GL=egl python eval_checkpoint.py \
  --run_dir=runs/<run_dir> \
  --epoch=300 \
  --eval_task_ids="1,2,3,4,5" \
  --eval_episodes_per_task=10
```

통합 rollout:

```bash
PYTHONPATH=. MUJOCO_GL=egl python -m rollout.run \
  --run_dir=runs/<run_dir> \
  --checkpoint_epoch=300 \
  --task_ids=1,2,3,4,5 \
  --mode=all
```

`rollout.run`은 `flags.json`의 `env_name`으로 maze/navigate와 ManipSpace play(cube/puzzle)를 자동 판별합니다.

- IDM/actor episode loop는 manip와 maze가 모두 `rollout.episode_runner.run_chunked_episode`를 공유합니다 (main eval과 동일한 `dynamics.infer_subgoal` + `_idm_action_chunk` / `actor.sample_actions` 경로).
- maze 계열에서만 `--mode=subgoal` (state-space open-loop + xy plot)이 의미가 있습니다. manip은 `--mode=all|idm|actor`만 지원하고 state plot은 그리지 않습니다.
- maze 시각화는 항상 navigator snap 모드입니다 (벽 통과 옵션은 더 이상 제공하지 않습니다).
- maze 단일 모드는 `python -m rollout.subgoal|idm|actor`로, manip 통합 실행은 `python -m rollout.manip_play_rollouts`로도 직접 호출할 수 있습니다 (별도 cube/puzzle wrapper 스크립트는 더 이상 제공하지 않습니다).

CPU 일괄 rollout 헬퍼 (3종 한 번에):

```bash
# Maze 계열 (antmaze-*-navigate / antmaze-*-teleport / humanoidmaze-*)
RUN_DIR=runs/<run_dir> CHECKPOINT_EPOCH=500 \
  ./scripts/rollout_maze_cpu_three.sh

# Manip play (cube-*-play / puzzle-*-play) — IDM/Actor MP4
RUN_DIR=runs/<run_dir> CHECKPOINT_EPOCH=1000 \
  ./scripts/rollout_cube_play_cpu_three.sh
```

## scripts/

| 스크립트 | 역할 |
|----------|------|
| `run_configs_sequential_nohup.sh` | 여러 yaml을 nohup + wait로 순차 학습 |
| `run_cube_single_detached.sh` | 단일 yaml을 setsid로 분리 실행 (default `config/cube_single.yaml`) |
| `run_giant_nav_teleport_u4_u8_resume.sh` | giant/teleport baseline을 가장 최근 checkpoint에서 resume |
| `rollout_maze_cpu_three.sh` | maze run의 subgoal/idm/actor CPU rollout 일괄 (RUN_DIR 필수) |
| `rollout_cube_play_cpu_three.sh` | manip cube/puzzle CPU rollout 일괄 |
| `rollout_cube_single_cpu_three.sh` | 위 스크립트의 cube-single default wrapper |
| `profile_proposal_build.py` | `build_actor_proposals` 마이크로 프로파일러 |

## 테스트

핵심 회귀 테스트는 `tests/`에 있습니다. JAX가 깔린 환경(`offrl` conda 등)에서 그냥 파일을 직접 실행합니다.

```bash
cd /path/to/Pathbridger
PYTHONPATH=. python tests/test_exact_residual_dynamics.py
PYTHONPATH=. python tests/test_distributional_subgoal.py
PYTHONPATH=. python tests/test_forward_bridge_planner.py
PYTHONPATH=. python -m pytest tests/test_critic_modes.py -v
PYTHONPATH=. python tests/test_prefix_progress_schedule.py
```

각 테스트는 끝에 `OK: ...` 또는 `All tests passed.`를 찍거나 pytest summary를 출력합니다. dynamics 관련 변경(특히 schedule, planner, state normalization)을 한 뒤에는 forward bridge / distributional subgoal 테스트를, critic 변경 뒤에는 `tests/test_critic_modes.py`를 확인하세요.

## 주의사항

- 저장소 루트에서 `PYTHONPATH=.` 없이 실행하면 import가 깨질 수 있습니다.
- `MUJOCO_GL=egl`을 설정하면 headless rollout/영상 생성이 안정적입니다.
- Hard bridge sweep에서는 `bridge_gamma_inv: 0.0`과 `subgoal_distribution: deterministic`을 같이 둡니다.
- Finite-γ soft bridge를 엄밀히 비교할 때는 planner 차이에 주의하세요. `exact_residual_chain` schedule은 `bridge_gamma_inv`를 posterior와 marginal에 반영하지만, `forward_bridge_coefficients`는 finite-γ denominator를 쓰더라도 planner endpoint를 위해 마지막에 `b[-1]=1`, `std[-1]=0`으로 clamp합니다. Hard bridge(`bridge_gamma_inv: 0.0`) 실험에서는 문제가 없습니다.
- `linear_beta`는 diffusion-style 기준 스케줄입니다. `make_dynamics_schedule`은 exact-residual chain의 diffusion index에 맞춰 forward-time theta를 뒤집어 쓰고, `forward_bridge_coefficients`는 raw ascending theta를 그대로 씁니다. planner 간 schedule 비교나 논문용 ablation은 `prefix_progress` 기준이 더 해석하기 쉽습니다.
- `theta_schedule=prefix_progress`로 학습한 체크포인트는 평가/롤아웃 시에도 같은 schedule 인자가 자동으로 `flags.json`에서 복원됩니다. 명시적으로 override할 일이 거의 없습니다.
- 새 학습 run은 `<ts>_seed<N>_<env>` prefix를 사용합니다.
