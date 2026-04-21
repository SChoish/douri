# douri 프로젝트 구조 및 GOUB 실험 가이드

이 저장소는 OGBench 기반 오프라인 실험 코드를 모아 둔 작업 디렉터리입니다. 현재 기준으로는 GOUB 계열 학습 경로가 가장 정리되어 있고, DQC / DEAS 계열 엔트리포인트도 함께 들어 있습니다.

GOUB 구현은 특정 논문을 그대로 복제한 버전이라기보다, `utils/goub.py`와 `agents/goub_dynamics.py`에서 브리지 스케줄과 경계 \(n=N\) 처리를 실험 목적에 맞게 단순화한 변형입니다. 실험 비교 시 이 점을 전제로 보는 편이 안전합니다.

## 프로젝트 구조 한눈에

작업 디렉터리는 항상 저장소 루트(`douri`)를 기준으로 사용하고, 실행 전 `PYTHONPATH=.` 를 설정하는 방식입니다.

| 경로 | 역할 |
|------|------|
| `agents/` | Flax/JAX 에이전트 정의. GOUB, DQC, critic 계열 구현 |
| `config/` | 실험별 YAML 설정 파일 |
| `utils/` | 데이터셋, 환경 로더, 네트워크, 로깅, GOUB 수학 헬퍼 |
| `scripts/` | 재현용 셸 스크립트 |
| `docs/` | 구조 리뷰, 설계 메모 |
| `main_*.py` | 학습 엔트리포인트 |
| `rollout_*.py` | 체크포인트 기반 평가/시각화 |
| `smoke_test_goub.py` | GOUB 관련 빠른 검증 스크립트 |

### 주요 학습 엔트리포인트

| 파일 | 역할 |
|------|------|
| `main_goub_dynamics.py` | GOUB 학습 엔트리포인트. path supervision + short rollout consistency 포함 |
| `main_dqc.py` | DQC 스타일 action-sequence 학습 |
| `main_critic.py` | DEAS / DQC unified critic 스택 학습 |
| `main_goub_chunk_low.py` | chunk-low 실험 엔트리포인트. 현재 워킹트리 기준 관련 agent 파일이 없으면 실행 전 동기화가 필요할 수 있음 |

## GOUB 관련 코드 위치

현재 기준으로 GOUB 구현의 핵심은 아래 파일에 모여 있습니다.

| 경로 | 역할 |
|------|------|
| `utils/goub.py` | `bridge_sample`, `posterior_mean`, `model_mean` 등 GOUB 수학 헬퍼 |
| `agents/goub_dynamics.py` | `GOUBDynamicsAgent`와 GOUB planner 공통 로직을 담은 source of truth |
| `main_goub_dynamics.py` | dynamics 학습 엔트리포인트 |
| `config/goub_dynamics_antmaze.yaml` | dynamics 기본 설정 |
| `rollout_subgoal_goub.py` | 상태 기반 서브골 롤아웃 시각화 |
| `rollout_idm_goub.py` | IDM 기반 롤아웃 평가 |
| `rollout_chunk_actor_goub.py` | chunk actor 기반 롤아웃 경로 |
| `smoke_test_goub.py` | GOUB dynamics 스모크 테스트 |

학습 경로는 이제 `main_goub_dynamics.py` 하나만 기준으로 사용합니다. 데이터셋은 `PathHGCDataset`을 사용하며, 기존 키 외에 `trajectory_segment`, `trajectory_indices`, `trajectory_start_indices`, `trajectory_terminal_indices`를 배치에 포함합니다.

## 학습 실행

### GOUB

```bash
cd /path/to/douri
export PYTHONPATH=.
python main_goub_dynamics.py
```

- 기본 YAML: `config/goub_dynamics_antmaze.yaml`
- startup에서 다음 정합성을 바로 확인합니다
  - `trajectory_segment[:, 0] == observations`
  - `trajectory_segment[:, -1] == high_actor_targets`
  - `trajectory_segment[:, 1] == next_observations`
  - `goub_N == trajectory_segment.shape[1] - 1`
- `path_loss_slice`, `rollout_loss_slice`는 디버그용 옵션이며 현재 기본 사용은 전체 state(`null`) 기준입니다

### 기타 엔트리포인트

```bash
cd /path/to/douri
export PYTHONPATH=.
python main_dqc.py
python main_critic.py
```

이 둘도 동일하게 YAML + timestamped run 디렉터리 스타일을 따릅니다. 현재 chunk 네이밍은 `full_chunk_*` / `action_chunk_*` 기준으로 통일되어 있습니다.

## 런 디렉터리 레이아웃

실행 결과는 보통 다음 형태로 저장됩니다.

```text
runs/<YYYYMMDD_HHMMSS>_goub_dynamics_seed<seed>_<env_name>/
  config_used.yaml
  flags.json
  train.csv
  run.log
  checkpoints/
    params_<epoch>.pkl
```

- 에포크당 스텝 수는 `ceil(dataset_size / batch_size)` 입니다
- `global_step`은 `epoch * steps_per_epoch` 기준 누적 업데이트 수입니다
- 체크포인트 파일명 `params_<epoch>.pkl` 의 숫자는 optimizer step이 아니라 epoch 번호입니다
- resume 시에는 `--resume_pkl` 과 `--resume_start_epoch` 를 함께 넘깁니다

## 손실과 로깅

GOUB는 기본적으로 아래 항을 함께 학습합니다.

- GOUB 항: ε 네트워크에 대한 가중 L1 평균 매칭
- 서브골 항: `subgoal_net(observations, high_actor_goals)` 와 `high_actor_targets` 사이 MSE
- IDM 항: `idm_net(s_t, s_{t+1}) -> a_t` MSE, `agent.idm_loss_weight` 로 가중

즉 한 번의 `agent.update(batch)` 안에서 GOUB, 서브골, IDM 헤드가 함께 갱신됩니다. 저장되는 `checkpoints/params_<epoch>.pkl` 안에도 IDM 관련 파라미터가 포함됩니다.

주요 로깅 키:

- `phase1/loss`
- `phase1/loss_goub`
- `phase1/loss_subgoal`
- `phase1/loss_idm`
- `phase1/eps_norm`
- `phase1/mu_true_norm`
- `phase1/mu_pred_norm`
- `phase1/xN_minus_1_norm`

에포크 집계 로그는 `train.csv`와 `run.log`, 필요 시 W&B에 함께 남습니다.

### dynamics 고유 항목

`GOUBDynamicsAgent`는 GOUB bridge 항 위에 아래 손실을 더합니다.

- step-aligned path loss
- short rollout consistency

추가 로깅 키 예시:

- `phase1/loss_path_step`
- `phase1/loss_roll`
- `phase1/first_step_l1`
- `phase1/first_step_xy_l2`
- `phase1/roll_h_l1`

## 추론 스택 개념

학습 시 브리지의 깨끗한 끝점 \(x_0\) 은 배치의 `high_actor_targets` 입니다. 사용 시에는 보통 아래 흐름을 따릅니다.

1. `predict_subgoal(s, g)` 로 현재 상태 `s` 와 상위 목표 `g` 에서 서브골을 추정합니다.
2. `plan(s, hat_w)` 로 역방향 체인을 전개합니다.
3. 이때 첫 역스텝 출력인 `next_step` 을 한 스텝 플래너 타깃처럼 사용할 수 있습니다.

경계 \(n=N\) 에서는 `bridge_var[N]=0` 특이성을 피하기 위해 `agents/goub_dynamics.py` 에서 `x_T + epsilon` 형태의 경계 매개변수를 사용합니다.

## 롤아웃과 보조 스크립트

상태 롤아웃 시각화 예시는 다음과 같습니다.

```bash
cd /path/to/douri
export PYTHONPATH=.
python rollout_subgoal_goub.py \
  --run_dir=runs/<your_run_folder> \
  --checkpoint_epoch=-1 \
  --traj_idx=0 \
  --max_steps=50 \
  --out_path=rollout_plot.png
```

- `--checkpoint_epoch=-1` 이면 `checkpoints/` 안에서 가장 큰 번호의 `params_<n>.pkl` 을 자동 선택합니다
- 기본적으로 데이터셋에서 자른 한 에피소드의 시작 상태에서 반복 계획을 수행하고, 관측 2차원 기준으로 궤적을 그립니다
- 추가 배치 실행 스크립트는 `scripts/` 아래에 있습니다
  - `run_goub_dynamics_antmaze_1000.sh`
  - `run_goub_dynamics_antmaze_missing_1000_rollouts.sh`
  - `regenerate_deterministic_rollouts.sh`

## 스모크 테스트

긴 학습 전에 GOUB dynamics 경로를 빠르게 점검하려면:

```bash
cd /path/to/douri
export PYTHONPATH=.
python smoke_test_goub.py
```

이 스크립트는 대략 아래 항목을 확인합니다.

1. 실제 배치에서 `trajectory_segment` 정합성
2. reverse index mapping
3. boundary-step supervision의 finite / non-zero gradient / tiny overfit 감소
4. rollout consistency 감소
5. synthetic held-out prefix fidelity 점검

## 의존성

`requirements.txt` 기준 핵심 패키지는 다음과 같습니다.

- `ogbench`
- `jax[cuda12]`
- `flax`
- `distrax`
- `ml_collections`
- `pyyaml`
- `wandb`
- `absl-py`
- `tqdm`
- `matplotlib`

JAX CUDA12 빌드를 사용하므로, 실제 실행 환경은 GPU / CUDA 세팅과 맞춰 두는 편이 좋습니다.

## 현재 구조 기준 주의사항

- 문서에서 예전 파일명인 `agents/goub_phase1.py` 를 기준으로 보면 현재 트리와 어긋납니다. 최신 구현 기준 파일은 `agents/goub_dynamics.py` 입니다.
- 저장소 루트에서 `PYTHONPATH=.` 없이 실행하면 상대 import가 깨질 수 있습니다.
- `main_goub_chunk_low.py` 는 남아 있지만 현재 워킹트리에서 관련 agent 파일이 빠져 있다면 바로 실행되지 않을 수 있습니다.
