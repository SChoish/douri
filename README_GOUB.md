# GOUB Phase-1 (OGBench / HIQL 코드베이스 확장)

이 디렉터리의 **GOUB Phase-1**은 HIQL 학습 엔트리포인트(`main.py`)와는 **별도**로, GOUB에 *영감을 받은* 엔드포인트 조건 브리지 플래너와 서브골 추정기를 **에포크 기반**으로 학습합니다.

> **중요:** `utils/goub.py` 및 `agents/goub_phase1.py`에 적힌 대로, 구현은 특정 논문의 GOUB를 **그대로 재현**하는 것이 아니라, 스케줄·분산 근사·경계 \(n=N\) 처리 등에서 **단순화·변형**이 있습니다. 실험·비교 시 이 점을 전제로 사용하세요.

## 구성 요소

| 경로 | 역할 |
|------|------|
| `utils/goub.py` | 브리지 스케줄, `bridge_sample`, `posterior_mean`, `model_mean` 등 수학 헬퍼 |
| `agents/goub_phase1.py` | `GOUBPhase1Agent`: baseline endpoint-only GOUB Phase-1 |
| `agents/goub_phase1_path.py` | `GOUBPhase1PathAgent`: GOUB prior + intermediate path supervision + short rollout consistency |
| `main_goub_phase1.py` | baseline 엔트리포인트 |
| `main_goub_phase1_path.py` | path-supervised 엔트리포인트 (`PathHGCDataset`, startup segment 검증 포함) |
| `config/goub_phase1_*.yaml` | 예시 설정 (antmaze, 스모크 등) |
| `config/goub_phase1_path_antmaze.yaml` | path-supervised 기본 설정 |
| `rollout_subgoal_goub.py` | 체크포인트 로드 후 오프라인 궤적 1개와 비교하는 상태 롤아웃 + 2D 플롯 |
| `smoke_test_goub_phase1_path.py` | path-supervised 구현용 가벼운 스모크 테스트 5종 |

baseline은 `HGCDataset`을 사용합니다. path-supervised 버전은 `PathHGCDataset`을 사용하며, 기존 키에 더해 `trajectory_segment`, `trajectory_indices`, `trajectory_start_indices`, `trajectory_terminal_indices`를 배치에 포함합니다.

## 학습 실행

작업 디렉터리는 **이 저장소(douri) 루트**이고, `PYTHONPATH=.` 를 설정합니다.

```bash
cd /path/to/douri
export PYTHONPATH=.
python main_goub_phase1.py
```

- 기본 YAML: `config/goub_phase1_antmaze.yaml` (`--run_config`로 다른 파일 지정).
- **명령줄이 YAML보다 우선:** `sys.argv`에 이미 넘긴 absl 플래그는 YAML이 덮어쓰지 않습니다.
- 에이전트 하이퍼는 `--agent.xxx=...` 또는 YAML의 `agent:` 블록으로 조정합니다 (`config_flags`).

### Path-supervised 실행

```bash
cd /path/to/douri
export PYTHONPATH=.
python main_goub_phase1_path.py
```

- 기본 YAML: `config/goub_phase1_path_antmaze.yaml`
- startup에서 다음을 바로 검증합니다.
  - `trajectory_segment[:, 0] == observations`
  - `trajectory_segment[:, -1] == high_actor_targets`
  - `trajectory_segment[:, 1] == next_observations`
  - `goub_N == subgoal_steps == trajectory_segment.shape[1] - 1`
- `path_loss_slice` / `rollout_loss_slice`는 디버그용 옵션으로 남겨 두었지만, 현재 권장 기본값은 둘 다 `null`(full state)입니다.

### 런 디렉터리 레이아웃

baseline 실행은 대략 다음 형태로 저장됩니다.

```
runs/<YYYYMMDD_HHMMSS>_seed<seed>_<env_name>/
  config_used.yaml      # 사용한 YAML 복사본
  flags.json            # 병합된 absl + agent 설정
  train.csv             # CsvLogger 메트릭
  run.log               # 사람이 읽기 쉬운 에포크 로그
  checkpoints/
    params_<epoch>.pkl
```

path-supervised 실행은 baseline과 헷갈리지 않게 `agent_name`을 폴더명에 포함합니다.

```
runs/<YYYYMMDD_HHMMSS>_goub_phase1_path_seed<seed>_<env_name>/
  config_used.yaml
  flags.json
  train.csv
  run.log
  checkpoints/
    params_<epoch>.pkl
```

- **에포크당 스텝 수:** `steps_per_epoch = ceil(dataset_size / batch_size)`.
- **`global_step`:** `epoch * steps_per_epoch` (옵티마이저 업데이트 누적).
- **체크포인트:** 파일명은 `params_<epoch>.pkl`(학습 에포크 번호). `save_every_n_epochs`마다 + 마지막 에포크가 저장 주기와 겹치지 않으면 최종 1회 추가 저장. (`train.csv`의 `global_step`과는 별개입니다.)

## 손실과 로깅

- **GOUB 항:** ε 네트워크에 대한 L1 평균 매칭(가중치는 스케줄의 \(1/(2g_n^2)\) 형태).
- **서브골 항(옵션):** `subgoal_net(observations, high_actor_goals)` → `high_actor_targets` MSE, `subgoal_loss_weight` 배율.
- **역동학(IDM) 항:** `idm_net(s_t, s_{t+1})` → `a_t` MSE에 `agent.idm_loss_weight`를 곱합니다. 가중치를 `0`에 두면 IDM 손실 항은 사실상 꺼집니다. (`agents/goub_phase1.py`의 `phase1/loss_idm`.)
- **합 손실:** `phase1/loss` = GOUB + 가중 서브골 + 위 IDM 항.

**Phase1 한 번의 학습 루프**에서 GOUB·(옵션) 서브골·**IDM**이 같은 `agent.update(batch)` 안에서 같이 갱신됩니다. 저장되는 `checkpoints/params_<epoch>.pkl`에는 `idm_net` 가중치도 포함됩니다(`main_goub_phase1.py` 상단 레이아웃 설명과 동일).

오프라인 데이터만으로 IDM만 따로 학습하려면 `train_inverse_dynamics.py`(별도 타임스탬프 런 폴더)를 쓰는 경로가 있습니다. 실환경 롤아웃에서 그 pickle을 쓰는 방법은 [`README_RUNS_AND_ROLLOUT.md`](README_RUNS_AND_ROLLOUT.md) §5를 참고하세요.

`train.csv` / W&B에는 배치 마지막 스텝의 `phase1/*`와, 로그 주기마다 집계된 다음이 함께 기록됩니다.

- `training/phase1/loss_epoch_mean`
- `training/phase1/loss_goub_epoch_mean`
- `training/phase1/loss_subgoal_epoch_mean` — 로그 한 줄의 **`loss_sub_mean`**에 대응 (해당 에포크에서 `phase1/loss_subgoal` 배치 평균의 평균).
- `training/phase1/loss_idm_epoch_mean` — 동일하게 IDM 배치 평균의 에포크 평균.

기타 `run.log`에 자주 나오는 항목: `eps_norm`, `mu_true_norm`, `mu_pred_norm`, `xN_minus_1_norm` 등.

### Path-supervised 추가 항목

`GOUBPhase1PathAgent`는 기존 GOUB 항을 유지하면서 아래를 추가합니다.

- **step-aligned path loss:** 실제 `trajectory_segment`에서 같은 reverse index `n`에 대응하는 상태쌍을 직접 맞춤
- **short rollout consistency:** `plan(s_t, s_{t+K})`의 짧은 prefix를 실제 `s_{t+1..t+H}`와 맞춤

추가 로깅 키:

- `phase1/loss_path_step`
- `phase1/loss_roll`
- `phase1/first_step_l1`
- `phase1/first_step_xy_l2`
- `phase1/roll_h_l1`

## 추론 스택 (개념)

학습 시 브리지의 “깨끗한 끝점” \(x_0\)은 배치의 `high_actor_targets`로 둡니다. 배포 시에는:

1. `hat_w = predict_subgoal(s, g)` — 현재 상태 \(s\)와 상위 목표 \(g\)로 서브골(타깃 상태 벡터) 추정.
2. `plan(s, hat_w)` — 전체 역방향 체인을 결정적으로 실행; **`next_step`**은 첫 역스텝 출력 \(x_{N-1}\)로, 학습된 **한 스텝 플래너 타깃**으로 쓰입니다.

경계 \(n=N\)에서는 `bridge_var[N]=0`으로 인한 특이성을 피하기 위해, 에이전트에서 **\(x_T + \epsilon\)** 형태의 학습 잔차 매개변수를 사용합니다 (`goub_phase1.py` docstring 참고).

## 롤아웃 시각화

> **`runs/` 폴더 구조·상태/IDM/청크 롤아웃 스크립트·일괄 재생성**은 [`README_RUNS_AND_ROLLOUT.md`](README_RUNS_AND_ROLLOUT.md)에 정리해 두었습니다.

플래너 체인 길이(`goub_N` 등)와 IDM 실환경 롤아웃에서 replan 사이 `env.step` 횟수 상한(`--inv_dyn_planner_freq`)의 차이는 [`README_RUNS_AND_ROLLOUT.md`](README_RUNS_AND_ROLLOUT.md) §3 도입 단락·§4.2·§5에 서술해 두었습니다.

환경 `step` 없이, 데이터셋에서 자른 **에피소드 하나**의 \(s_0\)에서 `max_steps`만큼 위 파이프라인을 반복하고, 관측의 두 차원으로 데이터 궤적과 겹쳐 그립니다.

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

- `--checkpoint_epoch=-1`(기본값): `checkpoints/` 안 **가장 큰** 접미사 `n`의 `params_<n>.pkl` 선택.
- 예전에 `global_step`으로 저장된 런은 파일명 정수가 스텝이지만, 같은 정수를 `--checkpoint_epoch`에 넣으면 로드됩니다. 구 플래그 `--checkpoint_step`은 호환용으로 남아 있으며 사용 시 경고가 출력됩니다.
- 기본 목표 \(s_g\): 해당 에피소드 **마지막 상태(터미널)**. 다른 정의가 필요하면 스크립트를 확장하세요.
- 체크포인트는 `pickle` + `flax.serialization.from_state_dict`로 직접 로드합니다 (`flags.json`의 `agent`로 구조를 맞춤).

## 스모크 테스트

path-supervised 구현은 long run 전에 아래 스크립트로 빠르게 점검할 수 있습니다.

```bash
cd /path/to/douri
export PYTHONPATH=.
conda run -n offrl python smoke_test_goub.py
```

이 스크립트는 5가지를 확인합니다.

1. 실제 OGBench 배치에서 `trajectory_segment` 정합성
2. reverse index mapping
3. boundary-step supervision의 finite / non-zero gradient / tiny overfit 감소
4. rollout consistency 감소
5. synthetic held-out prefix fidelity에서 baseline vs path-supervised 비교

## 의존성

`requirements.txt`에 명시된 패키지에 더해, 기존 HIQL 스택과 동일하게 **JAX / Flax / optax / ml_collections** 등이 필요합니다. `pyyaml`은 YAML 로드를 위해 추가되어 있습니다.

## HIQL-only 루트 README와의 관계

루트의 `README_HIQL_ONLY.md`는 **HIQL 전용 정리**를 설명합니다. GOUB Phase-1은 그 위에 **추가 모듈**로만 존재하며, `main.py` HIQL 경로를 바꾸지 않습니다.
