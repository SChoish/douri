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
- `plan_candidates`, `plan_noise_scale` (`plan_candidates > 1`이면 stochastic dynamics sampling)

`goub:` 블록에서 자주 만지는 옵션:

| 키 | 의미 |
|----|------|
| `bridge_gamma` | linear-SDE dynamics bridge의 endpoint precision. 클수록 hard endpoint bridge에 가까워짐 |
| `subgoal_distribution` | `"deterministic"`(기본) 또는 `"diag_gaussian"`. 분포형 서브골 학습 활성화 |
| `subgoal_log_std_min` / `subgoal_log_std_max` | log std 클리핑 범위 |
| `subgoal_use_mean_for_actor_goal` | true이면 SPI actor의 `spi_goals`로 분포의 평균을 사용 |
| `subgoal_nll_weight`, `subgoal_mse_weight`, `subgoal_var_reg_weight` | NLL / 평균 MSE / 분산 정규화 가중치 |
| `subgoal_fr_spi_weight`, `subgoal_fr_spi_tau` | (선택) FR/SPI 스타일 작은 서브골 정규화. 기본 0 |

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
- `spi_conditioned`로 위 식에서 쓰는 `g`를 선택합니다 (default `'subgoal'`):
  - `'subgoal'` — `g = predict_subgoal(s, g_global)` (현재까지의 동작; π/Q 모두 GOUB 예측 subgoal에 컨디셔닝).
  - `'goal'`    — `g = g_global` (HGC `high_actor_goals`); π/Q 모두 전역 goal에 컨디셔닝됩니다. 후보 chunk `μ_k`는 두 경우 모두 GOUB가 subgoal을 향해 만든 plan이므로, `'goal'` 모드에서는 prox가 "전역 goal로 가는 π"를 "subgoal로 가는 후보들"에 끌어당기는 형태가 됩니다.

### `goub.clip_path_to_goal` (subgoal/bridge 끝점 정합)

`PathHGCDataset`은 길이 `K = subgoal_steps`의 `trajectory_segment = (s_t, ..., s_{t+K})`를
샘플링합니다. `clip_path_to_goal`로 가까운 goal 처리 방식을 선택합니다 (default `false`).

- `false` (legacy): bridge 끝점 `x_0 = s_{t+K}`, subgoal_net 교사도 `s_{t+K}`. Goal이 `K`보다
  가까워도 항상 `K` 스텝 앞 상태를 가르치므로, 추론 시 가까운 goal 근처에서 subgoal이
  goal을 "넘어" 예측되며 bridge가 OOD 영역으로 들어가 정책이 goal 근처를 헤매는 현상을 유도할 수 있습니다.
- `true` (clip + pad): per-row 끝점 `x_0 = s_{min(t+K, t_g)}`, segment 꼬리는 `s_{t_g}`로 패딩.
  Bridge / subgoal_net 모두 "도달 후 머무르기" 신호를 학습합니다.
  - `L_goub`, `L_path`, `L_roll`은 식 변경 없음 (`x_0`/`segment` 키만 의미가 바뀜).
  - `L_subgoal`의 교사 `target = batch['high_actor_targets']`도 자동으로 클립 값을 받습니다.
  - Random goal 샘플은 영향 없음 (`t_g = T_final`, `t+K ≤ T_episode`라 어차피 `t+K`로 클립).
  - `validate_sample_batch`는 `clip_path_to_goal=true`일 때 `trajectory_indices`의 step 0을 허용하며, 0은 한 행의 contiguous 접미사로만 나타나야 합니다.

#### `actor.spi_conditioned`

SPI에서 actor MLP 입력과 critic Q (proposal rescoring + actor loss의 `Q(s,g,π)`)에 어떤 goal을 조건으로 줄지 선택합니다. (이전 이름 `spi_goal_conditioning`은 alias로 그대로 받습니다.)

| 값 | 의미 |
|----|------|
| `subgoal` (기본) | GOUB가 후보별로 만든 sub-goal (`candidate_goals`)을 critic Q에, `spi_goals`(=GOUB 평균 sub-goal 또는 `high_actor_goals` ― `goub.subgoal_use_mean_for_actor_goal` 플래그에 따름)을 actor MLP에 사용. 기존 동작 그대로. |
| `goal` | actor MLP와 critic Q **둘 다** 항상 전역 goal `high_actor_goals`로 conditioning. `candidate_goals`은 rescoring에서 사용하지 않으며, `subgoal_use_mean_for_actor_goal`은 무시됩니다. |

`main._build_actor_batch_from_goub`이 위 선택에 따라 `spi_goals` 키만 바꿔서 actor batch에 넣고, `_evaluate_env_tasks` / `rollout/actor.py`도 같은 conditioning을 inference에서 mirror하므로 train/eval이 일관됩니다.

### `goub.clip_path_to_goal` (subgoal/bridge 끝점 정합)

`PathHGCDataset`은 길이 `K = subgoal_steps`의 `trajectory_segment = (s_t, ..., s_{t+K})`를
샘플링합니다. `clip_path_to_goal`로 가까운 goal 처리 방식을 선택합니다 (default `false`).

- `false` (legacy): bridge 끝점 `x_0 = s_{t+K}`, subgoal_net 교사도 `s_{t+K}`. Goal이 `K`보다
  가까워도 항상 `K` 스텝 앞 상태를 가르치므로, 추론 시 가까운 goal 근처에서 subgoal이
  goal을 "넘어" 예측되며 bridge가 OOD 영역으로 들어가 정책이 goal 근처를 헤매는 현상을 유도할 수 있습니다.
- `true` (clip + pad): per-row 끝점 `x_0 = s_{min(t+K, t_g)}`, segment 꼬리는 `s_{t_g}`로 패딩.
  Bridge / subgoal_net 모두 "도달 후 머무르기" 신호를 학습합니다.
  - `L_goub`, `L_path`, `L_roll`은 식 변경 없음 (`x_0`/`segment` 키만 의미가 바뀜).
  - `L_subgoal`의 교사 `target = batch['high_actor_targets']`도 자동으로 클립 값을 받습니다.
  - Random goal 샘플은 영향 없음 (`t_g = T_final`, `t+K ≤ T_episode`라 어차피 `t+K`로 클립).
  - `validate_sample_batch`는 `clip_path_to_goal=true`일 때 `trajectory_indices`의 step 0을 허용하며, 0은 한 행의 contiguous 접미사로만 나타나야 합니다.

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
- `phase1/loss_subgoal`, `phase1/loss_subgoal_mse`, `phase1/loss_idm`
- `phase1/subgoal_value_mean`, `phase1/subgoal_value_bonus_mean`
- `phase1/eps_norm`, `phase1/mu_true_norm`, `phase1/mu_pred_norm`, `phase1/xN_minus_1_norm`
- `phase1/first_step_l1`, `phase1/first_step_xy_l2`, `phase1/roll_h_l1`
- (분포형 서브골 모드 추가) `phase1/subgoal_nll`, `phase1/subgoal_std_mean`, `phase1/subgoal_std_max`, `phase1/subgoal_fr_spi`, `phase1/subgoal_mode`
- (dynamics) `dynamics/bridge_gamma`, `dynamics/gamma_inv`

DQC critic / SPI actor / coupling 관련:

- `action_critic/value_loss`, `action_critic/distill_loss`, `chunk_critic/critic_loss`
- `spi_actor/actor_loss`, `spi_actor/q_mean`, `spi_actor/prox_mean`
- `coupling/critic_score_*`: critic이 GOUB 후보 chunk에 매기는 평균 점수
- `coupling/critic_score_gap_top1_top2`: top-1과 top-2 후보 점수 차 (margin 모니터링)
- `coupling/proposal_count`: 후보 수 (= `plan_candidates`)
- `coupling/proposal_goal_norm_mean`, `coupling/proposal_goal_std_mean`: 분포형 서브골에서
  샘플된 endpoint들의 norm 평균과 후보 간 표준편차 평균

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

## 분포형 서브골 / Linear Dynamics Bridge

기본 파이프라인은 **GOUB dynamics → DQC critic rescoring → SPI actor**이며 그대로 유지됩니다.
이번에 추가된 것은 (1) 서브골 인터페이스의 **확률화**와 (2) 브릿지의 **finite-γ 소프트화**
두 가지뿐이며, 둘 다 `goub:` 블록의 한두 줄로 켜고 끌 수 있습니다.

### 1) 결정론적 점 서브골 vs 분포형 서브골

기존(default, `subgoal_distribution: "deterministic"`):

- `SubgoalEstimatorNet(s, g_high) → ŝ_g`(점 추정)
- 손실: `MSE(ŝ_g, target) − α · V(ŝ_g, g_high)`
- `build_actor_proposals`는 이 점 `ŝ_g`로 GOUB 브릿지를 굴려 `plan_candidates`개의 trajectory를 만든 뒤
  IDM으로 chunk를 디코드.  stochasticity는 `plan_noise_scale`(브릿지 노이즈)로만 들어옴.

신규(`subgoal_distribution: "diag_gaussian"`):

- `DistributionalSubgoalEstimatorNet(s, g_high) → (μ, log_std)` (대각 Gaussian)
- 손실 (가산적):
  ```
  loss_sub = w_nll · NLL(target | μ, log_std)
           + w_mse · ||μ − target||²
           + w_var · mean(log_std²)
           − E[V(μ, g_high)]                # 기존 value bonus 그대로 유지 (μ 사용)
           + w_fr  · fr_term                # 선택, 기본 0
  ```
- `build_actor_proposals`는 먼저 `q(g_sub | s, g_high)`에서 endpoint를 N개 샘플하고,
  각 endpoint에 대해 별도로 GOUB 브릿지를 굴려 chunk를 디코드.  그래서 stochasticity가
  **subgoal 분포**에서도 흘러 들어옴.
- DQC critic은 후보별 **per-candidate goal**(`[B, N, D]`)로 rescore되며, SPI actor는
  `subgoal_use_mean_for_actor_goal=true`일 때 `μ`를 `spi_goals`로 사용.

`plan_candidates` / `plan_noise_scale`의 의미는 **그대로**입니다:

- `plan_candidates`: 최종 후보(=action chunk) 개수.
- `plan_noise_scale`: 각 endpoint 주변 GOUB 브릿지 샘플링 노이즈.
- 본 변경은 *value-filter centric* 재설계가 아닙니다.  단지 endpoint를 샘플 가능한 분포로 바꾼 것뿐.

### 2) Linear Dynamics Bridge

브릿지는 이제 단일 exact linear-SDE dynamics 구현만 사용합니다 (`utils/goub.py`).
`bridge_gamma`는 endpoint precision으로 쓰이며, γ가 클수록 hard endpoint bridge에 가까워집니다.
역방향 모델 평균과 analytic posterior teacher, forward-bridge planner가 모두 같은 dynamics schedule을 공유합니다.

### 3) 예시 YAML 스니펫

```yaml
goub:
  bridge_gamma: 1.0e7
  subgoal_distribution: diag_gaussian
  subgoal_log_std_min: -5.0
  subgoal_log_std_max: 1.0
  subgoal_temperature: 1.0
  subgoal_use_mean_for_actor_goal: true
  subgoal_nll_weight: 1.0
  subgoal_mse_weight: 0.25
  subgoal_var_reg_weight: 1.0e-4
  subgoal_fr_spi_weight: 0.0     # 끄려면 0
```

기존 YAML에 남아 있던 old bridge selector 값은 더 이상 사용하지 않습니다. `subgoal_distribution`을
명시하지 않으면 `"deterministic"`이 적용됩니다.

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
