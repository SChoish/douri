# AntMaze large joint — 런별 설정 요약 및 EVAL mean (ep100–600)

각 셀의 성공률은 각 런의 ``run*.log`` 안의 `=== EVAL START epoch=… ===` 블록에서 읽은 **태스크 평균** (`idm` / `actor`의 `success_rate_mean`)입니다. `RE-EVAL` 블록은 아래 표에 넣지 않았습니다(094849 부록 참고).

공통: `num_tasks=5`, `episodes_per_task=10`.

로컬 메모 사본: `runs/eval_task_scores_ep100_ep200_20260424_joint_antmaze_large.md`.

**갱신:** 2026-04-25 19:14 KST — 아래 표·CSV 블록·원본 로그 목록은 `python scripts/update_eval_task_scores_joint_antmaze_large.py` 로 재생성.

**sweep 시각화 (단일 통합 그림, τ×α = 4×4):** `docs/figures/goub_tau_alpha_sweep_antmaze_large_heatmaps.png` — **3행×4열**: (1) IDM 태스크 평균 success, (2) Actor 태스크 평균 success, (3) Actor 상대 증가율 `(Actor−IDM)/IDM` (RdBu_r, 0 중심). 두 sweep(`antmaze_navigate_goub_tau_alpha_sweep` + `antmaze_navigate_goub_tau_at_alpha0p3_sweep`)을 한 그리드에 합칩니다 (τ∈{0.5,1,5,10,20}, α∈{0,0.1,0.3,0.5,1}). 생성: `python scripts/plot_goub_tau_alpha_sweep_heatmaps.py`

**α=0.3 고정, τ∈{5,10,20} (large) 운영:** 설정 `config/sweep_goub_tau_at_alpha0p3/`, 실행 `scripts/launch_goub_tau_at_alpha0p3_large.sh` / 재개 `scripts/resume_goub_tau_at_alpha0p3_large.sh` — 그림은 위 통합 PNG에 포함됩니다.

## 런 디렉터리 (짧은 ID)

| run 디렉터리 | 짧은 ID |
|--------------|---------|
| `20260424_010648_joint_dqc_seed0_antmaze-large-navigate-v0` | **010648** |
| `20260424_031250_joint_dqc_seed0_antmaze-large-navigate-v0` | **031250** |
| `20260424_094849_joint_dqc_seed0_antmaze-large-navigate-v0` | **094849** |
| `20260424_113827_joint_dqc_seed0_antmaze-large-navigate-v0` | **113827** |
| `20260424_122655_joint_dqc_seed0_antmaze-large-navigate-v0` | **122655** |
| `20260424_131042_joint_dqc_seed0_antmaze-large-navigate-v0` | **131042** |
| `20260424_132645_joint_dqc_seed0_antmaze-large-navigate-v0` | **132645** |
| `20260424_141315_joint_dqc_seed0_antmaze-large-navigate-v0` | **141315** |
| `20260424_141543_joint_dqc_seed0_antmaze-large-navigate-v0` | **141543** |
| `20260424_175435_joint_dqc_seed0_antmaze-large-navigate-v0` | **175435** |
| `20260424_180911_joint_dqc_seed0_antmaze-large-navigate-v0` | **180911** |
| `20260424_191409_joint_dqc_seed0_antmaze-large-navigate-v0` | **191409** |
| `20260424_200005_joint_dqc_seed0_antmaze-large-navigate-v0` | **200005** |
| `20260424_211338_joint_dqc_seed0_antmaze-large-navigate-v0` | **211338** |
| `20260424_222946_joint_dqc_seed0_antmaze-large-navigate-v0` | **222946** |
| `20260424_234637_joint_dqc_seed0_antmaze-large-navigate-v0` | **234637** |
| `20260425_010031_joint_dqc_seed0_antmaze-large-navigate-v0` | **010031** |
| `20260425_021613_joint_dqc_seed0_antmaze-large-navigate-v0` | **021613** |
| `20260425_033320_joint_dqc_seed0_antmaze-large-navigate-v0` | **033320** |
| `20260425_044646_joint_dqc_seed0_antmaze-large-navigate-v0` | **044646** |
| `20260425_060132_joint_dqc_seed0_antmaze-large-navigate-v0` | **060132** |
| `20260425_093720_joint_dqc_seed0_antmaze-large-navigate-v0` | **093720** |
| `20260425_123020_joint_dqc_seed0_antmaze-large-navigate-v0` | **123020** |
| `20260425_142917_joint_dqc_seed0_antmaze-large-navigate-v0` | **142917** |
| `20260425_155630_joint_dqc_seed0_antmaze-large-navigate-v0` | **155630** |
| `20260425_163208_joint_dqc_seed0_antmaze-large-navigate-v0` | **163208** |
| `20260425_163835_joint_dqc_seed0_antmaze-large-navigate-v0` | **163835** |

## 런별 차이점 (`config_used.yaml` + 코드 기본값 병합)

| run | 차이점 |
|-----|--------|
| **010648** | `spi_tau=1.0` `subgoal_α=0.1` `bridge_type=goub` `clip_path_to_goal=true` `train_epochs=200` `plan_candidates=8` `run_group=antmaze_navigate` |
| **031250** | `spi_tau=5.0` `subgoal_α=0.1` `bridge_type=unidb_gou` `clip_path_to_goal=true` `train_epochs=1000` `run_group=antmaze_navigate` |
| **094849** | `spi_tau=5.0` `subgoal_α=0.1` `bridge_type=unidb_gou` `clip_path_to_goal=true` `train_epochs=1000` `run_group=antmaze_navigate` |
| **113827** | `spi_tau=5.0` `subgoal_α=0.1` `bridge_type=unidb_gou` `clip_path_to_goal=true` `train_epochs=200` `run_group=antmaze_navigate` |
| **122655** | `spi_tau=5.0` `subgoal_α=0.1` `bridge_type=unidb_gou` `clip_path_to_goal=true` `train_epochs=500` `run_group=antmaze_navigate` |
| **131042** | `spi_tau=1.0` `subgoal_α=0.1` `bridge_type=unidb_gou` `clip_path_to_goal=true` `train_epochs=500` `plan_candidates=8` `run_group=antmaze_navigate` |
| **132645** | `spi_tau=1.0` `subgoal_α=0.1` `bridge_type=goub` `clip_path_to_goal=true` `train_epochs=200` `plan_candidates=8` `run_group=antmaze_navigate` |
| **141315** | `spi_tau=1.0` `subgoal_α=0.1` `bridge_type=goub` `clip_path_to_goal=true` `train_epochs=500` `run_group=antmaze_navigate` |
| **141543** | `spi_tau=1.0` `subgoal_α=0.1` `bridge_type=goub` `clip_path_to_goal=true` `train_epochs=500` `run_group=antmaze_navigate` |
| **175435** | `spi_tau=1.0` `subgoal_α=0.1` `bridge_type=goub` `clip_path_to_goal=true` `train_epochs=500` `run_group=antmaze_navigate` |
| **180911** | `spi_tau=1.0` `subgoal_α=0.1` `bridge_type=goub` `clip_path_to_goal=true` `train_epochs=500` `run_group=antmaze_navigate` |
| **191409** | `spi_tau=1.0` `subgoal_α=0.1` `bridge_type=goub` `clip_path_to_goal=true` `train_epochs=500` `run_group=antmaze_navigate` |
| **200005** | `spi_tau=0.5` `subgoal_α=0.0` `bridge_type=goub` `clip_path_to_goal=true` `train_epochs=400` `run_group=antmaze_navigate_goub_tau_alpha_sweep` |
| **211338** | `spi_tau=0.5` `subgoal_α=0.5` `bridge_type=goub` `clip_path_to_goal=true` `train_epochs=400` `run_group=antmaze_navigate_goub_tau_alpha_sweep` |
| **222946** | `spi_tau=0.5` `subgoal_α=1.0` `bridge_type=goub` `clip_path_to_goal=true` `train_epochs=400` `run_group=antmaze_navigate_goub_tau_alpha_sweep` |
| **234637** | `spi_tau=1.0` `subgoal_α=0.0` `bridge_type=goub` `clip_path_to_goal=true` `train_epochs=400` `run_group=antmaze_navigate_goub_tau_alpha_sweep` |
| **010031** | `spi_tau=1.0` `subgoal_α=0.5` `bridge_type=goub` `clip_path_to_goal=true` `train_epochs=400` `run_group=antmaze_navigate_goub_tau_alpha_sweep` |
| **021613** | `spi_tau=1.0` `subgoal_α=1.0` `bridge_type=goub` `clip_path_to_goal=true` `train_epochs=400` `run_group=antmaze_navigate_goub_tau_alpha_sweep` |
| **033320** | `spi_tau=5.0` `subgoal_α=0.0` `bridge_type=goub` `clip_path_to_goal=true` `train_epochs=400` `run_group=antmaze_navigate_goub_tau_alpha_sweep` |
| **044646** | `spi_tau=5.0` `subgoal_α=0.5` `bridge_type=goub` `clip_path_to_goal=true` `train_epochs=400` `run_group=antmaze_navigate_goub_tau_alpha_sweep` |
| **060132** | `spi_tau=5.0` `subgoal_α=1.0` `bridge_type=goub` `clip_path_to_goal=true` `train_epochs=400` `run_group=antmaze_navigate_goub_tau_alpha_sweep` |
| **093720** | `spi_tau=5.0` `subgoal_α=0.3` `bridge_type=goub` `clip_path_to_goal=true` `train_epochs=400` `run_group=antmaze_navigate_goub_tau_at_alpha0p3_sweep` |
| **123020** | `spi_tau=10.0` `subgoal_α=0.3` `bridge_type=goub` `clip_path_to_goal=true` `train_epochs=400` `run_group=antmaze_navigate_goub_tau_at_alpha0p3_sweep` |
| **142917** | `spi_tau=10.0` `subgoal_α=0.5` `bridge_type=goub` `clip_path_to_goal=true` `train_epochs=400` `run_group=antmaze_navigate_goub_tau_alpha_sweep` |
| **155630** | `spi_tau=20.0` `subgoal_α=0.3` `bridge_type=goub` `clip_path_to_goal=true` `train_epochs=400` `run_group=antmaze_navigate_goub_tau_at_alpha0p3_sweep` |
| **163208** | `spi_tau=10.0` `subgoal_α=0.3` `bridge_type=goub` `clip_path_to_goal=true` `train_epochs=400` `plan_candidates=8` `run_group=antmaze_navigate` |
| **163835** | `spi_tau=10.0` `subgoal_α=0.1` `bridge_type=goub` `clip_path_to_goal=true` `train_epochs=400` `run_group=antmaze_navigate_goub_tau_alpha_sweep` |

## EVAL mean: IDM / Actor (태스크 평균)

형식: `IDM mean` / `Actor mean`. 해당 epoch에 `EVAL` 블록이 없으면 `—`. 로그는 각 런의 ``run*.log``(``run.log`` + ``run_resume_from*.log`` 등)를 합쳐 파싱합니다.

| run | ep100 (IDM / Actor) | ep200 (IDM / Actor) | ep300 (IDM / Actor) | ep400 (IDM / Actor) | ep500 (IDM / Actor) | ep600 (IDM / Actor) |
|-----|----------------------|----------------------|----------------------|----------------------|----------------------|----------------------|
| **010648** | 0.56 / 0.48 | 0.40 / 0.48 | — | — | — | — |
| **031250** | 0.58 / 0.48 | 0.58 / 0.54 | 0.56 / 0.76 | 0.66 / 0.82 | 0.64 / 0.58 | 0.66 / 0.74 |
| **094849** | 0.56 / 0.54 | 0.58 / 0.66 | 0.60 / 0.64 | 0.54 / 0.62 | — | — |
| **113827** | 0.42 / 0.16 | 0.72 / 0.30 | — | — | — | — |
| **122655** | 0.54 / 0.62 | 0.62 / 0.74 | — | — | — | — |
| **131042** | 0.52 / 0.56 | — | — | — | — | — |
| **132645** | 0.70 / 0.76 | 0.56 / 0.72 | — | — | — | — |
| **141315** | — | — | — | — | — | — |
| **141543** | 0.70 / 0.56 | 0.74 / 0.76 | 0.66 / 0.66 | — | — | — |
| **175435** | — | — | — | — | — | — |
| **180911** | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | — |
| **191409** | 0.46 / 0.00 | — | — | — | — | — |
| **200005** | 0.78 / 0.60 | 0.46 / 0.64 | 0.62 / 0.68 | 0.66 / 0.58 | — | — |
| **211338** | 0.74 / 0.64 | 0.88 / 0.62 | 0.78 / 0.64 | 0.70 / 0.80 | — | — |
| **222946** | 0.42 / 0.32 | 0.50 / 0.42 | 0.60 / 0.52 | 0.50 / 0.56 | — | — |
| **234637** | 0.72 / 0.56 | 0.76 / 0.52 | 0.68 / 0.60 | 0.64 / 0.66 | — | — |
| **010031** | 0.58 / 0.64 | 0.80 / 0.70 | 0.62 / 0.82 | 0.74 / 0.78 | — | — |
| **021613** | 0.36 / 0.30 | 0.44 / 0.26 | 0.44 / 0.46 | 0.64 / 0.46 | — | — |
| **033320** | 0.56 / 0.74 | 0.54 / 0.68 | 0.52 / 0.80 | 0.60 / 0.68 | — | — |
| **044646** | 0.66 / 0.58 | 0.82 / 0.70 | 0.78 / 0.68 | 0.80 / 0.82 | — | — |
| **060132** | 0.42 / 0.44 | 0.42 / 0.30 | 0.46 / 0.52 | 0.38 / 0.44 | — | — |
| **093720** | 0.64 / 0.82 | 0.78 / 0.82 | 0.80 / 0.86 | 0.76 / 0.76 | 0.76 / 0.72 | 0.80 / 0.78 |
| **123020** | 0.62 / 0.76 | 0.76 / 0.82 | 0.74 / 0.80 | 0.72 / 0.86 | 0.86 / 0.84 | 0.86 / 0.90 |
| **142917** | 0.66 / 0.70 | 0.78 / 0.74 | 0.78 / 0.80 | 0.64 / 0.78 | — | — |
| **155630** | 0.74 / 0.70 | 0.86 / 0.70 | — | — | — | — |
| **163208** | 0.68 / 0.70 | 0.64 / 0.74 | 0.82 / 0.70 | 0.70 / 0.88 | — | — |
| **163835** | 0.68 / 0.74 | 0.62 / 0.68 | 0.64 / 0.70 | 0.66 / 0.78 | — | — |

### 동일 표 (CSV)

```csv
run_id,ep100_idm_mean,ep200_idm_mean,ep300_idm_mean,ep400_idm_mean,ep500_idm_mean,ep600_idm_mean,ep100_actor_mean,ep200_actor_mean,ep300_actor_mean,ep400_actor_mean,ep500_actor_mean,ep600_actor_mean
010648,0.5600,0.4000,,,,,0.4800,0.4800,,,,
031250,0.5800,0.5800,0.5600,0.6600,0.6400,0.6600,0.4800,0.5400,0.7600,0.8200,0.5800,0.7400
094849,0.5600,0.5800,0.6000,0.5400,,,0.5400,0.6600,0.6400,0.6200,,
113827,0.4200,0.7200,,,,,0.1600,0.3000,,,,
122655,0.5400,0.6200,,,,,0.6200,0.7400,,,,
131042,0.5200,,,,,,0.5600,,,,,
132645,0.7000,0.5600,,,,,0.7600,0.7200,,,,
141315,,,,,,,,,,,,
141543,0.7000,0.7400,0.6600,,,,0.5600,0.7600,0.6600,,,
175435,,,,,,,,,,,,
180911,0.0000,0.0000,0.0000,0.0000,0.0000,,0.0000,0.0000,0.0000,0.0000,0.0000,
191409,0.4600,,,,,,0.0000,,,,,
200005,0.7800,0.4600,0.6200,0.6600,,,0.6000,0.6400,0.6800,0.5800,,
211338,0.7400,0.8800,0.7800,0.7000,,,0.6400,0.6200,0.6400,0.8000,,
222946,0.4200,0.5000,0.6000,0.5000,,,0.3200,0.4200,0.5200,0.5600,,
234637,0.7200,0.7600,0.6800,0.6400,,,0.5600,0.5200,0.6000,0.6600,,
010031,0.5800,0.8000,0.6200,0.7400,,,0.6400,0.7000,0.8200,0.7800,,
021613,0.3600,0.4400,0.4400,0.6400,,,0.3000,0.2600,0.4600,0.4600,,
033320,0.5600,0.5400,0.5200,0.6000,,,0.7400,0.6800,0.8000,0.6800,,
044646,0.6600,0.8200,0.7800,0.8000,,,0.5800,0.7000,0.6800,0.8200,,
060132,0.4200,0.4200,0.4600,0.3800,,,0.4400,0.3000,0.5200,0.4400,,
093720,0.6400,0.7800,0.8000,0.7600,0.7600,0.8000,0.8200,0.8200,0.8600,0.7600,0.7200,0.7800
123020,0.6200,0.7600,0.7400,0.7200,0.8600,0.8600,0.7600,0.8200,0.8000,0.8600,0.8400,0.9000
142917,0.6600,0.7800,0.7800,0.6400,,,0.7000,0.7400,0.8000,0.7800,,
155630,0.7400,0.8600,,,,,0.7000,0.7000,,,,
163208,0.6800,0.6400,0.8200,0.7000,,,0.7000,0.7400,0.7000,0.8800,,
163835,0.6800,0.6200,0.6400,0.6600,,,0.7400,0.6800,0.7000,0.7800,,
```

### 표 해석 메모

- **141315**, **175435**: `run.log`에 `EVAL` 없음(즉시 종료 또는 초기화만). **141315**는 **141543**과 동일 `config_used.yaml`이나 런만 조기 종료.
- **131042**: epoch 100까지만 학습/EVAL.
- **191409**: epoch 100 `EVAL`만 존재; Actor mean이 로그상 0.00.
- **141543**: `train_epochs=500`이나 로그는 epoch 300 `EVAL` 이후 없음 → ep400 없음.
- **010648**, **113827**, **122655**: `train_epochs=200` 등으로 ep300/400 `EVAL` 없음.
- **180911**: 로그상 학습 지표가 비정상(`goub_g`·`goub_roll` 등 0) 구간 — mean 전부 0.00.
- 태스크별 `task_1`…`task_5` 숫자는 각 `runs/…/run.log` 원문 참고.

---

## 부록: 094849 사후 RE-EVAL (동일 epoch 체크포인트, 조건 수정 후 재측정)

학습 로그 끝의 `RE-EVAL START` 결과입니다. 위 표의 **094849 인라인 EVAL**과 다릅니다.

### RE-EVAL epoch 100

| policy | mean | task_1 | task_2 | task_3 | task_4 | task_5 |

|--------|------|--------|--------|--------|--------|--------|
| IDM | 0.48 | 0.30 | 0.70 | 0.50 | 0.50 | 0.40 |
| Actor | 0.56 | 0.30 | 0.50 | 1.00 | 0.30 | 0.70 |

### RE-EVAL epoch 200

| policy | mean | task_1 | task_2 | task_3 | task_4 | task_5 |

|--------|------|--------|--------|--------|--------|--------|
| IDM | 0.64 | 0.90 | 0.50 | 1.00 | 0.50 | 0.30 |
| Actor | 0.52 | 0.40 | 0.40 | 0.80 | 0.50 | 0.50 |

---

## 원본 로그 경로

로컬 `runs/` 아래 (저장소에서는 `.gitignore`):

- `runs/20260424_010648_joint_dqc_seed0_antmaze-large-navigate-v0/run.log`
- `runs/20260424_031250_joint_dqc_seed0_antmaze-large-navigate-v0/run.log`
- `runs/20260424_094849_joint_dqc_seed0_antmaze-large-navigate-v0/run.log`
- `runs/20260424_113827_joint_dqc_seed0_antmaze-large-navigate-v0/run.log`
- `runs/20260424_122655_joint_dqc_seed0_antmaze-large-navigate-v0/run.log`
- `runs/20260424_131042_joint_dqc_seed0_antmaze-large-navigate-v0/run.log`
- `runs/20260424_132645_joint_dqc_seed0_antmaze-large-navigate-v0/run.log`
- `runs/20260424_141315_joint_dqc_seed0_antmaze-large-navigate-v0/run.log`
- `runs/20260424_141543_joint_dqc_seed0_antmaze-large-navigate-v0/run.log`
- `runs/20260424_175435_joint_dqc_seed0_antmaze-large-navigate-v0/run.log`
- `runs/20260424_180911_joint_dqc_seed0_antmaze-large-navigate-v0/run.log`
- `runs/20260424_191409_joint_dqc_seed0_antmaze-large-navigate-v0/run.log`
- `runs/20260424_200005_joint_dqc_seed0_antmaze-large-navigate-v0/run.log`
- `runs/20260424_211338_joint_dqc_seed0_antmaze-large-navigate-v0/run.log`
- `runs/20260424_222946_joint_dqc_seed0_antmaze-large-navigate-v0/run.log`
- `runs/20260424_234637_joint_dqc_seed0_antmaze-large-navigate-v0/run.log`
- `runs/20260425_010031_joint_dqc_seed0_antmaze-large-navigate-v0/run.log`
- `runs/20260425_021613_joint_dqc_seed0_antmaze-large-navigate-v0/run.log`
- `runs/20260425_033320_joint_dqc_seed0_antmaze-large-navigate-v0/run.log`
- `runs/20260425_044646_joint_dqc_seed0_antmaze-large-navigate-v0/run.log`
- `runs/20260425_060132_joint_dqc_seed0_antmaze-large-navigate-v0/run.log`
- `runs/20260425_093720_joint_dqc_seed0_antmaze-large-navigate-v0/run.log`
- `runs/20260425_123020_joint_dqc_seed0_antmaze-large-navigate-v0/run.log`
- `runs/20260425_142917_joint_dqc_seed0_antmaze-large-navigate-v0/run.log`
- `runs/20260425_155630_joint_dqc_seed0_antmaze-large-navigate-v0/run.log`
- `runs/20260425_163208_joint_dqc_seed0_antmaze-large-navigate-v0/run.log`
- `runs/20260425_163835_joint_dqc_seed0_antmaze-large-navigate-v0/run.log`
