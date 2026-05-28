# Antmaze residual × subgoal target-mode grid (@600 ep)

각 환경 scale의 **상위 2개 hparam 설정**에 대해  
`residual_target_mode` × `subgoal_target_mode` **2×2 = 4 runs** → 총 **6×4 = 24 runs**.

베이스 YAML은 **`../douri/runs/<ref_run>_seed0_<env>/config_used.yaml`** 에서 읽습니다  
(grid FBR + displacement sweep과 동일 cohort). Pathbridger에 없는 키(`residual_envelope` 등)만 제거하고,  
그리드 축(`residual_target_mode`, `subgoal_target_mode`)과 `train_epochs=600`만 덮어씁니다.

## 베이스 hparam (douri @600 리더보드)

| Scale | ID | α | gap | κ | ref run | douri config |
|-------|----|---|-----|---|---------|--------------|
| giant | g1 | 0.5 | 10 | 0.8 | 20260518_155648 | `grid_fbr_disp_antmaze_giant_a0p5_gap10p0_k0p8` |
| giant | g2 | 0.1 | 5 | 0.6 | 20260517_234529 | `..._a0p1_gap5p0_k0p6` |
| large | l1 | 0.0 | 10 | 0.9 | 20260520_073104 | `..._a0p0_gap10p0_k0p9` |
| large | l2 | 0.0 | 5 | 0.9 | 20260520_221741 | `..._a0p0_gap5p0_k0p9` |
| medium | m1 | 0.0 | 5 | 0.7 | 20260521_144335 | `..._a0p0_gap5p0_k0p7` |
| medium | m2 | 0.0 | 5 | 0.6 | 20260521_133620 | `..._a0p0_gap5p0_k0p6` |

douri 베이스와 동일하게 유지되는 항목 예:

- `subgoal_distribution: diag_gaussian`, `subgoal_stochastic_loss: nll`, `subgoal_num_samples: 1`
- `planner_type: forward_bridge_residual`, `forward_bridge_path_loss_horizon: 5`
- `theta_schedule: prefix_progress`, `theta_total: 1.0`, `progress_alpha: 0.8`
- `phi` goal, `batch_size: 1024`, critic `action_chunk_horizon: 5`

douri 베이스는 `subgoal_target_mode: displacement` (+ implicit residual absolute).  
이번 sweep에서 **residual / subgoal 각각 disp·abs 4조합**만 바꿉니다.

## 4-way grid

| 약어 | `residual_target_mode` | `subgoal_target_mode` |
|------|------------------------|------------------------|
| **rd_sd** | displacement | displacement |
| **rd_sa** | displacement | absolute |
| **ra_sd** | absolute | displacement |
| **ra_sa** | absolute | absolute |

## 실행 순서 (기본 `SWEEP_QUEUE=top1_then_top2`)

1. **Top-1 (IDM):** large `l1` → medium `m1` → giant `g1` (각 4 cells)
2. **Top-2:** large `l2` → medium `m2` → giant `g2` (각 4 cells)

셀마다 `scripts/sweep_res_subgoal_cell_status.py`로 상태 확인:

- **skip** — `params_600.pkl` + `done run_dir=` (이미 600ep 완료)
- **resume** — 중간 checkpoint에서 `--resume_run_dir` / `--resume_epoch`
- **run** — 신규 학습

이미 끝난 giant `g1_rd_sd`, `g1_rd_sa` 등은 자동 skip; `g1_ra_sd`처럼 중단된 run은 resume.

```bash
cd /path/to/Pathbridger
export PYTHONPATH=.
export MUJOCO_GL=egl

# douri runs가 ../douri/runs 에 있어야 함
bash scripts/sweep_antmaze_res_subgoal_grid_600ep.sh

# top-1만
SWEEP_QUEUE=top1_only bash scripts/sweep_antmaze_res_subgoal_grid_600ep.sh

# douri 경로 지정
DOURI_ROOT=/path/to/douri bash scripts/sweep_antmaze_res_subgoal_grid_600ep.sh

# nohup (antmaze만; cube는 별도)
nohup bash scripts/nohup_sweep_res_subgoal_grid_all.sh \
  > nohup_logs/sweep_res_subgoal_all_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

**기존 nohup 스윕이 돌고 있으면 먼저 중단**한 뒤 새 스크립트를 실행하세요 (구 순서: giant→large→medium).

생성 YAML: `scripts/sweep_generated/antmaze_res_subgoal_grid_600ep/antmaze_{scale}_{baseline}_{ra|rd}_{sa|sd}.yaml`
