# Cube-play residual × subgoal grid (@600 ep)

> **보관용 문서 (2026-06):** 이 sweep을 돌리던 스크립트(`write_cube_res_subgoal_grid_yaml.py`, `sweep_cube_res_subgoal_grid_600ep.sh` 등)는 저장소에서 제거되었습니다. 동일 축의 고정 YAML은 `config/grid_targetmode/`에 남아 있습니다.

triple / double / single 각 **상위 2개 gap·κ** × **residual × subgoal 4조합** = **24 runs**.

## 베이스 gap·κ (고정)

| Env | ID | gap | κ (κ_b=κ_d) |
|-----|----|-----|-------------|
| triple | t1 | 5 | 0.8 |
| triple | t2 | 5 | 0.6 |
| double | d1 | 5 | 0.6 |
| double | d2 | 5 | 0.7 |
| single | s1 | 1 | 0.9 |
| single | s2 | 5 | 0.9 |

## 템플릿 (douri → Pathbridger)

`scripts/write_cube_res_subgoal_grid_yaml.py`가 다음을 merge합니다.

1. `../douri/config/cube_double_play_baseline.yaml`  
   - `subgoal_num_samples: 4`, `subgoal_value_alpha: 0.3`, phi, diag_gaussian+NLL, …
2. `../douri/config/grid_fbr_displacement_antmaze/antmaze_medium_a0p0_gap5p0_k0p7.yaml`  
   - `forward_bridge_residual`, `prefix_progress`, path horizon 5, …

Pathbridger에 없는 `residual_envelope`는 제거합니다.  
그리드 축만 override: `residual_target_mode`, `subgoal_target_mode`, gap, κ, `env_name`, `batch_size` (triple=4096, else 1024).

## 4-way grid

| 약어 | residual | subgoal |
|------|----------|---------|
| rd_sd | displacement | displacement |
| rd_sa | displacement | absolute |
| ra_sd | absolute | displacement |
| ra_sa | absolute | absolute |

## 실행

```bash
cd /path/to/Pathbridger
export PYTHONPATH=.
export MUJOCO_GL=egl
bash scripts/sweep_cube_res_subgoal_grid_600ep.sh
```

YAML: `scripts/sweep_generated/cube_res_subgoal_grid_600ep/cube_{scale}_{baseline}_{ra|rd}_{sa|sd}.yaml`

Antmaze 동일 sweep: `scripts/sweep_antmaze_res_subgoal_grid_600ep.sh`
