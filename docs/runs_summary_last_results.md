# Runs summary (from `run.log` only)

각 런의 `run.log` 헤더(`run_setup`, `dynamics`, `subgoal`)와 마지막 학습 epoch 줄, 마지막 `EVAL END` 블록의 성공률만 사용했습니다. 로그에 없는 항목(예: `exact_residual_scale`, `subgoal_goal_representation`)은 표에 넣지 않습니다. **`actor_success_mean`이 로그에서 `0.00`으로 나온 경우**(액터 평가 파이프라인 버그로 보는 값)는 아래 모든 표에서 **`-`**로 적습니다.

| run_dir | env | seed | train_epochs | planner | model | theta_schedule | theta_total | progress_alpha | bridge_gamma_inv | subgoal_target_mode | last_epoch | last_dyn | last_critic | last_actor | final_eval_epoch | idm_success_mean | actor_success_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20260514_134005_seed0_antmaze-medium-navigate-v0 | antmaze-medium-navigate-v0 | 0 | 400 | exact_residual_chain | exact_residual | prefix_progress | 1 | 0.8 | 0 | displacement | 400 | 11.444170 | 0.633380 | -0.841869 | 400 | 0.86 | 0.86 |
| 20260514_145241_seed0_antmaze-medium-navigate-v0 | antmaze-medium-navigate-v0 | 0 | 400 | exact_residual_chain | exact_residual | prefix_progress | 1 | 0.8 | 0 | absolute | 400 | 11.457234 | 0.631252 | -0.839914 | 400 | 0.92 | 0.80 |
| 20260514_160349_seed0_antmaze-medium-navigate-v0 | antmaze-medium-navigate-v0 | 0 | 400 | exact_residual_chain | exact_residual | prefix_progress | 1 | 0.8 | 0 | displacement | 400 | 11.373690 | 0.562775 | -0.833217 | 400 | 0.82 | 0.86 |
| 20260514_171607_seed0_antmaze-medium-navigate-v0 | antmaze-medium-navigate-v0 | 0 | 400 | exact_residual_chain | exact_residual | prefix_progress | 1 | 0.8 | 0 | absolute | 400 | 11.368452 | 0.556757 | -0.831794 | 400 | 0.98 | 0.82 |
| 20260514_192029_seed0_antmaze-large-navigate-v0 | antmaze-large-navigate-v0 | 0 | 400 | exact_residual_chain | exact_residual | prefix_progress | 1 | 0.8 | 0 | displacement | 400 | 10.799052 | 0.571839 | -0.867577 | 400 | 0.60 | 0.70 |
| 20260514_203728_seed0_antmaze-large-navigate-v0 | antmaze-large-navigate-v0 | 0 | 400 | exact_residual_chain | exact_residual | prefix_progress | 1 | 0.8 | 0 | absolute | 400 | 10.777523 | 0.578757 | -0.864075 | 400 | 0.82 | 0.86 |
| 20260514_215204_seed0_antmaze-large-navigate-v0 | antmaze-large-navigate-v0 | 0 | 400 | exact_residual_chain | exact_residual | prefix_progress | 1 | 0.8 | 0 | displacement | 400 | 10.692914 | 0.495962 | -0.865317 | 400 | 0.72 | 0.66 |
| 20260514_230840_seed0_antmaze-large-navigate-v0 | antmaze-large-navigate-v0 | 0 | 400 | exact_residual_chain | exact_residual | prefix_progress | 1 | 0.8 | 0 | absolute | 400 | 10.703003 | 0.496133 | -0.861622 | 400 | 0.80 | 0.82 |
| 20260515_005909_seed0_antmaze-large-navigate-v0 | antmaze-large-navigate-v0 | 0 | 400 | exact_residual_chain | exact_residual | prefix_progress | 1 | 0.8 | 0 | displacement | 400 | 10.734674 | 0.550731 | -0.878415 | 400 | 0.68 | 0.70 |
| 20260515_021619_seed0_antmaze-large-navigate-v0 | antmaze-large-navigate-v0 | 0 | 400 | exact_residual_chain | exact_residual | prefix_progress | 1 | 0.8 | 0 | absolute | 400 | 10.645862 | 0.500361 | -0.869299 | 400 | 0.78 | 0.78 |
| 20260515_033138_seed0_antmaze-giant-navigate-v0 | antmaze-giant-navigate-v0 | 0 | 400 | exact_residual_chain | exact_residual | prefix_progress | 1 | 0.8 | 0 | displacement | 400 | 9.677207 | 0.619599 | -0.875950 | 400 | 0.14 | 0.16 |
| 20260515_045403_seed0_antmaze-giant-navigate-v0 | antmaze-giant-navigate-v0 | 0 | 400 | exact_residual_chain | exact_residual | prefix_progress | 1 | 0.8 | 0 | absolute | 400 | 9.778680 | 0.624003 | -0.873858 | 400 | 0.40 | 0.46 |
| 20260515_061436_seed0_antmaze-giant-navigate-v0 | antmaze-giant-navigate-v0 | 0 | 400 | exact_residual_chain | exact_residual | prefix_progress | 1 | 0.8 | 0 | displacement | 400 | 9.662757 | 0.563182 | -0.868966 | 400 | 0.32 | 0.10 |
| 20260515_073708_seed0_antmaze-giant-navigate-v0 | antmaze-giant-navigate-v0 | 0 | 400 | exact_residual_chain | exact_residual | prefix_progress | 1 | 0.8 | 0 | absolute | 400 | 9.753189 | 0.563550 | -0.866190 | 400 | 0.24 | 0.38 |

## 추가: `runs/` antmaze-large (forward_bridge_residual, phi/full 서브골 표)

위 표는 `exact_residual_chain` 플래너 기준 스냅샷입니다. 아래는 **현재 `douri/runs/`에 있는** `forward_bridge_residual` antmaze-large 런만 대상으로, 각 폴더에서 **`run.log` 다음 `run_resume*.log`(파일명 순)**을 이어 붙여 읽었습니다. `train_epochs`는 등장한 모든 `run_setup`의 `train_epochs` 중 **최댓값**, 마지막 학습 줄은 합친 로그에서 **마지막 `epoch=…` 한 줄**, 마지막 eval은 **마지막 `EVAL END` 직전**의 `idm` / `actor` `env_success_rate_mean`입니다.

| run_dir | env | seed | train_epochs | planner | model | theta_schedule | theta_total | progress_alpha | bridge_gamma_inv | subgoal_target_mode | last_epoch | last_dyn | last_critic | last_actor | final_eval_epoch | idm_success_mean | actor_success_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20260515_143212_seed0_antmaze-large-navigate-v0 | antmaze-large-navigate-v0 | 0 | 1000 | forward_bridge_residual | exact_residual | prefix_progress | 1 | 0.8 | 0 | absolute | 590 | 10.771437 | 0.572277 | -0.805916 | 500 | 0.86 | 0.92 |
| 20260515_151754_seed0_antmaze-large-navigate-v0 | antmaze-large-navigate-v0 | 0 | 400 | forward_bridge_residual | exact_residual | prefix_progress | 1 | 0.8 | 0 | displacement | 400 | 10.948478 | 0.573170 | 2.342210 | 400 | 0.54 | - |
| 20260515_160325_seed0_antmaze-large-navigate-v0 | antmaze-large-navigate-v0 | 0 | 400 | forward_bridge_residual | exact_residual | prefix_progress | 1 | 0.8 | 0 | absolute | 400 | 10.851225 | 0.496865 | 2.392000 | 400 | 0.77 | - |
| 20260515_164914_seed0_antmaze-large-navigate-v0 | antmaze-large-navigate-v0 | 0 | 400 | forward_bridge_residual | exact_residual | prefix_progress | 1 | 0.8 | 0 | displacement | 400 | 10.862344 | 0.497954 | 2.987138 | 400 | 0.67 | - |

---

## 추가: `runs/`에 있는 cube-double / cube-triple (`run.log` + `run_resume*.log`)

위 antmaze 표는 예전에 `run.log`만으로 적어 둔 스냅샷입니다. 아래 cube 표는 **현재 `douri/runs/`에 있는 폴더**만 대상으로, 각 폴더에서 **`run.log` 다음에 `run_resume*.log`(파일명 순)**을 이어 붙여 읽었습니다. `train_epochs`는 로그에 나온 `run_setup`의 `train_epochs` **최댓값**, 마지막 학습·eval 규칙은 위 cube 전용 설명과 동일합니다.

| run_dir | env | seed | train_epochs | planner | model | theta_schedule | theta_total | progress_alpha | bridge_gamma_inv | subgoal_target_mode | last_epoch | last_dyn | last_critic | last_actor | final_eval_epoch | idm_success_mean | actor_success_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20260514_170020_seed0_cube-double-play-v0 | cube-double-play-v0 | 0 | 400 | exact_residual_chain | exact_residual | prefix_progress | 1 | 0.8 | 0 | displacement | 230 | -356.832634 | 0.577182 | -0.934622 | 200 | 0.00 | - |
| 20260514_173602_seed0_cube-double-play-v0 | cube-double-play-v0 | 0 | 400 | exact_residual_chain | exact_residual | prefix_progress | 1 | 0.8 | 0 | displacement | 400 | -179.236687 | 0.567654 | -0.964340 | 400 | 0.82 | 0.74 |
| 20260514_183730_seed0_cube-double-play-v0 | cube-double-play-v0 | 0 | 400 | exact_residual_chain | exact_residual | prefix_progress | 1 | 0.8 | 0 | displacement | 210 | -1946.451600 | 0.577283 | -0.920417 | 200 | 0.00 | - |
| 20260514_191054_seed0_cube-double-play-v0 | cube-double-play-v0 | 0 | 400 | exact_residual_chain | exact_residual | prefix_progress | 1 | 0.8 | 0 | displacement | 400 | -197.171252 | 0.567262 | -0.960247 | 400 | 0.74 | 0.56 |
| 20260514_205311_seed0_cube-triple-play-v0 | cube-triple-play-v0 | 0 | 400 | exact_residual_chain | exact_residual | prefix_progress | 1 | 0.8 | 0 | displacement | 340 | -230.331488 | 0.478585 | -0.973807 | 300 | 0.18 | 0.20 |
| 20260514_235258_seed0_cube-triple-play-v0 | cube-triple-play-v0 | 0 | 400 | exact_residual_chain | exact_residual | prefix_progress | 1 | 0.8 | 0 | displacement | 200 | -106.557922 | 0.558801 | -0.987603 | 200 | 0.00 | - |
| 20260515_010336_seed0_cube-triple-play-v0 | cube-triple-play-v0 | 0 | 1200 | exact_residual_chain | exact_residual | prefix_progress | 1 | 0.8 | 0 | displacement | 870 | -987.196591 | 0.477158 | -0.909254 | 800 | 0.00 | - |
| 20260515_033849_seed0_cube-triple-play-v0 | cube-triple-play-v0 | 0 | 1200 | exact_residual_chain | exact_residual | prefix_progress | 1 | 0.8 | 0 | displacement | 1200 | -227.692898 | 0.473847 | -0.963419 | 1200 | 0.29 | 0.27 |
| 20260515_061331_seed0_cube-triple-play-v0 | cube-triple-play-v0 | 0 | 400 | exact_residual_chain | exact_residual | prefix_progress | 1 | 0.8 | 0 | displacement | 400 | -6485.528671 | 0.484450 | -0.844693 | 400 | 0.00 | - |
| 20260515_084914_seed0_cube-triple-play-v0 | cube-triple-play-v0 | 0 | 400 | exact_residual_chain | exact_residual | prefix_progress | 1 | 0.8 | 0 | displacement | 400 | -239.601158 | 0.485157 | -0.972795 | 400 | 0.00 | - |
