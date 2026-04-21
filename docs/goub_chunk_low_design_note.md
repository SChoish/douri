# GOUB Phase-1 + Chunk Low-Level Policy

## Why stacked one-step IDM on planner states was brittle

The existing `chunked_idm` baseline asks an inverse-dynamics model to recover an action for each synthetic planner transition
`(x_i, x_{i+1})`. Those state pairs come from the GOUB bridge model rather than from real environment rollouts, so the low-level
controller is asked to invert transitions that may be off the offline data manifold. When that mismatch compounds across several
stacked one-step inversions, the executed rollout can drift away from the planner trajectory even when the high-level planner is
reasonable.

## What the new chunk actor learns

The new low-level controller is trained only on real offline trajectory segments. For each valid dataset index `t`, it uses:

- current state `s_t`
- local plan context built from real future states in the same episode
- real action chunk `(a_t, ..., a_{t+H_pi-1})`

The local plan context uses only a planner-friendly slice by default:

`c_t = [p(s_{t+1}) - p(s_t), ..., p(s_{t+H_Q}) - p(s_t)]`

with `p(s) = s[low_goal_slice]`.

No synthetic planner-generated transitions are used as labels during chunk-policy training.

## How planner output is used at inference time

At inference time the GOUB Phase-1 planner stays frozen. It generates a local bridge trajectory from the current state and the
predicted subgoal. The chunk actor then replaces the training-time real local context with the planner-derived one:

`c_t^plan = [p(x_{K-1}) - p(s_t), ..., p(x_{K-H_Q}) - p(s_t)]`

The actor predicts an action chunk conditioned on the current state and that local planner context. This keeps planner output as a
guidance signal for low-level control, while keeping the action supervision entirely grounded in real offline behavior.

## Why we execute only the first `C` actions

The action chunk is not committed open-loop by default. We execute only the first `C` actions, then replan from the newly observed
state. This reduces compounding model error and makes the system more robust when either the planner context or low-level policy is
slightly inaccurate. The default setting is `C = 1`, so every executed action is followed by a fresh planner call and a fresh local
context.
