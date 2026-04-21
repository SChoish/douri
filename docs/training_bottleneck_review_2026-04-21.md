# Training bottleneck review (JIT/JAX/loop focus)

Date: 2026-04-21 (UTC)

## Scope
- Joint training entrypoint: `main.py`
- GOUB planner/learner: `agents/goub_dynamics.py`
- DQC critic + SPI actor: `agents/critic/dqc.py`, `agents/critic/actor.py`
- Dataset hot paths: `utils/dqc_sequence_dataset.py`, `utils/deas_sequence_dataset.py`, `utils/datasets.py`

---

## Top bottleneck candidates (priority order)

### P0) Host↔device transfer and Python orchestration in the inner training step
In the per-step loop, batches are sampled on CPU (`numpy`) and repeatedly converted to JAX arrays inside jitted updates. The loop also performs multiple Python-side calls and `np.asarray` materializations in the actor-batch build path.

Why this is likely expensive:
- The training loop runs per-step Python orchestration over data sampling, GOUB update, critic update, actor rescore, actor update.
- Actor batch construction repeatedly crosses device boundaries (`jnp` outputs immediately cast back to `np`).

Code points:
- `main.py`: step loop and batch build/update sequencing (`for _ in range(spe)` + `_build_actor_batch_from_goub` + update calls).
- `main.py`: `_build_actor_batch_from_goub` uses repeated `np.asarray(...)` around jitted model calls and repeatedly stacks candidate trajectories/actions.

Suggested fix:
1. Keep data and intermediates on device as long as possible (delay `np.asarray` until logging only).
2. Fuse actor-batch build + rescore into one jitted function (or at least a smaller number of device calls).
3. Convert the per-step Python loop to `lax.scan` for update kernels where feasible (at minimum for GOUB+critic updates with prebuilt batches).

Expected upside: typically the largest wall-time win in this codebase.

---

### P0) Candidate-plan generation uses Python loop over `plan_candidates`
For `plan_candidates > 1`, trajectory candidates are produced in Python by repeatedly splitting RNG and calling `sample_plan`/`plan` one-by-one.

Why this is likely expensive:
- Many small kernel launches instead of one vectorized launch.
- Repeated host control flow and synchronization points.

Code points:
- `main.py`: `_build_actor_batch_from_goub` loop over `for _ in range(plan_candidates - 1)`.

Suggested fix:
- Add a GOUB method that generates all candidates in one call (e.g., `vmap(sample_plan)` with candidate RNGs), returning `[B, N, T, D]` directly.

Expected upside: medium to high when `plan_candidates` increases.

---

### P1) `score_action_chunks` called frequently outside explicit `jit`
Critic scoring is invoked for proposal ranking and actor rescoring. The method contains shape-dependent logic and repeated reshape/repeat operations.

Why this can bottleneck:
- Frequent execution path during every training step.
- Potential recompilation pressure if candidate count / shapes change across runs.

Code points:
- `agents/critic/dqc.py`: `score_action_chunks`.
- `main.py`: calls in `_build_actor_batch_from_goub` and `_rescore_actor_batch_for_update`.

Suggested fix:
- Provide a jitted scorer path with fixed candidate-count signature used by training.
- Pre-flatten proposal tensors once and avoid repeated `repeat/reshape` where possible.

---

### P1) Dataset sampling allocates many temporary arrays every step
Both DQC/DEAS sequence datasets build multiple index arrays and derived tensors on each `sample` call.

Why this can bottleneck:
- High-frequency CPU allocations in hot path.
- Python + NumPy index composition each step.

Code points:
- `utils/dqc_sequence_dataset.py`: `sample` (index creation, chunk reshape, rewards/masks creation).
- `utils/deas_sequence_dataset.py`: `sample` (multi-axis index composition, mask creation).

Suggested fix:
- Preallocate/reuse index buffers for common batch sizes.
- Move repeated per-sample computations (where shape-static) into cached templates.
- Consider prefetch queue (CPU worker thread/process) so GPU/TPU is not waiting on sampling.

---

### P2) Duplicate critic scoring for actor path
Current flow scores candidates once during actor batch build and again after critic update for actor update input.

Why this may matter:
- Necessary for freshness, but doubles scorer work in actor-enabled runs.

Code points:
- First score: `_build_actor_batch_from_goub`.
- Rescore: `_rescore_actor_batch_for_update`.

Suggested fix:
- Keep only post-critic-update score for training-critical ranking.
- If pre-score metrics are needed, compute lightweight summaries less frequently (e.g., every K steps).

---

## JIT stability / recompilation risks to watch

1. Static arguments:
- `GOUBDynamicsAgent.sample_plan` has `noise_scale` as static arg; changing it frequently can trigger new compile artifacts.

2. Shape polymorphism:
- Candidate-dependent shapes (`plan_candidates`, `proposal_topk`) can create separate compiled traces if changed between runs.

3. Large PyTree arguments in jitted actor update:
- `JointActorAgent.update` receives `critic_agent`; ensure this path does not induce unnecessary retracing when structure changes.

---

## Quick measurement plan (low effort, high signal)

1. Enable timing metrics and confirm dominant phase:
```bash
python main.py --measure_timing=True --train_epochs=3 --log_every_n_epochs=1
```
Check `time/build_*`, `time/critic_update_*`, `time/actor_rescore_*`, `time/data_*`.

2. Compare actor-path cost:
- Run with and without `use_spi_actor`.
- Run with `plan_candidates=1` vs `>1`.

3. Inspect compile behavior:
- Set `JAX_LOG_COMPILES=1` and verify no unexpected per-step recompiles.

---

## Recommended optimization order

1. **Vectorize candidate plan generation** (remove Python loop in `_build_actor_batch_from_goub`).
2. **Reduce host/device ping-pong** (keep arrays on device until final logging).
3. **JIT/fuse proposal scoring path** for fixed training shapes.
4. **Add dataset prefetching / buffer reuse**.
5. **Optionally reduce duplicate scoring frequency** for monitoring-only metrics.

---

## Bottom line
The most probable bottleneck is **Python + host orchestration around candidate planning/scoring and repeated array materialization**, not the raw neural-network forward/backward kernels themselves. The existing timing hooks are already good; use them to validate and then attack P0 items first.
