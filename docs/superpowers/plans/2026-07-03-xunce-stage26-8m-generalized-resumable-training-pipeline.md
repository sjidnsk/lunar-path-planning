# Stage26.8M Generalized Resumable Training Pipeline

## Goal

Stage26.8M turns the specialized Stage26.8D/8H/8I recovery runners into a
configurable, artifact-driven training pipeline. It manages long-running
synthetic terrain PPO experiments as resumable jobs instead of one long command.

This stage changes orchestration only. It does not change PPO loss, reward,
network structure, Hybrid A* search semantics, default A*, candidate generation,
synthetic terrain generation, or deployment boundaries.

## Job Model

Each job is expanded from:

```text
horizon
seed
scenario_count
collector_rollout_steps
eval_rollout_steps
update_combo_id
```

The job id format is:

```text
h{horizon}_s{seed}_sc{scenario_count}_cr{collector_rollout_steps}_er{eval_rollout_steps}_u{combo_id}
```

Each job advances through:

```text
collector -> update -> eval_pre -> eval_post -> aggregate
```

## State Placement

All experiment state and large artifacts live under the D-drive output root:

```text
D:/CodexDownloads/lunar-path-planning/stage26_synthetic_terrain_augmentation/outputs/path_feedback_batch_xunce_stage26_8m_generalized_resumable_training_pipeline_v1
```

The C-drive repository stores only source, configs, tests, and documentation.

## Run Modes

- `run_next`: run the first pending phase, capped by `max_jobs_per_invocation`.
- `aggregate_only`: scan existing artifacts and rewrite summary/state/report.
- `run_job`: run the next pending phase for a selected job.
- `run_phase`: run a selected phase only when dependencies are complete.

Stage26.8M v1 does not support destructive rerun. Failed or stale artifacts
require a new output root or a future explicit repair mode.

## Completion Rules

- `collector`: Stage26.1 summary passed and safety/lineage counters are clean.
- `update`: Stage26.2 summary passed, policy KL is within threshold, checkpoint
  reload passed, and checkpoint is experimental-only.
- `eval_pre` / `eval_post`: high-fidelity summary passed and required inference
  and episode JSONL files exist.
- `aggregate`: Stage26.3 summary exists and binding/safety/lineage fields are
  readable, even if the route says policy signal remains insufficient.

## Routing

- Pending jobs route to `continue_stage26_8m_jobs`.
- Collector blockers route to `repair_stage26_8m_collector_binding_or_safety`.
- Update/KL/checkpoint blockers route to `repair_stage26_8m_update_stability`.
- Eval/aggregate binding or safety blockers route to
  `repair_stage26_8m_eval_binding_or_safety`.
- Clean action unchanged results route to
  `increase_stage26_synthetic_update_strength_or_sample_count`.
- Action changed with negative unit-distance main coverage routes to
  `repair_stage26_synthetic_credit_assignment`.
- Action changed with nonnegative efficiency routes to
  `resume_stage26_8d_seed_horizon_jobs_with_diverse_scenarios`.
- Majority positive configured jobs route to
  `run_stage26_9_synthetic_terrain_long_horizon_efficiency_pilot`.

## Validation

```powershell
python -m pytest tests\test_xunce_stage26_8m_generalized_resumable_training_pipeline.py tests\test_xunce_stage26_8d_resumable_seed_horizon_execution.py tests\test_xunce_stage26_8h_resumable_diverse_scenario_post_update_eval.py tests\test_xunce_stage26_8i_diverse_scenario_policy_signal_strength_repair.py tests\test_platform_stage_runner.py -q --basetemp outputs\pytest-stage26-8m

python -m py_compile scripts\run_xunce_stage26_8m_generalized_resumable_training_pipeline.py

python scripts\run_stage.py --stage xunce-stage26-8m-generalized-resumable-training-pipeline --dry-run
```
