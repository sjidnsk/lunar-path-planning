# Xunce Stage 18.9 Strict V3 Evidence Run

## Goal

Run the clean Stage 18.9 evidence path with canonical reward/guard profile v3.
The 6/12/24/36 candidate-count sweeps must be regenerated from Stage 18.4E
through Stage 18.9 with the same `xunce-coverage-cost-risk-boundary-v3`
`profile_hash`. Existing v2 lineage remains read-only history and must not be
used as final readiness evidence.

## Scope

- Add a v3 threshold adapter for Stage 18 scripts:
  `scripts/xunce_stage18_guard_thresholds.py`.
- Add strict v3 configs for Stage 18.4E, 18.5, 18.6, 18.7, and 18.9.
- Add strict v3 rollup:
  `scripts/run_xunce_stage18_9_strict_v3_evidence_rollup.py`.
- Register `xunce-stage18-9-strict-v3-evidence-rollup` in
  `configs/stage_registry.json`.
- Keep Stage 18.6 and Stage 18.7 as diagnostics only. Stage 18.9 trajectory
  audit remains the readiness authority.

## Output Root

Large artifacts go to D drive:

```text
D:\CodexDownloads\lunar-path-planning\stage18_9_strict_v3_evidence_run
```

The four sweep roots are:

```text
count_006: candidate_count=6,  proposal_pool=48
count_012: candidate_count=12, proposal_pool=96
count_024: candidate_count=24, proposal_pool=192
count_036: candidate_count=36, proposal_pool=288
```

## Rollup Artifacts

```text
xunce-stage18-9-strict-v3-evidence-summary.json
xunce-stage18-9-strict-v3-count-results.jsonl
xunce-stage18-9-strict-v3-next-stage-routing.json
xunce-stage18-9-strict-v3-report.md
xunce-stage18-9-strict-v3-manifest.json
```

## Routing

- Missing inputs, non-v3 lineage, or profile hash mismatch:
  `rerun_stage18_9_strict_v3_required_inputs`
- Any hard risk violation:
  `repair_path_risk_boundary_filtering`
- Coverage gain, path cost, or coverage efficiency failure:
  `refine_coverage_cost_reward_weights`
- Coverage/cost pass but soft risk exposure is abnormal:
  `calibrate_soft_risk_exposure_weight`
- At least one count has trajectory-clean Stage 18.9 evidence:
  `prepare_stage19_evaluator_critic_preflight`

`stage19_authorized` must remain false for every route.

## Verification

```powershell
python -m pytest tests\test_xunce_stage18_guard_thresholds.py tests\test_xunce_stage18_9_strict_v3_evidence_rollup.py -q
python -m py_compile scripts\xunce_stage18_guard_thresholds.py scripts\run_xunce_stage18_9_strict_v3_evidence_rollup.py
python scripts\run_stage.py --stage xunce-stage18-9-strict-v3-evidence-rollup --dry-run
```

## Non-Goals

- Do not start PPO.
- Do not publish checkpoints.
- Do not replace the default policy.
- Do not connect a real executor.
- Do not start canary traffic.
- Do not modify action space, network, or default A*.
- Do not treat Stage 18.6/18.7 candidate diagnostics as readiness authority.
