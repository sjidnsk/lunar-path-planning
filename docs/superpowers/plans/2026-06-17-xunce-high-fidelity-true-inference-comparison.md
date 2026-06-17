# Xunce High-Fidelity True Inference Comparison Plan

## Summary

Implement Stage 18 as an offline, read-only extension after Xunce Stage 15-17 governance:

- refresh the Stage 15-17 evidence chain before using Stage 18 outputs;
- expand LOLA quasi-real evidence to 24 slices / 8 ROI groups;
- upgrade high-fidelity comparison from path-feedback proxy selection to true checkpoint inference.

This plan does not approve default-policy replacement, checkpoint publication, real executor connection, online canary traffic, PPO training, network changes, action-space changes, or default A* changes.

## Implementation Notes

- `scripts/run_xunce_high_fidelity_real_map_comparison.py` must load the Xunce sandbox candidate checkpoint and incumbent experimental policy checkpoint read-only.
- Both models must score the same high-fidelity observation/candidate batch.
- Per-scenario artifacts must include logits, masked logits, action probabilities, selected action/rank, value, latency, finite-output flags, and mask-violation flags.
- Efficiency audit must use actual model parameter counts and actual median inference latencies, not checkpoint file size.
- Path-feedback candidate rows are allowed only as input evidence for candidate construction and cost/risk comparison; they must not select the Xunce action.
- If advantage is not established under true inference, the next required change is `xunce_research_iteration_required`.
- Existing LOLA LDEM/LDEC quasi-real data is the default Stage 18 data source; do not download more map products unless 24/8 evidence remains low-spread or an illumination/shadow-specific question requires it.

## Acceptance Criteria

- `TorchPolicyScorer.score()` remains backward compatible.
- `TorchPolicyScorer.score_detail()` exposes full masked inference detail.
- Comparison summary includes `true_model_inference_executed`, `proxy_selection_used`, checkpoint load flags, selected counts, finite-output count, mask-violation count, real parameter counts, real latency medians, and `latency_ratio_vs_incumbent`.
- Comparison writes `xunce-high-fidelity-model-inference-audit.json` and `xunce-high-fidelity-model-inference-results.jsonl`.
- Missing or invalid Xunce checkpoint routes to `fix_xunce_sandbox_candidate_preflight`.
- Missing or unsupported incumbent checkpoint routes to `fix_incumbent_policy_checkpoint`.
- Guard fallback threshold failures route to `fix_xunce_high_fidelity_guard_fallback`.
- No-advantage evidence passes only to `xunce_research_iteration_required`.
- Documentation records that Stage 18 remains research evidence, not release approval.

## Validation

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_xunce_high_fidelity_real_map_comparison.py -q
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest model-explorer/tests/test_model_explorer.py::TorchPolicyNetworkTests model-explorer/tests/test_model_explorer.py::PolicyTrainingTests -q
```

## Execution Result - 2026-06-17

The full offline evidence chain was refreshed on Windows with
`D:\conda_envs\lunar-explorer\python.exe`. Stage 18A passed with 24 LOLA
quasi-real slices and 8 ROI groups. Stage 18B then executed true checkpoint
inference for both models on the same high-fidelity scenario batch.

Key artifacts:

- `outputs/path_feedback_batch_xunce_high_fidelity_real_map_roi_expansion_v1/xunce-high-fidelity-real-map-roi-expansion-summary.json`
- `outputs/path_feedback_batch_xunce_high_fidelity_real_map_comparison_v1/xunce-high-fidelity-real-map-comparison-summary.json`
- `outputs/path_feedback_batch_xunce_high_fidelity_real_map_comparison_v1/xunce-high-fidelity-model-inference-audit.json`
- `outputs/path_feedback_batch_xunce_high_fidelity_real_map_comparison_v1/xunce-high-fidelity-model-inference-results.jsonl`
- `outputs/path_feedback_batch_value_stability_candidate_v1/experimental-hybrid-policy-candidate-standardization-audit.json`

Observed result:

- `true_model_inference_executed=true`
- `proxy_selection_used=false`
- `xunce_checkpoint_loaded=true`
- `incumbent_checkpoint_loaded=true`
- `model_inference_finite_output_count=48`
- `model_inference_mask_violation_count=0`
- `xunce_model_selected_count=24`
- `incumbent_model_selected_count=24`
- `xunce_better_than_incumbent_count=0`
- `xunce_worse_than_incumbent_count=16`
- `controlled_regression_count=16`
- `xunce_parameter_count=19171`
- `incumbent_parameter_count=2066`
- `latency_ratio_vs_incumbent=2.8214`
- `next_required_change=xunce_research_iteration_required`

The incumbent value-stability checkpoint was generated upstream as a
`controlled-hybrid-policy-candidate-checkpoint/v1`. For Stage 18B loading it was
standardized in-place to the `model-explorer-masked-policy/v2` wrapper using the
same trained `model_state_dict`; the original controlled-hybrid file was backed
up in the same ignored output root. This did not run a new training update and
did not publish or replace any checkpoint.

Conclusion: Stage 18B produced real model inference evidence, but did not
establish Xunce advantage. The next research step is data/evaluation iteration,
especially ROI spread, candidate discrimination, and mask-safety versus
performance-evidence separation. This remains offline research evidence only,
not real lunar release evidence.
