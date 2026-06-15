# Stage 12 Checkpoint Publication Sandbox Install Dry-Run Preflight v1

## Summary

Stage 12 is a preflight for the future `checkpoint_publication_sandbox_install_dry_run`.
It consumes the Stage 11 verified checkpoint package evidence and checks whether
the package can safely enter a sandbox install dry-run. It is not the sandbox
install dry-run itself and is not checkpoint publication.

The stage may write audit outputs under
`outputs/path_feedback_batch_checkpoint_publication_sandbox_install_dry_run_preflight_v1/`.
It must not copy checkpoint files into the sandbox, must not copy to
publication/default/live/executor paths, must not install a policy, must not
replace the default policy, and must not connect a real executor.

## Runner

- `scripts/run_checkpoint_publication_sandbox_install_dry_run_preflight.py`
- `scripts/run_checkpoint_publication_sandbox_install_dry_run_preflight.sh`

The runner reads Stage 11 summary, consumer manifest, integrity/load/lineage/
release/rollback audits; Stage 10 package manifest and package metadata; and
Stage 9/8/7 summaries for lineage and release-boundary cross-checks.

## Output Contract

Output root:

`outputs/path_feedback_batch_checkpoint_publication_sandbox_install_dry_run_preflight_v1/`

Required artifacts:

- `checkpoint-publication-sandbox-install-dry-run-preflight-summary.json`
- `checkpoint-publication-sandbox-preflight-manifest.json`
- `checkpoint-publication-sandbox-package-consumer-audit.json`
- `checkpoint-publication-sandbox-default-policy-boundary-audit.json`
- `checkpoint-publication-sandbox-path-boundary-audit.json`
- `checkpoint-publication-sandbox-lineage-audit.json`
- `checkpoint-publication-sandbox-release-boundary-audit.json`
- `checkpoint-publication-sandbox-rollback-preflight-audit.json`
- `checkpoint-publication-sandbox-install-dry-run-preflight-rejection-report.json`
- `checkpoint-publication-sandbox-install-dry-run-preflight-report.md`

## Pass Contract

- `status=passed`
- `reason_codes=[]`
- `preflight_verdict=eligible_for_checkpoint_publication_sandbox_install_dry_run`
- `checkpoint_publication_sandbox_install_dry_run_preflight_passed=true`
- `checkpoint_publication_sandbox_install_dry_run_approved=true`
- `package_checkpoint_sha256=9d9539c685ab965739c91958bf9cbfe90329c460b4bdbcc35881875aa62f0aa2`
- `package_checkpoint_size_bytes=17189`
- `next_required_change=checkpoint_publication_sandbox_install_dry_run`
- `checkpoint_publication_approved=false`
- `default_policy_replacement_approved=false`
- `real_executor_connection_approved=false`
- `publishes_checkpoint=false`
- `replaces_default_policy=false`
- `connects_real_executor=false`

The sandbox manifest only declares a future sandbox root and planned consumer
checkpoint path. The planned checkpoint path must not exist after Stage 12.
Do not copy to publication/default/live/executor paths.
不复制到发布/default/live/executor 路径。

## Failure Reasons

- `stage11_not_passed`
- `stage11_not_authorized_for_sandbox_preflight`
- `consumer_manifest_missing`
- `package_checkpoint_missing`
- `package_identity_mismatch`
- `package_load_not_verified`
- `sandbox_path_boundary_violation`
- `default_policy_boundary_violation`
- `lineage_incomplete`
- `rollback_preflight_invalid`
- `release_boundary_violation`
- `docs_not_updated`

## Documentation

Keep these docs aligned with the implemented artifact paths, pass contract, and
release boundaries:

- `README.md`
- `docs/算法设计与系统架构报告.md`
- `docs/superpowers/specs/2026-06-16-checkpoint-publication-sandbox-install-dry-run-preflight.md`

## Validation

```bash
PY=/home/kai/anaconda3/envs/lunar-explorer/bin/python
$PY -m pytest tests/test_checkpoint_publication_sandbox_install_dry_run_preflight.py tests/test_checkpoint_publication_package_verification.py -q
PYTHON=$PY bash scripts/run_checkpoint_publication_sandbox_install_dry_run_preflight.sh
jq '{status,reason_codes,preflight_verdict,checkpoint_publication_sandbox_install_dry_run_preflight_passed,checkpoint_publication_sandbox_install_dry_run_approved,checkpoint_publication_approved,default_policy_replacement_approved,real_executor_connection_approved}' outputs/path_feedback_batch_checkpoint_publication_sandbox_install_dry_run_preflight_v1/checkpoint-publication-sandbox-install-dry-run-preflight-summary.json
git diff --check
```

## Non-Goals

Stage 12 does not run PPO, expand rollout, modify reward, modify network/action
space/default A*, relax guards, publish checkpoint, replace default policy,
connect a real executor, claim real-world performance, claim
Ackermann-feasible trajectory, or treat IRIS/GCS/path-planner diagnostics as
release proof.
