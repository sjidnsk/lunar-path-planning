# Checkpoint Publication Package Verification v1

## Summary

Stage 11 independently verifies the isolated checkpoint package created by
Stage 10. It treats the Stage 10 package as the source of truth for consumer
paths, recomputes checkpoint identity, validates metadata, verifies that the
checkpoint can be loaded, and rechecks lineage, rollback, and release
boundaries.

This stage is package verification only. It is not checkpoint publication,
default-policy replacement, sandbox install dry-run, real executor connection,
or release approval.

## Inputs

- Stage 10 package preparation summary and audits from
  `outputs/path_feedback_batch_checkpoint_publication_package_preparation_v1/`.
- Stage 10 package files under
  `outputs/path_feedback_batch_checkpoint_publication_package_preparation_v1/checkpoint-publication-package/`.
- Stage 9 summary, candidate manifest, and load-evidence audit.
- Stage 8 and Stage 7 summaries for lineage and release-boundary checks.

## Outputs

Output root:
`outputs/path_feedback_batch_checkpoint_publication_package_verification_v1/`

Artifacts:

- `checkpoint-publication-package-verification-summary.json`
- `checkpoint-publication-package-consumer-manifest.json`
- `checkpoint-publication-package-integrity-audit.json`
- `checkpoint-publication-package-manifest-consistency-audit.json`
- `checkpoint-publication-package-metadata-verification-audit.json`
- `checkpoint-publication-package-load-verification-audit.json`
- `checkpoint-publication-package-lineage-verification-audit.json`
- `checkpoint-publication-package-release-boundary-audit.json`
- `checkpoint-publication-package-rollback-verification-audit.json`
- `checkpoint-publication-package-verification-rejection-report.json`
- `checkpoint-publication-package-verification-report.md`

## Pass Contract

- `status=passed`
- `reason_codes=[]`
- `verification_verdict=verified_for_checkpoint_publication_sandbox_install_dry_run_preflight`
- `checkpoint_publication_package_verification_passed=true`
- `checkpoint_publication_sandbox_install_dry_run_preflight_approved=true`
- `package_integrity_audit_passed=true`
- `package_manifest_consistency_audit_passed=true`
- `package_metadata_verification_audit_passed=true`
- `package_load_verification_audit_passed=true`
- `lineage_verification_audit_passed=true`
- `rollback_verification_audit_passed=true`
- `release_boundary_audit_passed=true`
- `package_checkpoint_sha256=9d9539c685ab965739c91958bf9cbfe90329c460b4bdbcc35881875aa62f0aa2`
- `package_checkpoint_size_bytes=17189`
- `next_required_change=checkpoint_publication_sandbox_install_dry_run_preflight`
- `checkpoint_publication_approved=false`
- `default_policy_replacement_approved=false`
- `real_executor_connection_approved=false`
- `publishes_checkpoint=false`
- `replaces_default_policy=false`
- `connects_real_executor=false`

## Failure Reasons

- `stage10_not_passed`
- `stage10_not_prepared_for_package_verification`
- `package_manifest_missing`
- `package_checkpoint_missing`
- `package_metadata_missing`
- `package_hash_mismatch`
- `package_size_mismatch`
- `package_manifest_inconsistent`
- `package_metadata_invalid`
- `package_load_verification_failed`
- `lineage_verification_incomplete`
- `rollback_boundary_invalid`
- `release_boundary_violation`
- `docs_not_updated`

## Documentation

Update and keep consistent:

- `README.md`
- `docs/算法设计与系统架构报告.md`
- `docs/superpowers/specs/2026-06-15-checkpoint-publication-package-verification.md`

## Verification

```bash
PY=/home/kai/anaconda3/envs/lunar-explorer/bin/python
$PY -m pytest tests/test_checkpoint_publication_package_verification.py tests/test_checkpoint_publication_package_preparation.py -q
PYTHON=$PY bash scripts/run_checkpoint_publication_package_verification.sh
jq '{status,reason_codes,verification_verdict,checkpoint_publication_package_verification_passed,checkpoint_publication_sandbox_install_dry_run_preflight_approved,checkpoint_publication_approved,default_policy_replacement_approved,real_executor_connection_approved}' outputs/path_feedback_batch_checkpoint_publication_package_verification_v1/checkpoint-publication-package-verification-summary.json
sha256sum outputs/path_feedback_batch_checkpoint_publication_package_preparation_v1/checkpoint-publication-package/experimental-hybrid-policy-candidate.pt
git diff --check
```

## Non-Goals

- 不发布 checkpoint。
- 不替换 default policy。
- 不连接真实执行器。
- 不运行 PPO 或扩大 rollout。
- 不重新封包或复制 checkpoint 到 default/live/release/executor 路径。
- 不修改 reward、network/action space/default A*。
- 不放松 guard。
- 不声明真实世界性能。
- 不声明 Ackermann-feasible trajectory。
- 不把 IRIS/GCS/path-planner 诊断当作 release proof。
