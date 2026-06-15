# Checkpoint Publication Package Preparation v1

## Summary

Stage 10 prepares the Stage 9-authorized selected experimental PPO checkpoint as
an isolated checkpoint publication package. It copies the checkpoint and
metadata only into the Stage 10 audit output package root, then freezes the
source/package identity, metadata, manifest, lineage, rollback, and release
boundary evidence.

This stage is package preparation, not checkpoint publication. It does not
publish checkpoint artifacts, replace the default policy, connect a real
executor, run PPO, modify network/action space/default A*, relax guards, or
claim Ackermann-feasible trajectory.

## Inputs

- Stage 9 summary and audits from
  `outputs/path_feedback_batch_checkpoint_publication_authorization_preflight_v1/`.
- Stage 8 scoped claim publication freeze summary.
- Stage 7 formal performance claim/release decision summary.
- Selected candidate promotion preflight summary.
- Selected checkpoint and metadata:
  - seed `0`
  - budget `epochs1_lr3e-6`
  - SHA-256 `9d9539c685ab965739c91958bf9cbfe90329c460b4bdbcc35881875aa62f0aa2`
  - size `17189`

## Outputs

Output root:
`outputs/path_feedback_batch_checkpoint_publication_package_preparation_v1/`

Package root:
`outputs/path_feedback_batch_checkpoint_publication_package_preparation_v1/checkpoint-publication-package/`

Artifacts:

- `checkpoint-publication-package-preparation-summary.json`
- `checkpoint-publication-package-manifest.json`
- `checkpoint-publication-package-hash-audit.json`
- `checkpoint-publication-package-metadata-audit.json`
- `checkpoint-publication-package-lineage-audit.json`
- `checkpoint-publication-package-release-boundary-audit.json`
- `checkpoint-publication-package-rollback-audit.json`
- `checkpoint-publication-package-rejection-report.json`
- `checkpoint-publication-package-preparation-report.md`
- `checkpoint-publication-package/experimental-hybrid-policy-candidate.pt`
- `checkpoint-publication-package/experimental-hybrid-policy-candidate-metadata.json`

## Pass Contract

- `status=passed`
- `reason_codes=[]`
- `package_preparation_verdict=prepared_for_checkpoint_publication_package_verification`
- `checkpoint_publication_package_prepared=true`
- `package_manifest_audit_passed=true`
- `package_hash_audit_passed=true`
- `package_metadata_audit_passed=true`
- `lineage_audit_passed=true`
- `rollback_audit_passed=true`
- `release_boundary_audit_passed=true`
- source/package checkpoint SHA-256 and size are identical
- `next_required_change=checkpoint_publication_package_verification`
- `checkpoint_publication_approved=false`
- `default_policy_replacement_approved=false`
- `real_executor_connection_approved=false`
- `publishes_checkpoint=false`
- `replaces_default_policy=false`
- `connects_real_executor=false`

## Failure Reasons

- `stage9_not_passed`
- `stage9_not_authorized_for_package_preparation`
- `source_checkpoint_missing`
- `source_metadata_invalid`
- `source_checkpoint_identity_mismatch`
- `package_checkpoint_hash_mismatch`
- `package_checkpoint_size_mismatch`
- `package_metadata_mismatch`
- `package_manifest_incomplete`
- `unexpected_publication_path`
- `rollback_boundary_invalid`
- `release_boundary_violation`
- `docs_not_updated`

## Documentation

Update and keep consistent:

- `README.md`
- `docs/算法设计与系统架构报告.md`
- `docs/superpowers/specs/2026-06-15-checkpoint-publication-package-preparation.md`

## Verification

```bash
PY=/home/kai/anaconda3/envs/lunar-explorer/bin/python
$PY -m pytest tests/test_checkpoint_publication_package_preparation.py tests/test_checkpoint_publication_authorization_preflight.py -q
PYTHON=$PY bash scripts/run_checkpoint_publication_package_preparation.sh
jq '{status,reason_codes,package_preparation_verdict,checkpoint_publication_package_prepared,package_hash_audit_passed,checkpoint_publication_approved,default_policy_replacement_approved,real_executor_connection_approved}' outputs/path_feedback_batch_checkpoint_publication_package_preparation_v1/checkpoint-publication-package-preparation-summary.json
sha256sum outputs/path_feedback_batch_checkpoint_publication_package_preparation_v1/checkpoint-publication-package/experimental-hybrid-policy-candidate.pt
git diff --check
```

## Non-Goals

- 不发布 checkpoint。
- 不替换 default policy。
- 不连接真实执行器。
- 不运行 PPO 或扩大 rollout。
- 不修改 reward、network/action space/default A*。
- 不放松 guard。
- 不声明真实世界性能。
- 不声明 Ackermann-feasible trajectory。
- 不把 IRIS/GCS/path-planner 诊断当作 release proof。
