# Stage 14 Checkpoint Publication Sandbox Install Dry-Run Verification v1

## Summary

Stage 14 is a read-only verification of the Stage 13 sandbox install dry-run.
It recomputes source/sandbox checkpoint identity, checks the Stage 13 install
manifest, re-loads the sandbox checkpoint, verifies sandbox metadata, and
rechecks lineage, rollback, and release boundaries.

It is not checkpoint publication, not a consumer smoke test, and not release
approval. It must not copy, overwrite, move, or delete checkpoint/metadata
files.

## Runner

- `scripts/run_checkpoint_publication_sandbox_install_dry_run_verification.py`
- `scripts/run_checkpoint_publication_sandbox_install_dry_run_verification.sh`

Inputs include Stage 13 summary and audits, Stage 12 preflight evidence,
Stage 11 package verification evidence, Stage 10 package manifest/files, and
Stage 9/8/7 summaries for lineage and release-boundary checks.

## Output Contract

Output root:

`outputs/path_feedback_batch_checkpoint_publication_sandbox_install_dry_run_verification_v1/`

Required artifacts:

- `checkpoint-publication-sandbox-install-dry-run-verification-summary.json`
- `checkpoint-publication-sandbox-consumer-verification-manifest.json`
- `checkpoint-publication-sandbox-installed-file-identity-audit.json`
- `checkpoint-publication-sandbox-install-manifest-consistency-audit.json`
- `checkpoint-publication-sandbox-consumer-load-reverification-audit.json`
- `checkpoint-publication-sandbox-metadata-verification-audit.json`
- `checkpoint-publication-sandbox-lineage-verification-audit.json`
- `checkpoint-publication-sandbox-release-boundary-audit.json`
- `checkpoint-publication-sandbox-rollback-verification-audit.json`
- `checkpoint-publication-sandbox-install-dry-run-verification-rejection-report.json`
- `checkpoint-publication-sandbox-install-dry-run-verification-report.md`

## Pass Contract

- `status=passed`
- `reason_codes=[]`
- `verification_verdict=verified_for_checkpoint_publication_sandbox_consumer_smoke_preflight`
- `checkpoint_publication_sandbox_install_dry_run_verification_passed=true`
- `checkpoint_publication_sandbox_consumer_smoke_preflight_approved=true`
- `source_package_checkpoint_sha256=9d9539c685ab965739c91958bf9cbfe90329c460b4bdbcc35881875aa62f0aa2`
- `sandbox_consumer_checkpoint_sha256=9d9539c685ab965739c91958bf9cbfe90329c460b4bdbcc35881875aa62f0aa2`
- `source_package_checkpoint_size_bytes=17189`
- `sandbox_consumer_checkpoint_size_bytes=17189`
- `next_required_change=checkpoint_publication_sandbox_consumer_smoke_preflight`
- `checkpoint_publication_approved=false`
- `default_policy_replacement_approved=false`
- `real_executor_connection_approved=false`
- `publishes_checkpoint=false`
- `replaces_default_policy=false`
- `connects_real_executor=false`

## Failure Reasons

- `stage13_not_passed`
- `stage13_not_ready_for_verification`
- `sandbox_install_manifest_missing`
- `sandbox_checkpoint_missing`
- `sandbox_metadata_missing`
- `source_package_checkpoint_missing`
- `source_package_metadata_missing`
- `sandbox_identity_mismatch`
- `source_package_identity_mismatch`
- `sandbox_install_manifest_inconsistent`
- `sandbox_load_reverification_failed`
- `sandbox_metadata_invalid`
- `default_policy_boundary_violation`
- `rollback_boundary_invalid`
- `lineage_verification_incomplete`
- `release_boundary_violation`
- `docs_not_updated`

## Documentation

Keep these docs aligned with Stage 14 artifacts and boundaries:

- `README.md`
- `docs/算法设计与系统架构报告.md`
- `docs/superpowers/specs/2026-06-16-checkpoint-publication-sandbox-install-dry-run-verification.md`

## Validation

```bash
PY=/home/kai/anaconda3/envs/lunar-explorer/bin/python
$PY -m pytest tests/test_checkpoint_publication_sandbox_install_dry_run_verification.py tests/test_checkpoint_publication_sandbox_install_dry_run.py -q
PYTHON=$PY bash scripts/run_checkpoint_publication_sandbox_install_dry_run_verification.sh
jq '{status,reason_codes,verification_verdict,checkpoint_publication_sandbox_install_dry_run_verification_passed,checkpoint_publication_sandbox_consumer_smoke_preflight_approved,checkpoint_publication_approved,default_policy_replacement_approved,real_executor_connection_approved}' outputs/path_feedback_batch_checkpoint_publication_sandbox_install_dry_run_verification_v1/checkpoint-publication-sandbox-install-dry-run-verification-summary.json
sha256sum outputs/path_feedback_batch_checkpoint_publication_sandbox_install_dry_run_preflight_v1/checkpoint-publication-sandbox-install-dry-run-sandbox/experimental-hybrid-policy-candidate.pt
git diff --check
```

## Non-Goals

不发布 checkpoint。Stage 14 does not copy, overwrite, move, or delete
checkpoint/metadata files; run PPO; expand rollout; modify reward; modify
network/action space/default A*; relax guards; replace default policy; connect
a real executor; claim real-world performance; claim Ackermann-feasible
trajectory; or treat IRIS/GCS/path-planner diagnostics as release proof.
