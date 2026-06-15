# Checkpoint Publication Authorization Preflight v1

## Summary

Stage 9 reviews whether the selected experimental PPO checkpoint is eligible to
enter checkpoint publication package preparation. It is an authorization
preflight only: it does not publish checkpoint artifacts, copy the checkpoint
into a publication path, replace the default policy, connect a real executor, or
change model behavior.

The runner is `scripts/run_checkpoint_publication_authorization_preflight.py`,
with a shell wrapper at
`scripts/run_checkpoint_publication_authorization_preflight.sh`. Outputs are
written under
`outputs/path_feedback_batch_checkpoint_publication_authorization_preflight_v1/`.

## Inputs

- Stage 8 scoped claim publication/evidence freeze summary and audits.
- Stage 7 formal performance claim/release decision summary.
- Stage 6 shadow/canary release performance validation summary.
- Stage 5B.7 cost-efficiency-aware coverage reward candidate-filter summary.
- Formal training, post-training replay, and selected candidate promotion
  preflight summaries.
- Selected checkpoint and metadata from the selected candidate summary.

The current selected checkpoint is seed `0`, budget `epochs1_lr3e-6`, SHA-256
`9d9539c685ab965739c91958bf9cbfe90329c460b4bdbcc35881875aa62f0aa2`.

## Outputs

- `checkpoint-publication-authorization-preflight-summary.json`
- `checkpoint-publication-candidate-manifest.json`
- `checkpoint-publication-identity-audit.json`
- `checkpoint-publication-metadata-audit.json`
- `checkpoint-publication-load-evidence-audit.json`
- `checkpoint-publication-lineage-audit.json`
- `checkpoint-publication-release-boundary-audit.json`
- `checkpoint-publication-authorization-matrix.json`
- `checkpoint-publication-rejection-report.json`
- `checkpoint-publication-authorization-preflight-report.md`

## Acceptance

The passing summary must report `status=passed`, `reason_codes=[]`,
`authorization_verdict=eligible_for_checkpoint_publication_package_preparation`,
`checkpoint_publication_authorization_preflight_passed=true`, and
`checkpoint_publication_package_preparation_approved=true`.

Identity, metadata, load evidence, lineage, and release-boundary audits must all
pass. The next required change must be
`checkpoint_publication_package_preparation`.

Release actions remain closed:
`checkpoint_publication_approved=false`,
`default_policy_replacement_approved=false`,
`real_executor_connection_approved=false`, `publishes_checkpoint=false`,
`replaces_default_policy=false`, and `connects_real_executor=false`.

## Failure Reasons

- `stage8_not_passed`
- `claim_scope_overbroad`
- `checkpoint_missing`
- `checkpoint_hash_mismatch`
- `checkpoint_size_mismatch`
- `checkpoint_metadata_invalid`
- `checkpoint_load_evidence_not_passed`
- `lineage_incomplete`
- `release_boundary_violation`
- `docs_not_updated`

## Documentation

Update `README.md` and `docs/算法设计与系统架构报告.md` with the Stage 9
status, output root, authorization boundary, and next stage. The docs must state
that Stage 9 不发布 checkpoint, 不替换 default policy, 不连接真实执行器, 不运行 PPO,
不修改 network/action space/default A*, 不放松 guard, 不声明 Ackermann-feasible
trajectory, and does not treat IRIS/GCS/path-planner diagnostics as release
proof.

## Validation

```bash
PY=/home/kai/anaconda3/envs/lunar-explorer/bin/python
$PY -m pytest tests/test_checkpoint_publication_authorization_preflight.py tests/test_scoped_claim_publication_evidence_freeze.py -q
PYTHON=$PY bash scripts/run_checkpoint_publication_authorization_preflight.sh
jq '{status,reason_codes,authorization_verdict,checkpoint_publication_authorization_preflight_passed,checkpoint_publication_package_preparation_approved,checkpoint_publication_approved,default_policy_replacement_approved,real_executor_connection_approved}' outputs/path_feedback_batch_checkpoint_publication_authorization_preflight_v1/checkpoint-publication-authorization-preflight-summary.json
git diff --check
```

## Non-Goals

Stage 9 does not run PPO, expand rollout, modify reward, copy checkpoint to a
publication path, publish checkpoint, replace the default policy, connect a real
executor, modify network/action space/default A*, relax guard, declare
Ackermann-feasible trajectory, or promote IRIS/GCS/path-planner diagnostics to
release proof.
