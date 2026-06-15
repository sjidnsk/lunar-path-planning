# Scoped Claim Publication & Evidence Freeze v1

## Summary

Stage 8 freezes the Stage 7 scoped offline performance claim into a publication
bundle. It is a claim-publication and evidence-indexing gate only: it does not
train PPO, publish checkpoint artifacts, replace the default policy, connect a
real executor, or expand the performance claim beyond the current offline
guarded shadow/canary evidence.

The runner is `scripts/run_scoped_claim_publication_evidence_freeze.py`, with a
shell wrapper at `scripts/run_scoped_claim_publication_evidence_freeze.sh`.
Outputs are written under
`outputs/path_feedback_batch_scoped_claim_publication_evidence_freeze_v1/`.

## Outputs

- `scoped-claim-publication-evidence-freeze-summary.json`
- `scoped-claim-publication-bundle-manifest.json`
- `scoped-claim-publication-claim-text.md`
- `scoped-claim-publication-scope-audit.json`
- `scoped-claim-publication-evidence-freeze-ledger.json`
- `scoped-claim-publication-doc-consistency-audit.json`
- `scoped-claim-publication-release-boundary-audit.json`
- `scoped-claim-publication-rejection-report.json`
- `scoped-claim-publication-evidence-freeze-report.md`

## Acceptance

The passing summary must report `status=passed`, `reason_codes=[]`,
`publication_verdict=approved_for_scoped_claim_publication`, and
`scoped_claim_publication_approved=true`. The claim scope must remain
`scoped_offline_guarded_shadow_canary_only`, and the next required change must
be `checkpoint_publication_authorization_preflight`.

The publication claim may mention the Stage 7 scoped metrics only inside the
current offline guarded shadow/canary and same-decision-set baseline boundary:
coverage return improvement, cumulative coverage rate delta improvement,
valuable area covered improvement, fallback rate, and no coverage efficiency
regression.

Release actions must remain closed:
`checkpoint_publication_approved=false`,
`default_policy_replacement_approved=false`,
`real_executor_connection_approved=false`, `publishes_checkpoint=false`,
`replaces_default_policy=false`, and `connects_real_executor=false`.

## Failure Reasons

- `stage7_not_passed`
- `claim_scope_overbroad`
- `claim_metric_mismatch`
- `docs_not_updated`
- `release_boundary_violation`
- `evidence_freeze_incomplete`
- `prohibited_claim_present`

## Documentation

Update `README.md` and `docs/算法设计与系统架构报告.md` with the Stage 8
status, output root, scoped claim boundary, and explicit non-goals. The docs
must state that Stage 8 不发布 checkpoint, 不替换 default policy, 不连接真实执行器,
不声明真实世界性能, 不声明 Ackermann-feasible trajectory, and does not treat
IRIS/GCS/path-planner diagnostics as release proof.

## Validation

```bash
PY=/home/kai/anaconda3/envs/lunar-explorer/bin/python
$PY -m pytest tests/test_scoped_claim_publication_evidence_freeze.py tests/test_formal_performance_claim_release_decision.py -q
PYTHON=$PY bash scripts/run_scoped_claim_publication_evidence_freeze.sh
jq '{status,reason_codes,publication_verdict,scoped_claim_publication_approved,checkpoint_publication_approved,default_policy_replacement_approved,real_executor_connection_approved}' outputs/path_feedback_batch_scoped_claim_publication_evidence_freeze_v1/scoped-claim-publication-evidence-freeze-summary.json
git diff --check
```

## Non-Goals

Stage 8 does not run PPO training, expand rollout, modify reward, publish a
checkpoint, package a checkpoint, replace the default policy, connect a real
executor, relax guard, modify network/action space/default A*, declare
Ackermann-feasible trajectory, or promote Stage 7 into real-world performance.
