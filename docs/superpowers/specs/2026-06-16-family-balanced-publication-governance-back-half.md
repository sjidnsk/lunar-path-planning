# Family-Balanced Publication Governance Back Half v1

## Summary

This stage family resumes release governance after the passed
`Family-Balanced Formal Performance Claim / Release Decision v1`. It creates a
family-balanced-only publication governance chain and ends at the
family-balanced default-policy candidate sandbox install preflight.

The chain is authorization, evidence freeze, package, sandbox dry-run,
consumer replay/canary, and default-policy-candidate preflight work only. It is
not default policy installation.

## Stages

- `family_balanced_scoped_claim_publication_evidence_freeze`
- `family_balanced_checkpoint_publication_authorization_preflight`
- `family_balanced_checkpoint_publication_package_preparation`
- `family_balanced_checkpoint_publication_package_verification`
- `family_balanced_checkpoint_publication_sandbox_install_dry_run_preflight`
- `family_balanced_checkpoint_publication_sandbox_install_dry_run`
- `family_balanced_checkpoint_publication_sandbox_install_dry_run_verification`
- `family_balanced_checkpoint_publication_sandbox_consumer_replay_canary`
- `family_balanced_default_policy_candidate_authorization_preflight`
- `family_balanced_default_policy_candidate_sandbox_install_preflight`

## Acceptance

Each stage writes an independent output root under
`outputs/path_feedback_batch_family_balanced_*_v1/` with summary, manifest,
audit, rejection report, and report artifacts. The final summary must report
`status=passed`, `reason_codes=[]`, and
`preflight_verdict=eligible_for_family_balanced_default_policy_candidate_sandbox_install_candidate`.

All stages must keep the release boundary closed:
`checkpoint_publication_approved=false`,
`default_policy_replacement_approved=false`,
`real_executor_connection_approved=false`, `publishes_checkpoint=false`,
`replaces_default_policy=false`, and `connects_real_executor=false`.

The final gate must audit kill-switch, rollback, telemetry, lineage,
default-policy read-only boundary, and path-planner isolation. It must state
不替换 default policy and 不连接真实执行器.

## Validation

```bash
PY=/home/kai/anaconda3/envs/lunar-explorer/bin/python
$PY -m pytest tests/test_family_balanced_*.py -q
PYTHON=$PY bash scripts/run_family_balanced_scoped_claim_publication_evidence_freeze.sh
PYTHON=$PY bash scripts/run_family_balanced_checkpoint_publication_authorization_preflight.sh
PYTHON=$PY bash scripts/run_family_balanced_checkpoint_publication_package_preparation.sh
PYTHON=$PY bash scripts/run_family_balanced_checkpoint_publication_package_verification.sh
PYTHON=$PY bash scripts/run_family_balanced_checkpoint_publication_sandbox_install_dry_run_preflight.sh
PYTHON=$PY bash scripts/run_family_balanced_checkpoint_publication_sandbox_install_dry_run.sh
PYTHON=$PY bash scripts/run_family_balanced_checkpoint_publication_sandbox_install_dry_run_verification.sh
PYTHON=$PY bash scripts/run_family_balanced_checkpoint_publication_sandbox_consumer_replay_canary.sh
PYTHON=$PY bash scripts/run_family_balanced_default_policy_candidate_authorization_preflight.sh
PYTHON=$PY bash scripts/run_family_balanced_default_policy_candidate_sandbox_install_preflight.sh
jq '{status,reason_codes,preflight_verdict,next_required_change,checkpoint_publication_approved,default_policy_replacement_approved,real_executor_connection_approved}' \
  outputs/path_feedback_batch_family_balanced_default_policy_candidate_sandbox_install_preflight_v1/family-balanced-default-policy-candidate-sandbox-install-preflight-summary.json
git diff --check
```

## Non-Goals

No PPO training, rollout expansion, reward change, network/action space/default
A* change, guard relaxation, checkpoint publication to a real publication path,
default policy replacement, real executor connection, real-world performance
claim, Ackermann-feasible trajectory claim, or treating IRIS/GCS/path-planner
diagnostics as release proof.
