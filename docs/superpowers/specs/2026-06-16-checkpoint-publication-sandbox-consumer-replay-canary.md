# Stage 15 Checkpoint Publication Sandbox Consumer Replay/Canary v1

Stage 15 consumes the Stage 14 verified sandbox checkpoint in an isolated
consumer replay/canary harness. It performs CPU checkpoint load and policy
forward on deterministic `PolicyObservation` inputs, writes step telemetry,
fallback, rollback, lineage, and release-boundary audits, and advances only to
`default_policy_candidate_authorization_preflight`.

Output root:
`outputs/path_feedback_batch_checkpoint_publication_sandbox_consumer_replay_canary_v1/`

Pass contract:
`status=passed`, `reason_codes=[]`,
`consumer_replay_canary_verdict=eligible_for_default_policy_candidate_authorization_preflight`,
`checkpoint_publication_sandbox_consumer_replay_canary_passed=true`,
`default_policy_candidate_authorization_preflight_approved=true`,
`consumer_step_count>=64`, no missing observation, no invalid/empty action mask,
no non-finite logits/log_prob/value, `controlled_regression_count=0`,
`fallback_rate<0.5`, and telemetry/rollback/release audits passed.

Boundaries stay closed: no checkpoint publication, no default policy
replacement, no real executor connection, no PPO, no rollout, no guard
relaxation, no network/action-space/default-A* change, no real-world or
Ackermann-feasible trajectory claim.
