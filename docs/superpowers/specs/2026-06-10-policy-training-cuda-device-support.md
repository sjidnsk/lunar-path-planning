# Policy Training CUDA Device Support v1

## Summary

Current readiness is `guarded_ppo_rollout_pilot_evaluated`. The guarded pilot
can collect PPO-trainable transitions, run one tiny PPO update, and pass
post-update raw/sequential/collector gates. This stage adds an opt-in
CPU/CUDA/auto device contract for policy training compute only. Default behavior
stays CPU so prior evidence semantics do not change.

## Interfaces

- Config:
  - `configs/policy_training_cuda_device_support_v1.json`
  - `training.device`: `cpu`, `cuda`, or `auto`
- Scripts:
  - `scripts/run_policy_training_cuda_device_support_smoke.py/.sh`
- Evidence root:
  - `outputs/path_feedback_batch_policy_training_cuda_device_support_v1/`
- Readiness:
  - new argument `--policy-training-cuda-device-support-summary`
  - new status `policy_training_cuda_device_support_evaluated`

## Behavior

`cpu` always resolves to CPU. `auto` resolves to CUDA when
`torch.cuda.is_available()` is true, otherwise falls back to CPU and records
`fallback_to_cpu=true`. Explicit `cuda` fails if CUDA is unavailable with
`cuda_requested_but_unavailable`.

`train_policy_on_episodes` and `run_limited_ppo_update_smoke.py` move the policy
network and all PPO tensors to the resolved device. Old log-prob/value checks,
loss, gradient, KL, and clipping are computed on that same device. Checkpoint
state dicts are serialized back to CPU so artifacts remain portable and
loadable with `map_location="cpu"`.

Guarded and iterative summaries preserve update-device provenance from the
limited PPO update step.

## Acceptance

- Default configs without `training.device` still resolve to CPU.
- `device=auto` resolves to CUDA on the current RTX 5070 Ti machine.
- `device=cuda` fails when CUDA is unavailable.
- GPU smoke summary `status=passed`.
- `optimizer_train_transition_count>=24`.
- `old_log_prob_max_abs_error<=1e-4`.
- `old_value_max_abs_error<=1e-4`.
- loss, grad, reward, return, and advantage are finite.
- `parameter_l2_delta>0`.
- `abs(approx_kl)<=0.25`.
- `max_grad_norm_after_clip<=1.0`.
- checkpoint can be loaded on CPU.
- readiness may report `policy_training_cuda_device_support_evaluated`.

## Non-goals

- No formal PPO rollout.
- No real-map training input.
- No checkpoint publication.
- No default policy replacement.
- No network, action-space, or default-A* change.
- No distance/path-risk/source-selection contract relaxation.
- No Ackermann-feasible trajectory claim.
- No IRIS/GCS/path-planner diagnostic treated as training release evidence.
- No policy performance claim.
