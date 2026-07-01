# Stage26.7G Repair Behavior-Policy KL Baseline

## Goal

Stage26.7F proved that every PPO update combo failed the KL gate while loss and gradients stayed finite. The blocker is now a metric-contract issue: synthetic credit uses a behavior mixture policy, so PPO ratio must use behavior old logprob, but the update-stability KL gate must compare new policy against old policy logprob.

Stage26.7G repairs that distinction and reruns the bounded update sweep.

## Contract

- PPO ratio source: `new_policy_log_prob - old_behavior_log_prob`.
- KL gate source: `new_policy_log_prob - old_policy_log_prob`.
- Behavior KL remains diagnostic only.
- `max_abs_approx_kl` stays fixed at `1.5`.
- No reward, network, Hybrid A*, synthetic terrain, default policy, executor, release, or canary behavior changes.

## Implementation

- Update `scripts/run_xunce_stage21_4_tiny_ppo_update_smoke.py` to emit behavior-vs-policy KL fields.
- Add `scripts/run_xunce_stage26_7g_repair_behavior_policy_kl_baseline.py`.
- Add `configs/xunce_stage26_7g_repair_behavior_policy_kl_baseline_v1.json`.
- Add `tests/test_xunce_stage26_7g_repair_behavior_policy_kl_baseline.py`.
- Register `xunce-stage26-7g-repair-behavior-policy-kl-baseline`.

## Audits

- Zero-update baseline audit: policy KL should be near zero; behavior KL may remain high.
- Update sweep audit: stable combos are judged by policy KL, finite loss/grad, checkpoint reload, and boundary fields.
- Post-update eval audit: only stable combos run Stage26.3, and conclusions use `main_coverage_per_100m_delta`.

## Verification

```powershell
python -m pytest tests\test_xunce_stage26_7g_repair_behavior_policy_kl_baseline.py tests\test_xunce_stage26_7f_synthetic_credit_ppo_update_stability_sweep.py tests\test_xunce_stage21_4_tiny_ppo_update_smoke.py tests\test_platform_stage_runner.py -q --basetemp outputs\pytest-stage26-7g
python -m py_compile scripts\run_xunce_stage26_7g_repair_behavior_policy_kl_baseline.py scripts\run_xunce_stage21_4_tiny_ppo_update_smoke.py scripts\run_xunce_stage26_2_synthetic_terrain_ppo_update_smoke.py scripts\run_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke.py
python scripts\run_stage.py --stage xunce-stage26-7g-repair-behavior-policy-kl-baseline --dry-run
```
