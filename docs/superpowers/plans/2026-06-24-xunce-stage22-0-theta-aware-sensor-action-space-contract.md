# Stage22.0 Theta-Aware Sensor Action Space Contract

## Summary

Stage22.0 establishes the contract for theta-aware exploration actions. The
previous Stage21 PPO line selects a candidate point `(x,y)`. The new contract is
a candidate viewpoint `(x,y,theta_deg)`, because the rover can stand at the same
cell and observe different terrain depending on sensor heading.

This stage is read-only. It audits existing Stage21 artifacts, recomputes a
simple theta-aware sensor footprint offline, and decides whether the next stage
must implement theta-aware candidate viewpoint generation.

## Sensor Contract

- Default theta bins: `0,45,90,135,180,225,270,315`.
- Coordinate convention: `0` degrees points toward positive x, `90` degrees
  points toward positive y.
- Default FOV: `90` degrees.
- Initial range: same scale as the old coverage radius.
- Occlusion: audit-only in Stage22.0.
- Path planning is unchanged: A* still plans to `(x,y)`. Theta only affects
  coverage after arrival.

## Artifacts

- `scripts/xunce_theta_sensor_coverage.py`
- `scripts/run_xunce_stage22_0_theta_aware_sensor_action_space_contract.py`
- `configs/xunce_stage22_0_theta_aware_sensor_action_space_contract_v1.json`
- `xunce-stage22-0-summary.json`
- `xunce-stage22-0-theta-action-contract-audit.json`
- `xunce-stage22-0-sensor-footprint-audit.jsonl`
- `xunce-stage22-0-action-space-expansion-audit.json`
- `xunce-stage22-0-ppo-contract-impact.json`
- `xunce-stage22-0-next-stage-routing.json`

## Routing

- Missing input lineage: `rerun_stage22_0_required_inputs`.
- Theta not material to coverage: `document_theta_optional_sensor_model`.
- Theta material but Stage21 batch is point-only:
  `implement_stage22_1_theta_aware_candidate_viewpoint_generation`.
- Theta-aware contract already complete:
  `run_stage22_2_theta_aware_coverage_reward_contract`.

## Boundaries

Stage22.0 does not start PPO, publish a checkpoint, replace the default policy,
connect an executor, start canary traffic, modify the network, or modify default
A*. Old point-only Stage21 PPO evidence cannot be used as theta-aware readiness
when theta materially changes coverage or action preference.
