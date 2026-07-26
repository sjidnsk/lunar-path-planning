# Midterm Hopper Internal Simulation Proxy Authority Design Addendum

## Decision

For the reduced midterm G2/G3 experiment, add one explicit, project-internal,
generic computational Hopper simulation-proxy parameter set:

`hopper_generic_internal_computational_simulation_proxy_midterm_g2g3/v1`

This is an additive authority capability. It is not a hardware model, does not
name or certify a real robot, and does not change the repository default Hopper
profile, action generation, search semantics, terrain semantics, safety
thresholds, executor state, checkpoint publication, default policy, or canary
state.

The upstream project authorization is frozen at:

- record:
  `.superpowers/sdd/2026-07-26-midterm-dual-gate-experiment/project-authorization.md`
- SHA-256:
  `720e11ef04ad2b57283421809a077ccf1f0b35167a9f482082241398ad0214d2`

That authorization permits implementation and generation, but it does not by
itself make any artifact formally eligible. Formal eligibility remains
fail-closed until the exact implementation, regression, independent-input,
technical-review, and approval artifacts are bound by SHA-256.

## Frozen parameter semantics

All numeric values are exact binary64 values:

| Field | Value | Exact binary64 |
|---|---:|---|
| body envelope radius | `0.375 m` | `0x1.8000000000000p-2` |
| launch reference height | `0.750 m` | `0x1.8000000000000p-1` |
| arc clearance margin | `0.125 m` | `0x1.0000000000000p-3` |
| landing footprint radius | `0.625 m` | `0x1.4000000000000p-1` |

The derivation grid is `0.5 m`:

- body envelope is half a cell diagonal rounded upward to the `0.125 m`
  quantum;
- arc margin is one quarter cell;
- launch height is effective arc radius `0.5 m` plus half a cell;
- landing footprint is body envelope plus half a cell.

The stop condition accepts only exact finite positive speeds
`speed_mps <= 2.5`; therefore the existing `1.5`, `2.0`, and `2.5 m/s` actions
pass and the existing `3.0 m/s` actions remain generated but fail the stop
check. The energy value is the dimensionless exact evaluation
`(speed_mps / 2.5) ** 2`.

Frozen identities:

- base profile:
  `hopper-generic-internal-computational-simulation-proxy-midterm/v1`
- capability revision:
  `simulation_proxy_generic_internal_lunar_ballistic/v2`
- stop condition:
  `hopper_generic_internal_same_height_capture_le_2p5mps_stop_computational_simulation_proxy/v1`
- stop evaluator:
  `hopper_generic_internal_stop_evaluator_exact_binary64/v1`
- energy model:
  `hopper_generic_internal_launch_speed_squared_normalized_2p5mps_relative_energy_computational_simulation_proxy/v1`
- energy evaluator:
  `hopper_generic_internal_relative_energy_evaluator_exact_binary64/v1`

## Two-layer authority

The path-planner implementation layer may recognize and execute the frozen
parameter semantics. Its record proves only that trusted code supports the
parameter set. It must never contain or infer project formal-evidence approval.

The root Task 7 evidence layer resolves formal eligibility. It must reject:

- the existing Gate5B test fixture;
- an implementation record without a separate immutable candidate record;
- a candidate record without an artifact-bound approval record;
- an approval whose authorization, source, test, reviewer, or independent-input
  hashes drift;
- any record claiming physical capability or third-party certification.

Only a separately materialized approved record may set
`formal_evidence_eligible=true`. It must bind the immutable candidate,
implementation commit and source hashes, regression evidence, independent
technical audit, G2 input bundle and attestations, and the upstream
authorization hash.

## Preserved invariants

- Hopper action generation remains exactly
  `4 speeds × 3 elevations × 16 azimuths = 192` actions in the same order.
- Lunar gravity, landing distribution, `0.99` unconditioned selected mass,
  `15°` landing slope, same-height support, no midcourse correction, and no
  inflight observation remain unchanged.
- `max_traversable_slope_deg=30.0` and `risk_weight=0.0` remain exact.
- Map-outside landing mass is never renormalized into the map.
- Provider, jump L2, route L2, and API return checks continue to reseal
  authority and independently recompute primitive and route semantics.
- The existing Gate5B fixture record, values, tokens, blocker, and
  `formal_evidence_eligible=false` remain unchanged.
- Default configuration retains six `null` fields; the new stack is opt-in for
  the reduced midterm G2/G3 experiment only.

## Authority and lineage requirements

The implementation registry is immutable, strictly sorted, duplicate-free, and
contains both the unchanged Gate5B fixture and the new implementation record.
Provider, jump L2, and route L2 must resolve the same exact record.

The in-memory seal includes exact evaluator object identities. Deterministic
lineage includes model IDs, evaluator IDs, and evaluator source digests, but
never memory addresses or `repr()` output. One-ULP numeric drift, evaluator
replacement, registry mutation, profile mutation, or provider/L2 mismatch
fails closed as `hopper_authority_contract_mismatch`.

## Eligibility boundary

Until the separate Task 7 candidate, independent audit, input bundle, approval,
and approved record all exist and all current hashes match:

```text
hopper_internal_proxy_implementation_supported = true
hopper_formal_simulation_proxy_parameter_set_approved = false
g2_formal_status = blocked
g3_formal_status = blocked
```

No diagnostic success may be rendered as a formal G2/G3 pass or as evidence of
real Hopper hardware capability.
