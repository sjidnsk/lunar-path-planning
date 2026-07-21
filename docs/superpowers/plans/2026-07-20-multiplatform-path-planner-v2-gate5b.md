# Multi-platform capability-aware path planning v2 — Gate5B approved implementation plan

## Approval, authority, and hard boundaries

- High-level option A was approved before the three section contracts.
- Sections 1, 2, and 3 were explicitly approved on 2026-07-20.
- The user explicitly approved the independent readiness/implementation contract on 2026-07-21.
- The user explicitly approved the Gate5B cadence/cardinal correction A2 on 2026-07-21 with the exact statement `批准 Gate5B cadence/cardinal correction A2`.
- This tracked file is the sole self-contained implementation authority for Gate5B slices 11B1-11B7, including the serial 11B1.5 A2 repair slice. The ignored source drafts and proposal listed below are provenance only.
- Authorization includes the plan-only promotion and A2-amendment commits, serial TDD work, fresh reviews, allowed commits/integrations, and bounded non-formal verification under the approved D-drive temp roots.
- Authorization excludes the controller, any formal Gate5 data access, `D:/xunce/out/path_v2/g5`, a formal blocked snapshot, Gate6, checkpoint publication, default-policy replacement, executor connection, and canary start.
- v1 remains the default and v2 remains opt-in. The protected C-drive Stage6 dirty worktree is never read, executed in, modified, or used as evidence.
- PPO Stage1 may retain only the exact inherited 13-nodeid failure set; the set may not expand.

## Promotion provenance

- Pre-promotion parent anchor: `0b6c12d8a5228a7f64f5b86ad0e6f0ba6181bb4e`.
- Pre-promotion nested HEAD and parent gitlink: `57c28fb2709aec4e043827a50425906a4cbddecc`.
- Promotion is valid only if its sole tree change is this file and the nested gitlink remains unchanged.

- `.superpowers/sdd/gate5b-design-decision-brief.md`: SHA-256 `76ecbf1352de91a4164028977ffcdc55b61826be23e5c68a2ea06e72b46ed02b`
- `.superpowers/sdd/gate5b-section1-continuous-clearance-draft.md`: SHA-256 `b09ef6e4007cdbc672bb45ae435281e902574b7dfa31004a111632aed0081fc8`
- `.superpowers/sdd/gate5b-section2-fixture-authority-draft.md`: SHA-256 `5f924aececd95901bfd9d1d034107b34f3b628ade2502bb1f49b4c24fdc61db9`
- `.superpowers/sdd/gate5b-section3-route-authority-draft.md`: SHA-256 `3981c81efa2c6dca1836f61c7b588bcd583c72d298e8071b33fb1f96443ced18`
- `.superpowers/sdd/gate5b-implementation-readiness-plan.md`: SHA-256 `176d5fee8fb9db051c5df15e95741f6e0e09ce9ff4217cac9171a0da6e0c6231`
- `.superpowers/sdd/gate5b-cadence-cardinal-correction-proposal.md`: SHA-256 `51c1f151f3513e4d7e1f87423fd31d82c2be7f53c3b0bead2fc64c1d9371f458`; this ignored proposal records A2 provenance, while this tracked plan contains the complete operative contract.
- A2 amendment parent anchor: `2f0e05e006f70148ba9dcc334a47c5bf53957ea5`.
- A2 amendment nested implementation anchor: `59d012375745a862fbdd9c92a92875a671e82ff3`; the parent may show only the expected unstaged `path-planner` gitlink drift while this plan-only amendment is committed.

## Selected high-level Option A — Same-height nominal-mean simulation proxy

- Every authoritative observed launch-footprint cell must have the same canonical elevation; that exact common value is launch support height `H0`. No mean, minimum, maximum, or hidden tolerance is allowed.
- Ballistic reference starts at `H0 + launch_reference_height_m` and returns to the same reference height.
- Every cell touched by every selected landing-zone footprint must have canonical elevation exactly equal to `H0`; otherwise the jump fails closed with a dedicated height-unreachable reason.
- The probability zone is a per-hop safety tube. Every selected landing cell means its entire closed cell square, dilated by the landing footprint radius, must pass. The theoretical mean pose and its footprint are checked independently.
- After a successful hop, the supported stop model performs an explicit instantaneous simulation-only reset to that checked nominal mean state. It does not claim or validate a physical recentering path from an actual touchdown.
- Each hop must cover at least `0.99` unconditioned probability mass. A multi-hop route may report only a union-bound diagnostic and must not claim whole-route probability `>=0.99`.
- Advantages: preserves the Gate5A public `sample_ballistic_arc()` and generic capped-helper behavior exactly, adds no hidden tolerance, and supports deterministic multi-hop search. Gate5B's versioned discrete-action helper uses exact cardinal directions for indices `0/4/8/12` while retaining the same ballistic model; those four Hopper sample byte strings intentionally need not equal the generic raw-libm residual bytes.
- Cost: supports only equal-height landing plateaus and relies on an explicit simulation-only nominal-state assumption.

## Part I — Approved Section 1 continuous-clearance contract

### Approval scope

- Gate5B high-level option A is approved.
- The user explicitly approved Section 1 on 2026-07-20.
- The user explicitly approved A2 on 2026-07-21; it narrowly replaces Gate5B operational sample cadence and discrete cardinal-direction authority as frozen below, without changing the Gate5A public/generic contract.
- It freezes only terrain height authority, continuous arc clearance, numeric enclosure, caps, and hard-obstacle overflight semantics.
- Section 1 approval by itself did not approve later sections or implementation. Sections 2 and 3 were separately approved on 2026-07-20, and readiness/implementation was separately approved on 2026-07-21. A still-later separate controller authorization remains mandatory for formal output.

### Versioned model

- Terrain height authority: `terrain_cell_piecewise_constant_top_surface_proxy/v1`.
- Airborne envelope authority: `hopper_euclidean_xy_disk_full_vertical_clearance_product_proxy/v1`.
- Each in-bounds non-hard fine cell is a closed vertical prism with a piecewise-constant top at its authoritative observed `elevation_m`.
- A hard-obstacle cell has no authoritative top-height provenance in the current terrain schema and is therefore an unbounded forbidden column in Gate5B v1.
- No bilinear interpolation, hidden subcell relief, or inferred obstacle height is permitted in v1.

### Launch and arc geometry

- Let `rho` be the exact-rational sum of the two canonical binary64 values `body_envelope_radius_m` and `arc_clearance_margin_m`; a rounded binary64 sum is evidence only and may not weaken the authority comparison.
- Gate5B v1 deliberately uses a separable conservative product proxy: exact Euclidean XY disk distance `<= rho` selects a terrain prism, and every selected non-hard prism then requires the full vertical clearance `z >= elevation + rho`. This is not an exact 3D sphere-to-prism distance claim and can reject a diagonally separated path that a true sphere model would accept; it cannot create a collision false-pass.
- Every launch-footprint cell intersecting the closed horizontal disk of radius `rho` must be in bounds, observed, traversable, non-hard, and within the 30-degree platform boundary.
- Every launch-footprint elevation must be exactly equal in canonical binary64. That common elevation is `H0`; `z_launch = H0 + launch_reference_height_m` must be representable and `launch_reference_height_m >= rho`.
- From the sealed profile and exact speed/elevation/azimuth indices, independently rebuild and reseal the Gate5B A2 canonical binary64 values `flight_time`, `horizontal_dx`, `horizontal_dy`, and `apex_height`, then lift those exact binary64 rationals into the continuous normalized curve `u=t/flight_time`, `x=x0+horizontal_dx*u`, `y=y0+horizontal_dy*u`, `z=z0+4*apex_height*u*(1-u)`. The four cardinal action indices use A2's exact direction table; the other twelve use raw `sin/cos` of their canonical azimuth words.
- Operational samples are independently replayed with the captured private `_sample_hopper_ballistic_arc_capped_v2` object from the sealed `HopperResourceAuthorityV2`, never with the public wrapper or generic capped helper. Their rounded coordinates need not equal evaluation of the exact-real continuous curve; sample partition and continuous-curve safety remain distinct authorities.

#### A2 operational cadence and discrete-direction authority

Gate5A public `sample_ballistic_arc()` and generic `sample_ballistic_arc_capped_v2()` retain their existing signatures, arbitrary-finite-azimuth behavior, bytes, failure priority, and versioned generic helper semantics. Gate5B alone adds this private, non-exported helper with the exact signature:

```python
def _sample_hopper_ballistic_arc_capped_v2(
    start: BallisticStartV2,
    speed_mps: float,
    elevation_rad: float,
    azimuth_index: int,
    g_mps2: float,
    *,
    max_sample_count: int,
) -> tuple[BallisticSampleV2, ...]: ...
```

- `azimuth_index` is an exact built-in `int`, rejects `bool` and subclasses, and must be in `0..15`. The helper derives the canonical azimuth in exact binary64 operation order `2.0 * pi * azimuth_index / 16.0`; no caller supplies an azimuth float, direction vector, or cadence.
- Indices `0`, `4`, `8`, and `12` map exactly to `(1.0, 0.0)`, `(0.0, 1.0)`, `(-1.0, 0.0)`, and `(0.0, -1.0)` respectively, with every zero canonical positive zero. The other twelve indices use raw `cos(azimuth)` and `sin(azimuth)`. No tolerance, modulo, interval, nearest-direction snap, or post-hoc endpoint rewrite is permitted.
- The exact cardinal azimuth words are respectively `0x0.0p+0`, `0x1.921fb54442d18p+0`, `0x1.921fb54442d18p+1`, and `0x1.2d97c7f3321d2p+2`. Action authority, helper replay, continuous `horizontal_dx/horizontal_dy`, theoretical mean endpoint, landing-distribution orientation, primitive resources, provider L2, and API L2 all independently reconstruct and exact-word compare the same index-derived azimuth and direction semantics.
- Inside the helper, derive `vertical_speed = speed_mps * sin(elevation_rad)`, then `vertical_time_scale = vertical_speed / g_mps2`, then `flight_time = 2.0 * vertical_time_scale` in that exact order. The same `flight_time` word is the private core cadence. The production signature contains no `dt_s`; any extra positional or keyword cadence argument is rejected.
- After the existing finite/type/range, true nonzero-displacement, velocity/displacement/origin/apex representability, cap, and fail-closed checks, operational output is exactly two samples at exact times `(0.0, flight_time)`. The caller independently performs the same flight-time derivation and requires exact final-time equality, exact sample count `2`, and exact normalized times `(0,1)`; any mismatch is `hopper_numeric_contract_mismatch`.
- Fine cadence and `nextafter(flight_time, +/-inf)` cadence exist only at a private, non-exported direct-core adversarial seam fed by independently produced test samples. They are forbidden as production helper/provider/candidate/API/config inputs, module-global overrides, or serialized lineage. They may prove partition-equivalent safety but never define operational action bytes.
- The helper name stays absent from wildcard exports, `path_planner.v2.__init__`, every `__all__`, and every public export allowlist. Capturing its exact private module object in `HopperResourceAuthorityV2` is the sole operational entry.
- `dt_s` is not a field of any candidate, primitive, profile, parameter set, resource authority, validation result, public outcome, or deterministic lineage. The sealed profile, three exact action indices, helper/schema IDs, and operation order reconstruct it. This cadence is part of `hopper-jump-primitive/v1`, jump-oracle, and route-L2 semantics; changing it requires a version bump and new approval.

### Exact continuous horizontal overlap

Fresh captured Hopper-specific replay defines the unique operational analytic partition. Operational replay first requires exact `S=2`, so the only normalized interval is exact `[0,1]`; no interior/repair sample exists. The generic audit identity remains `1 + sum(S_i - 1)` for independently produced private direct-core adversarial samples: for consecutive exact samples `s_i,s_(i+1)`, lift canonical binary64 `time_s` and `flight_time` to rationals and define `[u_i,u_(i+1)] = [time_i/flight_time,time_(i+1)/flight_time]`, requiring exact endpoints, strict increase, and gap-free closed coverage. Any operational `S != 2`, altered time, or malformed private-test partition is `hopper_numeric_contract_mismatch`.

For each such adjacent analytic interval and each candidate cell square `B`:

1. Lift canonical binary64 inputs to exact rationals. For one axis with exact expanded interval `[L,U]`, grid origin `O`, and positive resolution `r`, enumerate the closed-cell superset from `ceil((L-O)/r)-1` through `floor((U-O)/r)`, inclusive. Apply the same formula on both axes before any clipping; this deliberately includes both cells at an exact grid boundary. OOB indices remain candidates until exact refinement, so a broad-phase false positive cannot become a boundary failure.
2. Refine each broad-phase cell with the exact closed condition that the XY centerline has Euclidean distance at most `rho` from `B`. This exact refinement is the horizontal half of the frozen separable product proxy; the L-infinity range is only a candidate broad phase.
3. Treat boundary contact as overlap; adjacent boundary-sharing cells are both checked.
4. Partition at exact rational x/y cell-boundary crossing parameters. On each closed region, reduce the exact squared-distance polynomial to `q(u)=A*u^2+B*u+C`. If `A=0`, require exact `B=0` and compare the constant `C` with `rho^2` over the whole region; `A=0,B!=0` is a numeric-kernel mismatch. If `A>0`, split at the exact vertex only when it lies inside the region.
5. Do not use the ordinary floating quadratic formula as authority. Rational roots remain exact; every irrational root uses exact `Q(sqrt(D))` representation and sign comparison. Ordered-float brackets are optional evidence only and never select or discard an endpoint.
6. The per-region result is a closed empty/interval/singleton set. Merge exact-touching adjacent results into a closed union, preserving singleton tangencies and exact grid-edge/corner contacts; no binary64 conversion may round a true overlap away.
7. Interval count, candidate width, candidate height, each per-interval Cartesian product, global distinct cells, and global interval-cell pair visits are separate counters. Each is checked before allocation/enumeration/insertion against `MAX_REPLAY_STEPS = 100_000`; satisfying one counter never excuses another. A finite work-cap excess fails as a replay-work resource failure.
8. Every prospective exact-integer numerator, denominator, discriminant intermediate, and root-comparison intermediate is pre-bounded before allocation by the single canonical constant `HOPPER_EXACT_MAX_INTEGER_BITS_V2 = 262_144`. An operation whose conservative output bound exceeds that limit is `hopper_numeric_contract_mismatch`; it is never attempted and is not reclassified as an ordinary memory-budget failure. Bare `Fraction`/big-int operations without this pre-reservation are forbidden. The separately approved Section 3 contract freezes a 128-slot exact arena and deterministic byte reservation outside this Section 1 approval scope; no request memory setting can waive the Section 1 numeric ceiling, and neither design approval authorizes implementation.

The implementation must have an independent exact reference oracle for adversarial tests covering nonzero origins, grid boundaries, tangency, subnormal gaps, large representable coordinates, algebraic roots between adjacent floats, and `nextafter` pairs.

### Continuous vertical clearance

- For each connected closed XY-overlap component, evaluate the exact normalized Gate5B A2 curve at its exact rational/`Q(sqrt(D))` endpoints.
- Since the normalized curve is concave, the smaller endpoint value is the exact lower bound over that component.
- Apply the frozen product-proxy comparison `z_min >= exact(elevation_m[cell]) + rho`; equality meets the required clearance, while any exact value below fails. Ordinary binary64 evaluation cannot authorize a pass. This full vertical margin is intentionally more conservative than exact 3D sphere-to-prism distance near a horizontal edge or corner.
- Any non-finite/absorbed derived offset, inconsistent sample, snapshot/profile/query drift, or unrepresentable bound fails closed.
- A private direct-core alternate cadence may change only analytic work partitioning; it must not turn a true collision/unknown/boundary contact into a pass. Production has no caller-controlled cadence and always uses the unique `[0,1]` partition.

### Hard-obstacle altitude rule in this approved section

- During flight, any exact horizontal envelope overlap with an observed hard-obstacle cell fails as an unbounded forbidden column, regardless of `elevation_m`.
- At launch and throughout every landing footprint, a hard-obstacle cell likewise always fails.
- Overflight becomes eligible only in a future model version that adds an independently validated obstacle-top layer and provenance; current `elevation_m` cannot be reinterpreted as that missing authority.

### Stable semantic reasons

- Expanded envelope touching outside geometry: `hopper_arc_boundary_violation`.
- Required in-bounds cell unobserved: `hopper_arc_unknown`.
- Observed prism failing the clearance inequality, or envelope overlap with an unbounded hard column: `hopper_arc_clearance_violation`.
- Snapshot identity drift: shared `terrain_snapshot_identity_mismatch`.
- Snapshot content drift: shared `terrain_snapshot_hash_mismatch`.
- Terrain helper/output drift: shared `terrain_query_contract_mismatch`.
- Grid origin/resolution/shape or cell-square reconstruction drift: `hopper_terrain_geometry_contract_mismatch`.
- Non-finite, overflow, absorbed offset, inconsistent Gate5A replay, or exact-kernel arithmetic failure: `hopper_numeric_contract_mismatch`.
- Any of the independent public work counters exceeding its cap: `hopper_replay_work_budget_exceeded`.
- Deadline expiry: shared `planning_deadline_expired`.

At every checkpoint, request/deadline/profile/snapshot/geometry/query authority is resealed before selecting a semantic result. A detected authority mismatch overrides boundary/unknown/clearance; a cap or expired deadline then stops all later interval/cell work. The complete cross-layer total order is frozen in Section 3 rather than duplicated here.

### RED boundary matrix

- Safe endpoints with an unsafe or unknown cell only between the two operational samples; exact continuous overlap must still detect it.
- Exact disk-square tangency, an algebraic corner root between adjacent floats, and `nextafter` on both sides.
- Operational Hopper replay has exactly two samples, exact normalized times `(0,1)`, and one `[0,1]` interval; any extra/missing/nonmonotone/altered sample fails. Private direct-core fine/`nextafter` cadence tests include repair samples and prove the corresponding exact partitions cover `[0,1]` once without becoming a production input.
- Constant `q=0`, constant `q=rho^2`, constant safe-outside `q>rho^2`, and impossible `A=0,B!=0` kernel cases; closed components merge without losing a singleton.
- Arc AABB touching a grid line/corner, including nonzero and large origins.
- Overlap root rounded inward by a naive implementation.
- Normalized-curve minimum taken at exact overlap endpoints rather than the apex or sample endpoints; a fixture where direct-parabola and Gate5A normalized evaluation differ by one ULP.
- Clearance equality pass and one-ULP-below fail.
- Any airborne hard-column overlap fails even at high z; hard launch/landing always fail; a future top-height field is not inferred.
- Exact vertical false-pass where rounded binary64 values compare equal but exact `z_min < elevation + rho`.
- A diagonally separated case that an exact 3D sphere would accept but the documented product proxy rejects, proving the conservative capability boundary is intentional.
- Candidate cap checked before allocation; overflow and absorbed offsets return stable failures.
- All 16 canonical azimuth words and directions are exact; at nonzero and large world origins every one of the 192 frozen actions preserves its intended direction, while real nonzero displacement absorption, underflow, or overflow still fails globally. Near-cardinal arbitrary azimuths are never snapped.
- Generic public/capped helper outputs and failure priority remain byte-equivalent under A2; the private Hopper helper is absent from wildcard/top-level/`__all__` exports and rejects any `dt_s` argument.
- Exact lower/upper index formula at a grid boundary includes the two incident cells; a broad-phase-only OOB candidate that fails exact disk-square refinement does not cause a false boundary rejection.
- Repeated runs and different `PYTHONHASHSEED` values produce identical cells, reason, and evidence bytes.


## Part II — Approved Section 2 fixture/authority contract

### Approval scope

- Sections 1, 2, and 3 were explicitly approved on 2026-07-20; readiness/implementation was separately approved on 2026-07-21.
- A2 was explicitly approved on 2026-07-21 and narrowly amends flight-time/cadence reconstruction without adding a profile, parameter-set, primitive, resource, or outcome field.
- Section 2 approval by itself freezes this design contract only and authorizes no implementation; implementation authority comes only from the separate 2026-07-21 readiness/implementation approval and this tracked plan.
- It freezes one simulation-only algorithm fixture, stop/energy formulas, provider binding, and machine separation from formal Gate5 evidence.
- `HopperProfileV2` keeps its Gate5A field shape; no `fixture_only` or `formal_evidence_eligible` booleans are added to the profile itself.

### Exact algorithm fixture

- fixture parameter-set ID: `hopper_gate5b_algorithm_fixture/v1`.
- fixture base profile ID: `hopper-lunar-ballistic-gate5b-fixture/v1`.
- `body_envelope_radius_m = 0.25`.
- `launch_reference_height_m = 0.50`.
- `arc_clearance_margin_m = 0.10`.
- `landing_footprint_radius_m = 0.30`.
- stop model ID: `hopper_same_height_nominal_recenter_capture_le_3mps_stop_simulation_proxy/v1`.
- energy model ID: `hopper_launch_speed_squared_relative_energy/v1`.

These values are exact algorithm-test inputs only. They are not defaults, formal parameters, hardware dimensions, or physical capability evidence.

Registry comparison uses canonical binary64 words, not decimal tolerance: `0.25 -> 0x1.0000000000000p-2`, `0.50 -> 0x1.0000000000000p-1`, `0.10 -> 0x1.999999999999ap-4`, and `0.30 -> 0x1.3333333333333p-2`.

### Stop model

- The same-height landing zone, every dilated landing footprint, and the nominal mean footprint must first pass the independent Hopper L2 oracle.
- Under the frozen same-height, no-drag Gate5A model, touchdown speed magnitude is exactly the sealed canonical launch `speed_mps`. The stop evaluator uses that value directly; it does not finite-difference samples or introduce a square-root operation.
- The closed exact-binary64 condition is `speed_mps <= 3.0`.
- On pass, the simulation model records `post_capture_speed_mps = 0.0` and performs the explicitly selected instantaneous nominal-mean state reset.
- It models no rebound, settling time, load, actuator, physical recenter path, or hardware feasibility.

### Energy and time model

- Primitive relative energy evaluates `ratio = speed_mps / 3.0` and then `ratio * ratio` in that exact binary64 operation order. It is dimensionless and must never be labeled Joules.
- The four speed levels yield `0.25`, `4/9`, `25/36`, and `1.0` in exact-real semantics; serialized binary64 values are resealed evidence.
- Flight time remains the raw ballistic resource: fresh captured Hopper-specific replay's second/final `BallisticSampleV2.time_s`, which must exact-word-match `flight_time` independently rederived as `vertical_speed = speed_mps * sin(elevation_rad)`, `vertical_time_scale = vertical_speed / gravity_mps2`, then `2.0 * vertical_time_scale`, is copied into inherited `RoutePrimitiveV2.duration_s`. Operational cadence is that same word, but no candidate/primitive/profile/parameter/resource/outcome `dt_s` field or hidden normalizer exists.
- `risk_weight` must be exact canonical `0.0`; any nonzero value returns `hopper_risk_objective_unsupported` before action generation. Per-hop mass remains a diagnostic/safety threshold, not an unapproved additive risk-cost model.
- Search and final `CostBreakdownV2` apply request objective weights exactly once. Energy may not include time, and time may not include energy.
- Route energy/time are deterministic sums over all primitives; candidate-set membership cannot change a primitive's resource values.

### Provider authority wrapper

Add this exact frozen/slotted authority wrapper in the listed field order:

```text
HopperProviderAuthorityV2
- hopper_profile: HopperProfileV2
- parameter_set_id: str | None
- authority_schema_version: str
```

- The exact authority schema is `hopper-provider-authority/v1`.
- Wrapper `hopper_profile` must be exact `HopperProfileV2`; `parameter_set_id` must be exact nonempty built-in `str` or exact `None`; `authority_schema_version` must exact-match the schema above. The wrapper never resolves or validates support in its constructor.

The trusted registry value is this exact frozen/slotted record in the listed field order:

```text
HopperParameterSetRecordV2
- parameter_set_id: str
- base_profile_id: str
- body_envelope_radius_m: float
- launch_reference_height_m: float
- arc_clearance_margin_m: float
- landing_footprint_radius_m: float
- stop_condition: str
- energy_model: str
- stop_evaluator: callable
- energy_evaluator: callable
- evidence_class: str
- simulation_proxy: bool
- formal_evidence_eligible: bool
- schema_version: str
```

The exact record schema is `hopper-parameter-set-record/v1`. Strings are exact built-in nonempty strings; the two model strings satisfy the existing versioned-proxy-ID contract. The four numeric fields are exact positive built-in binary64 values with the words listed above. Both evaluators are callable and must be the exact trusted registry objects. The fixture record requires exact `evidence_class="test_fixture"`, `simulation_proxy is True`, and `formal_evidence_eligible is False`; subclasses/coercions fail.

Two different tokens are frozen:

- In-memory authority token, in record field order: exact strings, four canonical float words, evaluator object identities, exact booleans, and schema. It is compared only with `is`/exact equality inside the trusted process.
- Deterministic lineage token: the same sequence with evaluators omitted; the versioned `stop_condition`/`energy_model` IDs carry serialized implementation lineage. Object addresses, `repr`, and process-specific IDs never enter a digest or artifact.

The registry container is an exact tuple of exact records, strictly sorted by `parameter_set_id` with no duplicate. Lookup is an exact deterministic scan; each record's `parameter_set_id` must equal its registry key/identity. The provider exposes the wrapper under the exact read-only Hopper-specific attribute name `hopper_authority`. Registry lookup of `parameter_set_id=None` or an unknown exact ID returns the canonical token `("hopper-parameter-set-absent/v1", requested_parameter_set_id)` rather than an authority error. API/provider seal the registry-container token and this absent token before and after dispatch; stable absence reaches `hopper_parameter_set_unsupported`, while absence changing to/from a record during a call is `hopper_authority_contract_mismatch`.
- `parameter_set_id=None` is a representable preflight-only wrapper state. With a structurally incomplete profile, `plan()` returns `UNSUPPORTED_CAPABILITY / hopper_proxy_profile_incomplete`; with a structurally complete profile, it returns `UNSUPPORTED_CAPABILITY / hopper_parameter_set_unsupported`. Neither path invokes Hopper ballistics, Hopper terrain oracle, action generation, or search. Existing API-wide start/goal terrain queries retain their current cross-platform precedence and may already have occurred before provider dispatch.
- A structurally complete profile requires a registered parameter-set ID before any operational evaluator runs.
- The authorized 11B3 implementation of this approved section must provide exactly one immutable Gate5B registry entry, keyed by `hopper_gate5b_algorithm_fixture/v1`, binding the exact record above.
- No alias, fallback, dynamic import, string-only whitelist, or partial numeric match is allowed.
- Structural completeness from `audit_hopper_profile_v2()` means only six fields are non-null; it never means supported or formally eligible.
- The generic provider protocol remains unchanged for wheel/legged. API Hopper dispatch reads exact `provider.hopper_authority` and captures one token containing the exact wrapper/profile token, either the complete in-memory registry-record token or canonical absent token, and authority schema. A Python bound method is sealed as the stable pair `(__self__, __func__)` compared by object identity; the ephemeral bound-method wrapper object itself is never used as an identity token. API compares the full token before dispatch, after either success or failure, before/after route L2, and immediately before return.
- The complete profile, parameter-set record, implementations, primitive provenance, route digest, and outcome are sealed before and after every untrusted helper and at API return. `KeyboardInterrupt`, `SystemExit`, and `MemoryError` propagate; an ordinary helper exception is mapped only after the mandatory post-call reseal.

### Capability failure order

1. Authority/profile/snapshot drift detected at a checkpoint overrides the current semantic result.
2. Existing API profile resolution and generic endpoint authority remain unchanged for all platforms.
3. Provider mapping and base-profile identity.
4. Exact authority-wrapper type and deep profile seal.
5. Any missing field: `hopper_proxy_profile_incomplete`.
6. Missing/unregistered parameter set or any exact fixture-number/base-profile-ID mismatch: `hopper_parameter_set_unsupported`.
7. Stop ID mismatch: `hopper_stop_condition_unsupported`.
8. Energy ID mismatch: `hopper_energy_model_unsupported`.
9. Nonzero risk objective: `hopper_risk_objective_unsupported`; required accelerator: `hopper_accelerator_required_unsupported`.
10. Only then the provider-internal deadline checkpoint, goal-heading/hold preflight, resource setup, launch, arc, landing, replay, and stop semantics.

The existing API has its own pre-dispatch deadline checkpoint after provider resolution/Hopper authority sealing and before calling `plan()`. If already expired there, API timeout wins and provider capability preflight is not called. Once dispatch begins, the provider order above applies internally. After `plan()` returns, API performs the mandatory post-call seal and then its `provider_completion` deadline checkpoint; expiry there overrides the returned capability/semantic outcome unless a higher-ranked authority mismatch is detected. Only a nonexpired path proceeds to exact failure audit or success route L2, then the `provider_postcondition` deadline checkpoint and final seal. Section 3 freezes the complete sequence.

The registry record itself is exact-type/deep-sealed before semantic comparisons. A mutated registry or evaluator binding is `hopper_authority_contract_mismatch`. For a valid record, comparisons are deliberately staged so dedicated reasons remain reachable: base profile ID plus the four numeric fields first, `stop_condition` second, and `energy_model` third.

### Default and formal isolation

- The authorized 11B7 runner implementation must create repo-default `configs/xunce_path_v2_gate5_hopper_v1.json` with all six fields explicitly `null` and `parameter_set_id=null`. That file does not exist before 11B7.
- Any later modification of a default field to non-null is a config-contract failure, not permission to execute the fixture.
- Algorithm tests may construct the fixture only through a dedicated test factory plus the exact authority wrapper. A bare complete `HopperProfileV2` cannot run.
- The authorized 11B7 Gate5 runner implementation must never feed fixture rows into the existing formal benchmark aggregate and must never read formal labels, optima, or schedules while the profile-freeze blocker is first.
- Only after separate controller authorization, a future `D:/xunce/out/path_v2/g5` snapshot must have `status=blocked`, `formal_metrics_status=not_evaluated`, `formal_evidence_eligible=false`, `formal_row_count=0`, and primary blocker `freeze_hopper_simulation_proxy_profile_parameters` repeated consistently in summary, routing, review, manifest, and report. That output root does not exist at this design stage.
- CLI exit code zero for `blocked` is never treated as success; controllers must read and cross-check the exact status/eligibility fields.
- The authorized 11B7 stage-registry/runner implementation must hard-code and cross-check `execution_class=blocked_profile_freeze` and `formal_evidence_eligible=false`; config self-report is not trusted.
- Future formal Hopper capability requires a new capability/schema revision and a new immutable formal registry entry. The fixture entry cannot be promoted by flipping a boolean.

### RED boundary matrix

- Every missing field returns the stable incomplete reason with zero Hopper ballistics/oracle/action/search calls; API generic endpoint-query calls retain their established behavior.
- Structurally complete but missing authority, unknown parameter set, one-ULP numeric mismatch, wrong base profile, stop ID mismatch, and energy ID mismatch follow the exact precedence above.
- Post-construction mutation of any nested profile/registry/implementation binding fails reseal.
- All four energy values, touchdown `3.0` equality, `nextafter(3,+inf)` rejection, raw flight-time accounting, route sums, nonzero-risk rejection, and candidate-set independence.
- Default config exact nulls; any non-null fails config audit.
- Fixture tests may pass while every stage artifact remains blocked/formal-ineligible with zero formal rows.
- Existing formal aggregate is never called with fixture data; attempts to load fixture rows as formal evidence fail closed.
- API output, primitive provenance, route digest, summary, routing, review, manifest, and report all agree on the parameter-set and evidence class.


## Part III — Approved Section 3 route/search/reseal contract

### Approval scope

- Sections 1, 2, and 3 were explicitly approved on 2026-07-20; readiness/implementation was separately approved on 2026-07-21.
- A2 was explicitly approved on 2026-07-21 and narrowly amends the Hopper action/helper/replay/resource binding frozen in this Part III.
- Section 3 approval by itself freezes this design contract only; implementation authority comes only from the separate 2026-07-21 readiness/implementation approval and this tracked plan.
- It freezes the typed Hopper state/primitive, deterministic nominal-mean multi-hop search, independent complete-route L2 replay, probability diagnostic, exact success/failure envelope, reason/category/stage/phase mapping, API/provider reseal, the `min(requested,100_001)` route-state clamp, Hopper-only 512 MiB deterministic accounted-memory ceiling, complete outcome/transient accounting, the unchanged generic explicit-cap surfaces plus A2's private index-aware Hopper ballistic surface, and the separate resource-authority record/tokens. Its design approval plus the later explicit A2 approval covers those contracts only; implementation authority comes from the separate 2026-07-21 readiness/implementation approval, the 2026-07-21 A2 approval, and this tracked plan.
- v1/default/PPO/executor/canary boundaries remain unchanged. Gate5 fixture success can never become formal Gate5 evidence.

### Minimal typed nominal search state

```text
HopperSearchStateV2
- nominal_state: PoseStateV2
- support_height_m: float
- schema_version: str
```

- The exact schema value is `hopper-nominal-mean-search-state/v1`. The object is exact frozen/slotted; subclasses, non-finite values, mutated nested fields, and noncanonical heading fail closed. Every signed zero is canonicalized to positive zero before token/key construction.
- `support_height_m` is the exact common same-height support `H0`. Reference z is derived from sealed `H0 + launch_reference_height_m`; it is not added to public `PoseStateV2`.
- Parameter-set identity is request/provider-global authority and is deliberately not duplicated in every state.
- Define `HOPPER_STATE_KEY_TAG_V1 = 0x484F505045525631`. The exact queue key is `(HOPPER_STATE_KEY_TAG_V1, bits(x), bits(y), bits(canonical_heading), bits(H0))`. Every member is an exact int accepted by `StableSearchQueueV2`; no snap, tolerance, decimal rendering, digest, or randomized hash participates. Schema/parameter authority is resealed separately.

### Minimal typed primitive

`HopperJumpPrimitiveV2` is an exact frozen/slotted subclass of `RoutePrimitiveV2`. Its additional fields are only:

```text
start_hopper_state: HopperSearchStateV2
end_hopper_state: HopperSearchStateV2
speed_index: int
elevation_index: int
azimuth_index: int
parameter_set_id: str
selected_landing_mass: float
primitive_schema_version: str
```

- The exact primitive schema is `hopper-jump-primitive/v1`. Inherited `kind` is `BALLISTIC_JUMP`; inherited start/end poses exactly equal the two typed nominal poses; validation level is L2.
- Action key, raw speed/elevation/canonical azimuth, and canonical A2 direction components are derived from the three indices plus the sealed profile. `H0` comes from the typed states; stop/energy IDs and implementations come from the immutable parameter-set record. No raw duplicate action values or `dt_s` are carried.
- End x/y are the fresh Hopper-specific A2 theoretical mean endpoint; heading and `H0` are invariant. The explicit stop model, not a snap, owns the instantaneous nominal-mean reset.
- Inherited `duration_s`, `distance_m`, and `energy_cost` are canonical replay values; `observation_contribution` is exact `0.0` because inflight observation remains disabled.
- `selected_landing_mass` is the unconditioned frozen simulation-proxy mass of the exact shortest prefix and must be at least the canonical binary64 threshold `0.99`. The field is not called certified physical probability.
- Full ballistic samples and up to one million landing candidates are not retained in every primitive/search node. They are transiently rebuilt under public caps by the primitive oracle and complete-route L2. No primitive instance method or cached evidence is an authority source.
- The primitive token/digest includes its schema, all inherited fields, both typed states, three indices, parameter-set ID, and selected mass. It excludes derived duplicate values and implementation object addresses.

### Fixed action and search order

- Every expanded state enumerates exactly 192 actions in nested order: speed index `0..3`, elevation index `0..2`, azimuth index `0..15`.
- Derived action key is `"{speed_index:02d}:{elevation_index:02d}:{azimuth_index:02d}"`. One frozen Gate5B action helper derives world azimuth in exact order `2.0*pi*k/16.0` and the A2 direction: exact cardinal table at `k=0/4/8/12`, raw `sin/cos` for the other twelve, and canonical positive zero. Provider, primitive oracle, provider route L2, and API route L2 independently rebuild and exact-word compare both azimuth and direction; no endpoint repair or tolerance is allowed. Equal-range 30-degree/60-degree actions remain distinct because duration/apex/clearance differ.
- Search is deterministic Dijkstra: `StableSearchQueueV2`, `h=0`, and goal testing only after a valid node is dequeued.
- Candidate IDs are unique monotonic strings `hopper-node-{serial:020d}`. A state key is never reused as candidate ID because retired queue IDs cannot be reopened.
- Every queue entry has a nonempty authoritative primitive key. The root sentinel is exactly `hopper-root/v1`; each non-root entry uses the derived three-index action key. The root candidate is `hopper-node-00000000000000000000`, later serials increase by one, and both keys participate unchanged in queue tie order.
- `best_by_state[state_key] = (exact_binary64_total_cost, candidate_id)`. Dequeue performs stale-skip then closed-skip. A child replaces an open state only on strictly lower exact cost; an exact tie keeps the first action-order candidate. A lower-cost path to an already closed state is `hopper_search_cost_contract_mismatch`, never an implicit reopen.
- Parent/action lineage is immutable. `determinism_seed` is sealed but does not alter actions, ties, oracle order, or evidence.
- Goal heading incompatibility returns `hopper_goal_heading_unreachable` before search. Position has exact zero tolerance and is never snapped. Exact start==goal cannot produce the nonempty `TypedRouteV2` required by the public contract and returns `hopper_hold_oracle_unavailable` rather than a fabricated zero-hop success.
- A local semantic jump rejection records ranked telemetry and continues to the next fixed action. Deadline, authority drift, numeric failure, and any resource cap are global aborts. Search exhaustion returns `hopper_no_complete_route`; no partial route is returned.

### Cost authority

- Gate5B v1 requires `risk_weight` to be exact canonical `0.0`; nonzero risk is `hopper_risk_objective_unsupported`. `risk_cost` is exact `0.0` and landing probability is not silently converted into an additive cost.
- Per primitive resources are canonical A2 index-derived horizontal range for distance, registry-bound relative energy using Section 2's exact `ratio = speed_mps / 3.0` then `ratio * ratio` binary64 order, and captured Hopper-specific replay's second/final `BallisticSampleV2.time_s` for time. That sample time must exact-word-match the independently rederived `flight_time` in the frozen operation order and is copied into inherited `RoutePrimitiveV2.duration_s`. No time normalization or serialized cadence exists.
- Starting from exact zero component totals, each edge executes these binary64 operations in this exact order: multiply weight by primitive resource, add it to the parent component, then compute `total_cost = sum((distance_cost, 0.0, energy_cost, time_cost))` in tuple order. Every intermediate must be finite, nonnegative, and canonical-zero.
- Provider dominance uses that exact total. Independent route L2 reconstructs the same published operation order from sealed request weights and fresh primitive resources; it does not trust provider component totals or a primitive method. Every component and total must match by exact binary64 word/`.hex()`, not `isclose`.
- Candidate-set membership cannot alter a primitive resource. Route and `CostBreakdownV2` totals apply each request weight exactly once.

### Route probability diagnostic

- Success remains only a per-hop `selected_landing_mass >= 0.99` contract. Whole-route `>=0.99` is neither a search condition nor a capability claim.
- `HopperRouteProbabilityDiagnosticV2(per_hop_mass, union_bound_lower_mass, schema_version)` is exact frozen/slotted with schema `hopper-route-probability-diagnostic/v1`.
- Convert each canonical mass with `as_integer_ratio()`, compute exact `max(0, 1 - sum_i(1-p_i))`, convert to binary64, and if the rounded float exceeds the exact rational take one `nextafter(..., -inf)`. This prevents an overstated lower bound.
- No Hopper route subclass is introduced because API success requires exact `TypedRouteV2`. `HopperRouteValidationResultV2` is the internal validator carrier for the diagnostic; `PlanningSuccessV2` keeps its existing shape and does not embed it. Public diagnostic/report code calls the same pure rebuild helper on exact primitive masses, and fixture artifacts may serialize that separate result. It is labeled simulation-proxy only.

### Exact success envelope and telemetry

- Hopper success remains exact `PlanningSuccessV2`; no public outcome subclass is introduced. `ObservationProjectionV2.source` is exactly `path-planner-v2-hopper-inflight-observation-disabled/v1`. For `H` primitives, `sample_states` is exactly `(primitive_0.start_state, primitive_0.end_state, ..., primitive_(H-1).end_state)`, length `H+1`, with the shared connected endpoints represented once and no inflight samples. Both observation metrics are exact canonical `0.0`; the source explicitly makes no observation-gain claim.
- Validation evidence is exact `validator_id="path-planner-v2-hopper-route-l2/v1"`, level L2, `passed=True`, and `checks=("hopper_route_l2_valid",)`. Cost breakdown is the exact L2-rebuilt five-component value. Cache evidence is the disabled value frozen below. The separately rebuilt probability diagnostic remains in the exact internal validation carrier and is not smuggled into an existing public field.
- Success telemetry has exact `rejected_l0=0`, `rejected_l1=0`, `timed_out=False`, `accelerator_used=False`, `ackermann_feasible_claimed=False`, and `termination_reason="hopper_route_l2_valid"`. `elapsed_s` is an exact finite nonnegative canonical binary64 sampled from the sealed deadline at outcome construction and then only word-sealed; API cannot independently reconstruct wall-clock history.
- `expanded_states` increments exactly once, immediately before action index zero of a fresh, non-stale, non-closed, non-goal dequeued state, after authority/deadline/expansion-budget checks. `generated_primitives` increments immediately before invoking the jump oracle for each successfully derived fixed action. `rejected_l2` increments exactly once for each well-formed local semantic jump rejection; dominance rejection, route-state admission rejection, and authority/numeric/deadline/resource aborts do not increment it. Therefore `rejected_l2 <= generated_primitives`; for success or ordinary search exhaustion `generated_primitives == 192*expanded_states`, while a global abort inside an expansion satisfies `192*(expanded_states-1) <= generated_primitives <= 192*expanded_states` with the lower bound clamped to zero.
- Structural telemetry audit requires exact `SearchTelemetryV2` and `type(value) is int` for all five count fields; bool, int subclass, and coercion are rejected. All are nonnegative, `expanded_states <= sealed_request.resource_budget.max_expanded_states`, `rejected_l2 <= generated_primitives`, and every outcome satisfies `max(0,192*(expanded_states-1)) <= generated_primitives <= 192*expanded_states`. Because hold is unavailable and success routes are nonempty, success has `expanded_states>=1` and exact `generated_primitives=192*expanded_states`. Ordinary `hopper_no_complete_route` also has `expanded_states>=1` and the same equality. Every provider-returned failure, regardless of outward detecting stage, is structurally checked only against the universal clamped prefix bound and cannot use stage to claim a post-search origin. Inside the trusted provider, reaching complete-route L2 after goal dequeue separately requires `expanded_states>=1` and exact `generated_primitives=192*expanded_states`; this internal invariant is sealed/tested before converting the carrier to an outward failure but is not independently provable from `PlanningFailureV2`. `elapsed_s` must be exact built-in float, finite, nonnegative, and canonical-zero; every telemetry boolean and termination string must be exact built-in type.
- `HopperRouteValidationResultV2` is an exact frozen/slotted internal carrier with schema `hopper-route-validation-result/v1` and fields in this exact order: `passed`, `reason_code`, `category`, `stage`, `details`, `route_state_count`, `replay_work_counts`, `l2_peak_accounted_bytes`, `cost_breakdown`, `probability_diagnostic`, `route_digest`, `evidence`, `schema_version`. `passed` is exact bool; `reason_code`/`stage` are exact nonempty built-in strings; `category` is exact `FailureCategoryV2` or exact `None`; non-pass `stage` is the frozen underlying detecting stage (arc/landing/stop/resource/etc.), not a synthetic overall `route_validation` stage. `details` follows the exact failure template; `route_state_count` and `l2_peak_accounted_bytes` are exact nonnegative built-in ints and reject bool; the latter is the greatest admitted `R(H)+active_transient` total reached by this L2 call (an over-limit rejected attempt remains only in failure details). `replay_work_counts` is an exact six-element tuple of `(counter_id, exact_nonnegative_int)` pairs in `HOPPER_REPLAY_COUNTER_IDS_V1` order; every count rejects bool and is at most `100_000`. Each value is the elementwise maximum admitted value for that counter across every completed hop and the active processed hop prefix; counters reset to zero per hop, never sum across hops, and an over-limit attempted active-hop value appears only in failure details. Cost/diagnostic/digest values are exact typed values on pass and exact `None` on every non-pass; `evidence` is exact L2 `ValidationEvidenceV2` with validator ID `path-planner-v2-hopper-route-l2/v1` and `checks=(reason_code,)`.
- A pass carrier is exactly `passed=True`, `reason_code="hopper_route_l2_valid"`, `category=None`, `stage="route_validation"`, empty details, all six elementwise-max replay counters present, non-None cost/diagnostic/digest, and passed evidence. A non-pass carrier is exactly `passed=False`, a frozen L2 non-pass reason/category/underlying-stage/details row, failed evidence, prefix-complete elementwise-max counters, and exact `None` for cost/diagnostic/digest regardless of how far validation progressed. When provider final L2 converts it to `PlanningFailureV2`, the outward evidence stage remains this underlying stage. The carrier token follows field order and is resealed before use. A subclass, missing/extra counter, invalid optional presence, malformed details, or throwing/non-carrier result is not a well-formed non-pass.

### Search and replay resource accounting

- Gate5A's `MAX_BALLISTIC_SAMPLES_V2=100_000` and `MAX_LANDING_ZONE_CANDIDATES_V2=1_000_000` remain independent pre-allocation caps. Section 1 adds independent `MAX_REPLAY_STEPS=100_000` interval/cell counters.
- `request.resource_budget.max_route_states` counts full replay states under the generic audit identity `1 + sum(S_i-1)`, not merely hop count. Operational replay first requires exact `S_i=2` for every hop, so each child delta is exact `1` and an `H`-hop route has exactly `H+1` replay states; `S_i != 2` is `hopper_numeric_contract_mismatch`, not an accepted cadence. The single effective bound is `effective_max_route_states = min(requested_max_route_states, MAX_REPLAY_STEPS+1) = min(requested_max_route_states, 100_001)`. It is used unchanged by child admission and provider/API L2 with the subtraction-safe check after proving `parent_route_states <= effective`. A nonempty one-hop route needs two route states, so effective `0` or `1` fails at resource setup. Observation's endpoint-only `H+1` `sample_states` has the same number but is a different carrier and authority; it cannot prove ballistic replay or safety.
- Memory is versioned deterministic admission accounting, not a claim about CPython object size, allocator behavior, process RSS, or physical peak bytes. `sys.getsizeof`, `tracemalloc`, RSS sampling, and post-allocation measurement are not authority. The exact accounting ID is `hopper_deterministic_admission_bytes/v1` and `HOPPER_HARD_ACCOUNTED_MEMORY_BYTES_V2 = 536_870_912` (512 MiB).
- The existing generic sentinel is preserved: requested `max_memory_bytes == 0` means that the caller supplies no smaller byte limit, not a zero-byte budget. Hopper still applies its hard ceiling. The exact effective limit is `HARD` when requested is zero and `min(requested, HARD)` otherwise. It is enforced even when the caller requests more than 512 MiB.
- Persistent search accounting is `P(N) = 4_096 + 1_024*N`, where `N` is every admitted record including the root. The 1,024-byte bundle is frozen as node/state/cost/parent-and-action lineage `384`, active-or-retired queue entry `256`, anchor index `128`, best slot `128`, future closed slot `64`, and map/container reserve `64`. `auxiliary_heuristics` is exact empty and provider queue discard is disabled.
- Every admitted root/open/popped/retired/closed/stale record and every improved history record is charged forever within the search scope; there is no refund. Oracle rejection, a non-improving child, or a child rejected before admission is not charged. Root and child admission atomically check `P(N+1)` before any node, queue, anchor, best, or closed write. Persistent lineage stores only a parent candidate ID, packed action indices, selected mass, and scalar edge resources; it never owns samples, landing candidates, or a duplicate full primitive.
- `HopperResourceAuthorityV2` is an exact frozen/slotted record with schema `hopper-resource-authority/v1` and fields in this exact order: `ballistic_helper`, `ballistic_helper_id`, `landing_helper`, `landing_helper_id`, `memory_accounting_id`, `max_ballistic_samples`, `max_landing_candidates`, `max_replay_steps`, `max_exact_integer_bits`, `max_exact_live_integer_slots`, `hard_accounted_memory_bytes`, `search_base_bytes`, `search_record_bytes`, `ballistic_build_base_bytes`, `ballistic_build_per_sample_bytes`, `ballistic_retained_per_sample_bytes`, `landing_build_base_bytes`, `landing_build_per_candidate_bytes`, `exact_base_bytes`, `exact_integer_slot_overhead_bytes`, `exact_distinct_cell_bytes`, `route_build_base_bytes`, `route_primitive_bytes`, `schema_version`. Callable fields are the exact trusted objects; helper IDs are respectively `sample_hopper_ballistic_arc_capped/v1` and `landing_zone_cells_capped/v1`; the accounting ID is `hopper_deterministic_admission_bytes/v1`; every numeric field is an exact positive built-in int with the values/formulas in this section. This explicit ballistic callable/ID replacement changes both the in-memory token and deterministic lineage token without changing record field order, schema, caps, or byte coefficients. The provider exposes the canonical singleton under exact read-only attribute `hopper_resource_authority`; it must be the same object/token used by API and both L2 callers. Section 2's `hopper_authority` wrapper and field order remain unchanged.
- The in-memory resource token follows that exact record field order: callable fields contribute object identity, strings/schema contribute exact built-in values, and numeric fields contribute exact ints. The deterministic lineage token is exactly `("hopper-resource-authority-lineage/v1", ballistic_helper_id, landing_helper_id, memory_accounting_id, max_ballistic_samples, max_landing_candidates, max_replay_steps, max_exact_integer_bits, max_exact_live_integer_slots, hard_accounted_memory_bytes, search_base_bytes, search_record_bytes, ballistic_build_base_bytes, ballistic_build_per_sample_bytes, ballistic_retained_per_sample_bytes, landing_build_base_bytes, landing_build_per_candidate_bytes, exact_base_bytes, exact_integer_slot_overhead_bytes, exact_distinct_cell_bytes, route_build_base_bytes, route_primitive_bytes, schema_version)`; callables/addresses are omitted because their versioned IDs carry serialized lineage.
- API's exact in-memory composite token order is `("hopper-api-composite-authority/v1", section2_hopper_authority_token, resource_authority_in_memory_token, bound_plan_self_func_pair, jump_oracle_callable_identity, route_l2_callable_identity)`. `section2_hopper_authority_token` is reused byte-for-byte with its existing wrapper/registry/absent-record order; the resource token is a separate following element and never alters either Section 2 record. API acquires both only from exact read-only provider attributes `hopper_authority` then `hopper_resource_authority`; module-global fallback, aliasing, or dynamic lookup is forbidden.
- Gate5B retains the generic explicit-cap module helpers `sample_ballistic_arc_capped_v2(..., max_sample_count=...)` and `landing_zone_cells_capped_v2(..., max_candidate_count=...)`; their existing public wrappers, behavior, bytes, and generic helper ID semantics stay unchanged. A2 adds the private index-aware `_sample_hopper_ballistic_arc_capped_v2(..., azimuth_index, ..., max_sample_count=...)` above. All explicit caps require exact positive built-in integers, use only the explicit argument for every pre-allocation/work check, and never reread a mutable module cap global. Hopper binds only the exact private specialized ballistic object plus the exact landing helper and passes exactly `100_000` and `1_000_000` from its sealed resource-authority record.
- `_call_captured_ballistic_helper_v2` has the exact index-aware call shape `(authority, start, speed_mps, elevation_rad, azimuth_index, g_mps2)` with no `azimuth_rad` or `dt_s`. It entry-reseals the authority, reads only the captured `authority.ballistic_helper` and sealed cap, calls that captured object with the exact index and keyword-only `max_sample_count=100_000`, and performs mandatory reseal after an ordinary exception and immediately before a normal return. Critical process exceptions retain the existing propagation rule. Mandatory reseal also read-only checks that `ballistics._sample_hopper_ballistic_arc_capped_v2` is still the captured trusted object; this detects specialized module-binding drift but never dynamically dispatches through the module. Rebinding generic `sample_ballistic_arc_capped_v2` cannot change Hopper dispatch or its seal.
- The exact in-memory resource-authority token contains the specialized ballistic and capped landing helper object identities plus the exact values of ballistic cap, landing cap, replay-work cap, integer-bit cap, live-slot cap, memory hard ceiling, and every byte coefficient. Its deterministic token uses the versioned helper IDs and the same integer values without object addresses. API, provider, primitive oracle, and route L2 seal this token before and after every helper call and invoke the captured local callable rather than rereading a module attribute for dispatch. A changed specialized helper/landing helper/cap/coefficient is `hopper_authority_contract_mismatch` before materialization; tests that mutate legacy globals or the generic ballistic helper cannot change the explicit-cap Hopper path. No temporary global rewrite is permitted.
- These helpers are reserved at their explicit hard cap before invocation because their internal allocation cannot otherwise be stopped by the request budget. Ballistic build remains `HOPPER_BALLISTIC_BUILD_MAX_BYTES_V2 = 4_096 + 96*100_000 = 9_604_096`; A2's operational two-sample result does not lower this reservation or the `100_000` cap. A returned sample retained for the following arc phase remains `HOPPER_RETURNED_BALLISTIC_SAMPLE_BYTES_V2 = 48` per actual sample, hence exact operational retained bytes are `48*2 = 96`. Landing build remains `HOPPER_LANDING_BUILD_MAX_BYTES_V2 = 65_536 + 384*1_000_000 = 384_065_536`. The last formula deliberately uses the public million-candidate cap rather than depending on the current concentric-square implementation's smaller `998_001` maximum retained square.
- Exact algebraic work uses a bounded arena, not runtime object inspection. The same single canonical Section 1 constant is `HOPPER_EXACT_MAX_INTEGER_BITS_V2 = 262_144`; `HOPPER_EXACT_LIVE_INTEGER_SLOTS_V2 = 128`, and one reserved slot is `16 + ceil(HOPPER_EXACT_MAX_INTEGER_BITS_V2/8) = 32_784` bytes. With exact base `4_096` and `HOPPER_EXACT_DISTINCT_CELL_BYTES_V2 = 320` for at most `100_000` distinct cells, `HOPPER_EXACT_ORACLE_MAX_BYTES_V2 = 4_096 + 128*32_784 + 100_000*320 = 36_200_448`. Candidate ranges and interval-cell pairs are streamed; only the bounded global distinct-cell ledger persists. Every exact operation reserves a conservative result bit bound and its prospective live slot before any integer allocation. Attempting to acquire slot 129 or to exceed the canonical bit ceiling returns `hopper_numeric_contract_mismatch`, performs no integer operation, and is never reclassified as `hopper_memory_budget_exceeded`. If an implementation cannot statically account for every live slot and also enforce that runtime pre-allocation admission, it cannot enter the provider slice and must propose a new reviewed accounting revision.
- Transient scopes do not overlap implicitly. While search records remain live, an action checks in order: `P(N)+9_604_096` before ballistic build; after build and the exact `S=2` postcondition, `P(N)+96+36_200_448` before arc work; then releases samples/arc state while retaining only sealed scalar resources; then checks `P(N)+384_065_536` before landing build. The generic audit formula remains `P(N)+48*S+36_200_448`, but no production path accepts `S != 2`. Landing work and its returned prefix are disposed before child admission. A helper hard/work cap is checked before its memory reservation, and both are checked before materialization.
- Complete outcome materialization is `R(H) = 4_096 + 512*H` for `H` primitives. The 4,096-byte base explicitly covers `TypedRouteV2`, `PlanningSuccessV2`, `ObservationProjectionV2`, `CostBreakdownV2`, `ValidationEvidenceV2`, `SearchTelemetryV2`, `CacheEvidenceV2`, the internal L2 result/probability diagnostic, digest/outcome-token containers, and the first observation-state slot. Each 512-byte hop bundle covers the exact primitive, one endpoint observation-state slot, route/digest tuple slots, and construction reserve. No ballistic sample, landing candidate, exact-arena object, or search record is included or retained. Before construction the provider checks `P(N)+R(H)` while search structures still exist, then explicitly ends the search scope and proves no alias retains queue/node/best/closed state. Provider route L2 and API route L2 retain this complete outcome reservation plus only one hop's transient scope, so each independently checks `R(H) + max(9_604_096, 96+36_200_448, 384_065_536)` for the operational active hop rather than accumulating evidence for all hops; the worst-case ballistic build `9_604_096`, exact oracle `36_200_448`, six replay counters, 262,144-bit ceiling, and 128-slot arena remain unchanged.
- `HopperSearchMemoryLedgerV2` is an exact frozen/slotted provider-internal carrier with schema `hopper-search-memory-ledger/v1` and exact fields `accounting_id`, `requested_max_memory_bytes`, `effective_max_memory_bytes`, `admitted_record_count`, `persistent_peak_bytes`, `transient_reservation_peak_bytes`, `combined_accounted_peak_bytes`, `route_materialization_peak_bytes`, `schema_version`. `persistent_peak_bytes` is the greatest `P(N)` reached; `transient_reservation_peak_bytes` is the greatest phase-local reservation alone; `combined_accounted_peak_bytes` is the greatest simultaneous persistent-plus-transient accounted total over all search, materialization, and L2 scopes; `route_materialization_peak_bytes` is exact `P(N)+R(H)` or zero if no route was materialized. Each update creates and reseals a new exact ledger value. The provider audits it before constructing success or a memory failure; it is not added to `PlanningSuccessV2`.
- Memory-failure evidence is phase arithmetic, not a peak summary. For `root_admission`/`child_admission`, `persistent_accounted_bytes` is current `P(N)`, `transient_reserved_bytes=0`, and attempted bytes is prospective `P(N+1)`. For `ballistic_build`, `arc_oracle`, and `landing_build`, persistent is current `P(N)`, transient is respectively `9_604_096`, `48*S+36_200_448`, or `384_065_536`, and attempted is their sum. For `route_materialization`, persistent is `P(N)`, transient is `R(H)`, and attempted is the sum. For `route_l2`, persistent is `R(H)`, transient is the active-hop ballistic/arc/landing reservation, and attempted is the sum. Every reported component and sum is an exact nonnegative built-in int.
- Resource-setup precedence is expansion minimum, route-state minimum, then root admission memory. Within an action, specialized ballistic or landing helper hard/work cap precedes memory reservation; admission memory precedes any persistent write. Expanded-state count, route-state count, persistent admitted records, transient peak reservations, landing candidates, all six replay-work counters, exact-integer ceiling, live-slot ceiling, and deadline remain independent. Passing one never waives another.
- Cache is explicitly disabled in v1 with exact `cache_namespace="path-planner-v2-hopper-cache-disabled/v1"`, `cache_key="hopper-cache-disabled/v1"`, and `hit=False`. Existing generic validation cache cannot carry Hopper landing/mass/parameter provenance and never replaces fresh replay.

### Independent complete-route L2

Create a dedicated Hopper route-validation module and only a thin re-export from shared `validation.py`. `validate_hopper_route_l2()` independently performs, in order:

1. Seal exact request/deadline, Hopper authority wrapper, profile/parameter-set/implementation identities, the complete resource-authority/helper/cap token, snapshot identity/hash/query authority, and grid geometry.
2. Require exact `TypedRouteV2`, Hopper platform, complete/nonempty route, exact `HopperJumpPrimitiveV2` elements, public counts, and a fresh complete route digest.
3. Verify first state, every pairwise typed/base-pose connection, common parameter set, invariant heading/H0, and final exact goal. Route structure, start, connectivity, and goal have distinct reasons.
4. For each primitive in index order, independently derive raw speed/elevation, canonical azimuth, A2 canonical direction, flight time, and two-sample cadence from the sealed profile and three exact indices; invoke only the captured specialized ballistic helper with the sealed fixed cap; require exact two-sample/final-time/direction postconditions; run the exact full-parabola continuous arc oracle over the single `[0,1]` operational interval; rebuild the landing prefix before any landing-terrain query; validate selected/mean footprints and stop; and compare mass/resources by exact binary64 token. Neither provider fields nor observation endpoints authorize this replay.
5. Rebuild the generic replay-state audit identity and require its operational specialization to exact `H+1` against the single effective `min(requested,100_001)` bound, all six per-hop work counters, complete outcome-materialization bytes, every active-hop transient reservation, cost components/total, per-hop diagnostic, primitive tokens, and route hash. Route L2 does not claim that a route can reconstruct stale/retired search history or the provider's persistent admitted-record peak.
6. Reseal route, request, deadline, authority/profile/registry/implementations, snapshot, helper results, and validation result immediately before return.

Calling an untrusted instance method requires a before/after object token; normal validation does not need one. Final provider validation follows `pre_digest -> independent L2 -> post_digest -> final digest/result-token reseal`. API Hopper success repeats independent L2, seals the provider outcome before/after it, and then performs a final authority/outcome seal.

For claimed-success API L2, exact non-pass reasons are partitioned completely:

- `HOPPER_API_L2_AUTHORITY_REASONS_V1 = (planning_deadline_contract_mismatch, planning_request_contract_mismatch, hopper_authority_contract_mismatch, hopper_profile_contract_mismatch, terrain_snapshot_identity_mismatch, terrain_snapshot_hash_mismatch, hopper_terrain_geometry_contract_mismatch)`. After mandatory reseal, API returns the same higher-ranked `INTERNAL_ERROR` authority reason at its detecting stage; API-only detail `actual` is that reason and `expected="stable_hopper_authority"`.
- `HOPPER_API_L2_VALIDATOR_REASONS_V1 = (terrain_query_contract_mismatch, hopper_numeric_contract_mismatch, hopper_jump_oracle_contract_mismatch, hopper_route_oracle_contract_mismatch)`. API maps these to `INTERNAL_ERROR / hopper_route_oracle_contract_mismatch / provider_postcondition`, with `actual` equal to the carrier reason and `expected="hopper_route_l2_valid"`.
- `HOPPER_API_L2_PROVIDER_INVALID_REASONS_V1 = (route_hash_contract_mismatch, hopper_primitive_structure_mismatch, hopper_primitive_contract_mismatch, hopper_route_structure_mismatch, hopper_route_start_mismatch, hopper_route_connectivity_mismatch, hopper_route_goal_mismatch, hopper_route_cost_contract_mismatch, hopper_route_probability_contract_mismatch, hopper_launch_unknown, hopper_launch_unsafe, hopper_arc_boundary_violation, hopper_arc_unknown, hopper_arc_clearance_violation, hopper_landing_probability_below_threshold, hopper_landing_zone_unknown, hopper_landing_zone_unsafe, hopper_landing_slope_exceeded, hopper_landing_height_unreachable, hopper_landing_theta_unreachable, hopper_stop_condition_failed, hopper_route_state_budget_exceeded, hopper_memory_budget_exceeded, hopper_replay_work_budget_exceeded)`. API maps every one uniformly to `INTERNAL_ERROR / hopper_provider_outcome_contract_mismatch / provider_postcondition`, with `actual` equal to the carrier reason and `expected="hopper_route_l2_valid"`; it never exposes the inner validation/resource reason as the claimed-success outcome.
- A well-formed `planning_deadline_expired` carrier is followed by mandatory reseal and the exact API deadline checkpoint. If the checkpoint is expired, the trusted API timeout wins; otherwise the inconsistent carrier maps to route-oracle mismatch with `actual="planning_deadline_expired"` and `expected="deadline_checkpoint_consistent"`. Any other carrier reason, invalid field/prefix payload, non-carrier, subclass, or ordinary validator exception maps to route-oracle mismatch with `actual="malformed_or_exception"` for malformed/exception cases (otherwise the unsupported exact reason) and `expected="hopper-route-validation-result/v1"`.

All three postcondition remaps use the API-only exact details/zero-telemetry template below. The distinct structure/start/connectivity/goal/cost/probability/hash/helper/resource/semantic reasons remain authoritative inside `HopperRouteValidationResultV2` and provider-internal L2; the API remap does not erase their testability. Authority drift and an API deadline checkpoint still outrank outcome/validator remaps under the total order below.

Persistent search memory is deliberately a provider-algorithm property, not route evidence: the provider audits its internal ledger before ending the search scope and before invoking route L2. The unchanged public success/telemetry shapes do not carry enough history for API L2 to reproduce admitted/stale/retired queue records, so API L2 must not invent that claim. If a future contract requires independent API audit of search history, it needs a new public success/telemetry schema rather than overloading this fixture contract.

Provider failures undergo the exact structural/authority audit below: exact type, request/platform, allowed category/reason/stage, evidence-key/value shape, telemetry invariants, outcome token, and full Hopper authority pre/post seal. This prevents an unregistered reason or malformed failure from bypassing route validation merely because generic `_outcome_is_valid()` accepts request/platform fields. It does not prove that a well-shaped search history actually occurred: persistent-memory and search-exhaustion truth remain properties of the sealed trusted provider algorithm. Independent proof of that history would require a new public transcript/success/failure schema and is outside Gate5B.

### Frozen failure reasons

#### Existing shared API/authority reasons retained

- `platform_profile_unresolved`, `terrain_snapshot_invalid`, and the existing generic endpoint terrain-query reason unchanged.
- `primitive_provider_unregistered`, `primitive_provider_profile_mismatch`.
- `primitive_provider_exception` for an ordinary exception escaping the bound provider `plan` call after Hopper post-call reseal.
- `planning_request_contract_mismatch`, `planning_deadline_contract_mismatch`.
- `terrain_snapshot_identity_mismatch`, `terrain_snapshot_hash_mismatch`, `terrain_query_contract_mismatch`.
- `route_hash_contract_mismatch`, `planning_deadline_expired`.

#### Hopper capability reasons

- `hopper_proxy_profile_incomplete`, `hopper_parameter_set_unsupported`.
- `hopper_stop_condition_unsupported`, `hopper_energy_model_unsupported`.
- `hopper_risk_objective_unsupported`, `hopper_accelerator_required_unsupported`.
- `hopper_hold_oracle_unavailable`.

`hopper_midcourse_correction_unsupported` is intentionally absent: Gate5A freezes the profile field to false, so mutation is a profile-contract mismatch rather than a reachable request capability.

#### Hopper authority/structure reasons

- `hopper_authority_contract_mismatch`, `hopper_profile_contract_mismatch`.
- `hopper_terrain_geometry_contract_mismatch`, `hopper_numeric_contract_mismatch`.
- `hopper_primitive_structure_mismatch`, `hopper_primitive_contract_mismatch`, `hopper_jump_oracle_contract_mismatch`.
- `hopper_route_structure_mismatch`, `hopper_route_start_mismatch`, `hopper_route_connectivity_mismatch`, `hopper_route_goal_mismatch`.
- `hopper_route_cost_contract_mismatch`, `hopper_route_probability_contract_mismatch`, `hopper_route_oracle_contract_mismatch`.
- `hopper_search_cost_contract_mismatch`, `hopper_provider_outcome_contract_mismatch`.

#### Hopper semantic reasons

- `hopper_goal_heading_unreachable`.
- `hopper_launch_unknown`, `hopper_launch_unsafe`.
- `hopper_arc_boundary_violation`, `hopper_arc_unknown`, `hopper_arc_clearance_violation`.
- `hopper_landing_probability_below_threshold`.
- `hopper_landing_zone_unknown`, `hopper_landing_zone_unsafe`, `hopper_landing_slope_exceeded`, `hopper_landing_height_unreachable`, `hopper_landing_theta_unreachable`.
- `hopper_stop_condition_failed`.

#### Hopper resource/search reasons

- `hopper_expansion_budget_exhausted`, `hopper_route_state_budget_exceeded`, `hopper_memory_budget_exceeded`, `hopper_replay_work_budget_exceeded`.
- `hopper_no_complete_route`.

Pass-only oracle reasons `hopper_jump_l2_valid` and `hopper_route_l2_valid` are not failure reasons.

### Category and exact stage universe

- Capability/model/risk/accelerator/hold unsupported: `UNSUPPORTED_CAPABILITY`, stage `capability_preflight`.
- Goal-heading incompatibility: `GOAL_POSE_UNREACHABLE`, stage `goal_preflight`.
- Initial launch unknown/unsafe only: `UNSAFE_START`, stage `launch_validation`, semantic evidence phase `initial_launch`. The same reason surfaced by complete-route replay is `VALIDATION_FAILED` with phase `route_replay`; an ordinary search-action rejection is never returned directly.
- Complete-route semantic replay failure: `VALIDATION_FAILED`, one of `launch_validation`, `arc_validation`, `landing_probability`, `landing_validation`, `stop_validation`, or `route_validation`.
- Expansion/route/memory/replay caps: `RESOURCE_LIMIT`, respectively `search_expansion`, `route_state_admission`, `search_memory`, or `replay_work`.
- Search exhaustion: `NO_COMPLETE_ROUTE / hopper_no_complete_route`, stage `hopper_search`.
- Deadline: `TIMEOUT / planning_deadline_expired`, the exact checkpoint stage.
- Request/deadline/authority/profile/snapshot/query/geometry/numeric/structure/cost/oracle/outcome mismatch: `INTERNAL_ERROR`, the exact detecting stage.

The allowed Hopper stage strings are exactly: `provider_authority`, `capability_preflight`, `provider_dispatch`, `provider_execution`, `provider_completion`, `goal_preflight`, `search_setup`, `search_expansion`, `route_state_admission`, `search_memory`, `launch_validation`, `arc_candidate_enumeration`, `arc_validation`, `landing_probability`, `landing_validation`, `stop_validation`, `replay_work`, `hopper_search`, `route_validation`, and `provider_postcondition`. Existing API stages `profile_resolution`, `terrain_validation`, `start_safety`, `goal_safety`, and `provider_resolution` retain their current behavior.

#### Exact provider-failure structural policy

Generic failures constructed by API before/around dispatch remain under their existing cross-platform policy: `platform_profile_unresolved`, `terrain_snapshot_invalid`, generic endpoint-query reasons, `primitive_provider_unregistered`, `primitive_provider_profile_mismatch`, and the post-exception trusted `primitive_provider_exception`. API-only Hopper postcondition constructions also include `hopper_provider_outcome_contract_mismatch` and `hopper_route_oracle_contract_mismatch / provider_postcondition`; a provider may use the latter reason only at its own `route_validation` stage. A provider-returned Hopper failure may not claim any API-only reason/stage. Its stage must belong to the exact tuple `HOPPER_PROVIDER_FAILURE_STAGES_V1 = (provider_authority, capability_preflight, goal_preflight, search_setup, search_expansion, route_state_admission, search_memory, launch_validation, arc_candidate_enumeration, arc_validation, landing_probability, landing_validation, stop_validation, replay_work, hopper_search, route_validation)`, and its reason must match this exact immutable policy:

An API-only Hopper postcondition failure uses `INTERNAL_ERROR`, stage `provider_postcondition`, `checks=(reason_code,)`, exact contract/mismatch details `actual`, `expected`, `phase` with `phase="provider_postcondition"`, and fresh zero-count telemetry whose termination reason is the API-created reason, timeout/accelerator/Ackermann flags are false, and elapsed time is the sealed deadline value. It never copies untrusted provider telemetry. API-created authority drift uses the same detail/telemetry template with its higher-ranked authority reason and `phase` exactly equal to its actual API detecting stage. These rules are separate from provider-returned structural acceptance.

- The seven Hopper capability reasons map only to `UNSUPPORTED_CAPABILITY / capability_preflight`; `hopper_goal_heading_unreachable` maps only to `GOAL_POSE_UNREACHABLE / goal_preflight`.
- Launch reasons map only as the phase-sensitive rule above. Arc reasons map to `VALIDATION_FAILED / arc_validation`; landing probability to `VALIDATION_FAILED / landing_probability`; landing zone/slope/height/theta to `VALIDATION_FAILED / landing_validation`; stop failure to `VALIDATION_FAILED / stop_validation`.
- The four resource reasons map exactly to `RESOURCE_LIMIT` and their four stages in the order expansion/route-state/memory/replay above. `hopper_no_complete_route` maps only to `NO_COMPLETE_ROUTE / hopper_search`. A provider-returned `planning_deadline_expired` maps only to `TIMEOUT` and one member of `HOPPER_PROVIDER_DEADLINE_STAGES_V1 = (capability_preflight, search_setup, search_expansion, launch_validation, arc_candidate_enumeration, arc_validation, landing_probability, landing_validation, stop_validation, replay_work, route_validation)`. API-created deadline failures separately use exactly `provider_dispatch`, `provider_completion`, or `provider_postcondition`.
- All contract/mismatch reasons map to `INTERNAL_ERROR`. Request/deadline/Hopper/profile/snapshot identity/hash/geometry mismatch may use any member of `HOPPER_PROVIDER_FAILURE_STAGES_V1` where its mandatory seal detects drift. `terrain_query_contract_mismatch` is limited to `launch_validation`, `arc_validation`, `landing_validation`, or `route_validation`. `hopper_numeric_contract_mismatch` is limited to `search_setup`, `arc_candidate_enumeration`, `arc_validation`, `landing_probability`, `landing_validation`, `stop_validation`, `replay_work`, or `route_validation`. Primitive structure/contract/jump-oracle mismatch is limited to `search_expansion` or `route_validation`; provider-returned route structure/start/connectivity/goal/cost/probability/oracle and route-hash mismatch is only `route_validation`; search-cost mismatch is only `search_expansion`. Provider-outcome mismatch is never provider-returned.

For every provider-returned failure, `evidence.checks` is exactly `(reason_code,)`, `search_telemetry.termination_reason` is exactly `reason_code`, `timed_out` is true if and only if the reason is `planning_deadline_expired`, and both accelerator/Ackermann booleans are false. Failure evidence details use exactly one sorted template, with every listed key present and non-applicable values encoded as exact `None`:

- Capability: `actual`, `expected`, `parameter_set_id`, `profile_id`.
- Goal: `actual`, `expected`.
- Semantic or no-route representative: `action_key`, `actual`, `candidate_id`, `cell_x`, `cell_y`, `hop_index`, `parameter_set_id`, `phase`, `reason_rank`, `segment_index`. For a direct semantic failure `actual` is the direct reason; for `hopper_no_complete_route` it is the ranked representative local reason or `None`. A representative never replaces the outer no-route reason.
- Contract/mismatch: `actual`, `expected`, `phase`.
- Expansion: `attempted_expanded_states`, `max_expanded_states`.
- Route state: `attempted_route_states`, `effective_max_route_states`, `requested_max_route_states`.
- Replay work: `attempted_work_units`, `max_work_units`, `phase`.
- Memory: `accounting_id`, `admitted_record_count`, `attempted_accounted_bytes`, `effective_max_memory_bytes`, `max_memory_bytes`, `persistent_accounted_bytes`, `phase`, `transient_reserved_bytes`.
- Deadline: empty details.

The memory phase is exactly one of `root_admission`, `ballistic_build`, `arc_oracle`, `landing_build`, `child_admission`, `route_materialization`, or `route_l2`. Replay-work phase names the exact Section 1 counter. Failure detail keys are drawn only from the fixed lexicographic whitelist `accounting_id`, `action_key`, `actual`, `admitted_record_count`, `attempted_accounted_bytes`, `attempted_expanded_states`, `attempted_route_states`, `attempted_work_units`, `candidate_id`, `cell_x`, `cell_y`, `effective_max_memory_bytes`, `effective_max_route_states`, `expected`, `hop_index`, `max_expanded_states`, `max_memory_bytes`, `max_work_units`, `parameter_set_id`, `persistent_accounted_bytes`, `phase`, `profile_id`, `reason_rank`, `requested_max_route_states`, `segment_index`, `transient_reserved_bytes`. This whitelist applies to provider-returned Hopper failures; existing API-constructed generic details retain their existing keys.

The exact replay counter IDs, in frozen algorithm order, are `HOPPER_REPLAY_COUNTER_IDS_V1 = ("hopper_arc_interval_count/v1", "hopper_arc_candidate_width/v1", "hopper_arc_candidate_height/v1", "hopper_arc_interval_cartesian_product/v1", "hopper_arc_distinct_cell_count/v1", "hopper_arc_interval_cell_visit_count/v1")`; replay-work `phase` must equal exactly one member. Direct semantic `phase` is only `initial_launch` or `route_replay`; no-route representative `phase` is exactly `search_exhaustion`; contract/mismatch `phase` must exactly equal `evidence.stage`. Memory phases use only their seven-value tuple above. No other provider-failure phase string is accepted.

Every detail key is an exact built-in nonempty string. All count/byte/cap values are exact nonnegative built-in ints and reject `bool`. `cell_x`, `cell_y`, `hop_index`, `segment_index`, and `reason_rank` are exact built-in ints or exact `None` only where non-applicable; when present they reject `bool`, hop/segment/rank are nonnegative, and OOB cells may be signed. `action_key`, `candidate_id`, `parameter_set_id`, `profile_id`, and `phase` are exact nonempty built-in strings or exact `None` only where their template marks non-applicable. `actual`/`expected` are exact JSON scalars; any float is finite canonical binary64 and any boolean is exact built-in bool. Template presence, key order, and these value types are part of API structural audit.

Provider failure telemetry uses the same increment points as success. Before search expansion all five counts are exact zero. During/after search, `rejected_l0=rejected_l1=0`, `rejected_l2<=generated_primitives`, and the generated-prefix bounds above apply. Ordinary no-route exhaustion has exact `generated_primitives=192*expanded_states`; a global abort may stop at the prefix bound. `elapsed_s` is exact finite nonnegative canonical binary64 and is word-sealed, but the API structural audit cannot prove elapsed time or search-history truth from scalar telemetry. A shape-valid lie from the exact bound provider remains an implementation-authority violation, not independently refutable route evidence; a future stronger threat model requires a public transcript schema.

Semantic `reason_rank` values are frozen: launch unknown/unsafe `10/11`; arc boundary/unknown/clearance `20/21/22`; landing probability `30`; landing unknown/unsafe/slope/height `40/41/42/43`; landing theta `50`; stop failure `60`. Internal `primitive_index` is exposed as zero-based `hop_index`; arc segment index or landing-prefix rank is exposed as zero-based `segment_index`. A non-applicable optional value is evidence `None`, and deterministic ordering encodes an optional integer as `(0,value)` when present or `(1,0)` when absent. Within one primitive use `(reason_rank, optional(segment_index), optional(cell_y), optional(cell_x))`; across a route prefix `hop_index`; across search representatives then `action_key` are inserted before segment/cell fields. This is the only winner-selection order.

### Authority-drift total order and no-later-call rule

At any checkpoint, simultaneously observable failures use this total order:

1. `planning_deadline_contract_mismatch`.
2. `planning_request_contract_mismatch`.
3. `hopper_authority_contract_mismatch`.
4. `hopper_profile_contract_mismatch`.
5. `terrain_snapshot_identity_mismatch`.
6. `terrain_snapshot_hash_mismatch`.
7. `hopper_terrain_geometry_contract_mismatch`.
8. Input/primitive/route token drift, ordered primitive contract then `route_hash_contract_mismatch` then provider outcome contract.
9. `planning_deadline_expired`.
10. Helper-output mismatch for the active phase: terrain query, specialized-ballistic numeric/two-sample/cadence/direction contract, jump oracle, route oracle, then cost/probability contract.
11. Resource failure, then the phase's ranked semantic result. Within resource failures the executed checkpoint order is authoritative: setup expansion minimum -> route-state minimum -> root memory; action helper hard/work cap -> helper memory; accepted primitive route-state admission -> child memory; route materialization memory; route-L2 route-state/work/memory in replay order.

An ordinary helper exception does not bypass its mandatory post-call reseal; `KeyboardInterrupt`, `SystemExit`, and `MemoryError` propagate. Specialized ballistic entry/post-call reseal includes read-only private module-binding identity drift detection, while the actual call always uses the captured authority object; generic helper rebinding is irrelevant. Semantic winner selection uses only the frozen rank/optional/hop/action/segment/cell encoding above; there is no second tuple or implicit container order. Once selected, no later-phase helper runs.

Cross-layer order is fixed:

1. API: profile resolution -> snapshot validation -> start safety -> goal safety -> provider resolution/profile binding -> Hopper authority seal -> existing pre-dispatch deadline checkpoint -> dispatch. An expiry here returns timeout without calling provider preflight.
2. Provider capability: authority/profile seal -> incomplete -> parameter set -> stop -> energy -> risk -> accelerator -> deadline -> goal heading/hold -> resource setup.
3. Primitive: structure/authority -> launch -> arc -> pure landing-prefix reconstruction/probability -> landing terrain/mean footprint -> theta replay -> stop.
4. Route L2: entry authority/digest/resource token -> route structure/start/connectivity/goal -> primitives by index with route-state/work/memory checks -> cost/probability/hash -> pass evidence.
5. API completion/postcondition: mandatory post-call authority/outcome seal -> `provider_completion` deadline checkpoint -> exact failure audit or success independent route L2 -> post-L2 authority/outcome/digest seal -> `provider_postcondition` deadline checkpoint -> final authority/outcome seal. A deadline flag never suppresses the final mandatory seal; simultaneous results use the authority-drift total order, then deadline, then outcome/oracle semantics.

No-later-call consequences:

- API profile/snapshot/start/goal or pre-dispatch timeout failure: zero provider/Hopper operational helper calls.
- Capability failure: zero Hopper ballistics/oracle/action/search/route-L2 calls; generic endpoint queries may already have occurred.
- Goal or resource-setup failure: zero action generation/primitive-oracle/search insertion.
- Launch failure: zero arc/landing/theta/stop calls. Arc failure: zero landing/theta/stop calls.
- Probability failure: zero landing-terrain/theta/stop calls. Landing failure: zero theta/stop calls. Theta failure: zero stop calls.
- Stop failure: zero primitive emission/cost/search insertion.
- Local semantic rejection may advance only to the next fixed action. Authority/deadline/numeric/resource global failure, including a specialized helper binding, direction, flight-time, `S != 2`, or cadence mismatch, permits no later action or final L2.
- Provider failure gets the exact structural/authority API audit but no route validator. Claimed success with any member of the complete provider-invalid L2 partition (structure/start/connectivity/goal/cost/probability/hash, semantic, or resource) becomes provider-outcome mismatch and cannot be returned as success.
- If bound `plan()` raises an ordinary exception, API first performs the mandatory Hopper post-call seal; authority drift wins, then the `provider_completion` deadline wins, otherwise API constructs trusted `INTERNAL_ERROR / primitive_provider_exception / provider_execution`. Critical process exceptions propagate.

### RED boundary matrix

- Exact field/type/frozen/slot/schema contracts; mutated nested fields, subclasses, signed-zero canonicalization, NaN/inf rejection, and wrong authority tokens.
- All 192 actions and exact key/order; all 16 canonical azimuth words/directions; exact cardinal `0/4/8/12` components at nonzero and large origins; equal-range 30/60-degree actions both replayed. A 192-action matrix must prove that A2 removes only raw-libm cardinal residuals, while true nonzero displacement absorption/underflow/overflow still causes global numeric abort.
- State-key tag, root primitive-key sentinel, exact-float cycles, opposite actions, nonzero/large origins, exact cost ties, monotonic candidate IDs, stale/closed behavior, and seed/hash independence.
- Goal only on dequeue; start==goal hold blocker; no snap/tolerance/partial route.
- Primitive carries no samples/zone; transient cap and persistent/peak-memory accounting catch adversarial large evidence.
- `max_route_states` requested/effective/attempted boundaries use the same `min(requested,100_001)` and subtraction-safe formula in child admission and both L2 callers. Tests freeze generic identity `1+sum(S_i-1)`, operational exact `S_i=2`, child delta `1`, and route count `H+1`; any alternate production sample count is numeric mismatch. Replay-state authority remains distinct from observation's endpoint-only `H+1`, and route state remains independent from hop, memory, and work caps even though the operational counts are arithmetically related.
- Deterministic memory boundaries cover root/child admission, improved and stale history with no refund, requested zero and over-hard-limit clamping, explicit-cap ballistic/landing worst-case reservation, exact-arena bit/slot caps, phase disposal, complete outcome materialization, and provider/API L2. Tests forbid runtime-size/RSS authority and forbid L2 from claiming it reconstructed persistent search history.
- Cap-aware helper tests mutate legacy module globals, generic and specialized helper module attributes, explicit cap arguments, specialized helper identity/ID, and every resource-token coefficient. Hopper calls only the captured specialized ballistic helper and capped landing helper; specialized binding drift is caught by mandatory reseal, generic ballistic rebinding has no dispatch effect, every true authority mismatch is caught before materialization, and no temporary global rewrite is allowed.
- Full success envelope tests freeze observation source and `H+1` endpoint-only states, independently rebuilt replay-state `H+1`, exact two-sample/final-time/cadence/direction postconditions, zero observation metrics, L2 evidence, cost, cache, probability carrier, success telemetry, and `R(H)` residency. No sample/landing/cadence evidence may survive into the outcome.
- Provider-failure structural-policy tests cover every reason/category/stage/template row, exact checks/termination/timed-out values, exact-int/no-bool telemetry, request expansion upper bound, nonzero success/no-route expansion, the trusted provider's separate final-L2 expansion invariant, counter increment boundaries, and honest inability to prove well-shaped search history from an outward failure.
- Route-result carrier tests freeze exact fields/order/schema, six counter pairs, phase-dependent optional payload, pass/non-pass evidence, subclasses, and token reseal. Claimed-success API L2 exhaustively tests authority-preserving, validator-oracle, provider-invalid semantic/structure/cost/hash/resource, deadline-consistency, unsupported-reason, malformed, and exception partitions including exact `actual`/`expected` strings.
- Risk nonzero rejected; raw-time and energy operation order; exact component/total tokens and lower-cost-to-closed mismatch.
- First/middle/last primitive type/token/disconnection/start/goal/cost/mass tampering and route hash drift have distinct stable reasons inside the route-result carrier and provider-internal L2; claimed-success API postcondition remaps the complete provider-invalid partition as frozen above.
- Probability-before-terrain-query spy; per-hop `0.99`; two-hop union-bound diagnostic rounded downward without whole-route 99% claim.
- Success and failure mutation during API calls; ordinary helper exception plus simultaneous authority drift; every adjacent total-order/no-later-call pair.
- Public/generic ballistic bytes and failure priority remain unchanged; the private Hopper helper rejects wrong index type/range and all `dt_s` inputs, stays absent from every public export, and private direct-core fine/`nextafter` cadence proves partition-equivalent full-arc safety without becoming production configuration.
- Wheel/legged API authority and v1 default behavior remain unchanged under Hopper registration.


## Part IV — Approved serial implementation and verification contract

### Status and authority boundary

- High-level option A is selected.
- Sections 1, 2, and 3 were explicitly approved on 2026-07-20. This readiness/implementation contract was explicitly approved on 2026-07-21.
- Gate5B cadence/cardinal correction A2 was explicitly approved on 2026-07-21. Its authority is limited to this tracked amendment, serial 11B1.5 TDD/review, and the amended 11B2-11B7 implementation contract.
- Approval order is strictly: Section 1 -> Section 2 -> Section 3 -> explicit readiness-plan/implementation authorization -> A2 cadence/cardinal correction authorization -> tracked A2 amendment -> 11B1.5 and remaining serial code/review -> separate explicit controller authorization -> formal blocked snapshot. Approval of any design section or A2 never approves a later gate or the controller; completing code and review never authorizes the controller step.
- This file is the exact tracked plan `docs/superpowers/plans/2026-07-20-multiplatform-path-planner-v2-gate5b.md`. It is self-contained: Parts I-III carry the complete approved Sections 1-3+A2 contracts and Part IV carries the complete amended execution order. Links or hashes to ignored drafts/proposal are provenance only; ignored files are never semantic or implementation authority.
- This plan is promoted in one root-only commit whose parent is the pre-promotion anchor `0b6c12d8a5228a7f64f5b86ad0e6f0ba6181bb4e`, whose only tree change is this new tracked plan, and whose `path-planner` gitlink remains `57c28fb2709aec4e043827a50425906a4cbddecc`. The resulting clean parent HEAD becomes the implementation anchor; later implementation checks must reproduce that new anchor or an explicitly reviewed descendant, not require the pre-promotion anchor again.
- This tracked plan authorizes only 11B1-11B7 serial TDD implementation including 11B1.5, reviews, allowed commits/integrations, and the explicitly bounded D-drive non-formal verification. It does not authorize a controller run, formal data access, `D:/xunce/out/path_v2/g5`, a formal blocked snapshot, Gate6, checkpoint publication, default-policy replacement, executor connection, or canary start.
- The approved design authorities are the three Section contracts as narrowly amended by approved A2. This plan controls execution order and verification and makes that amendment self-contained.

### Authoritative starting evidence

- Parent branch: `codex/multiplatform-path-planner-v2`.
- Pre-promotion parent anchor: `0b6c12d8a5228a7f64f5b86ad0e6f0ba6181bb4e`; the authorized root-only tracked-plan commit creates and records the new implementation anchor.
- Nested HEAD and parent gitlink: `57c28fb2709aec4e043827a50425906a4cbddecc`.
- Parent and nested tracked trees are clean.
- Gate5A verified baselines: focused `420 passed`; named v2 `1461 passed`; nested full `1641 passed, 17 skipped`; root Gate0 plus gate-runner `322 passed`; PPO Stage1 `43 passed` plus the exact frozen 13-nodeid inherited failure set, with zero errors/skips.
- `configs/xunce_path_v2_gate5_hopper_v1.json` and `D:/xunce/out/path_v2/g5` are absent.
- The protected C-drive Stage6 worktree is out of scope: never read it, run commands in it, or use its contents as evidence.
- At A2 amendment entry, the parent HEAD is exact `2f0e05e006f70148ba9dcc334a47c5bf53957ea5`, nested HEAD is exact reviewed 11B1 descendant `59d012375745a862fbdd9c92a92875a671e82ff3`, the nested tracked tree is clean, and the parent's only worktree difference is the expected unstaged `path-planner` gitlink drift. The A2 amendment changes only this tracked plan and does not integrate the gitlink.

### Why original Task 11 is replaced

The original Task 11 combines oracle, provider, API, exports, runner, and formal evidence into one change. It also says incomplete capability fails during provider construction, which conflicts with the current `PrimitiveProviderV2.plan(...) -> PlanningOutcomeV2` contract and the approved Gate5A preflight seam. Gate5B therefore replaces that task with six nested serial TDD slices, the narrowly inserted serial 11B1.5 A2 repair slice, and one final root blocked-evidence slice.

### Common TDD and integration protocol

For every nested slice:

1. Reconfirm branch, exact parent/nested lineage, absent formal output, and forbidden-surface diff under the applicable state contract: 11B1 entry requires both tracked trees clean and gitlink equality at the recorded implementation anchor; 11B1.5 entry requires the A2-amended parent plan descendant, exact reviewed 11B1 nested anchor `59d012375745a862fbdd9c92a92875a671e82ff3`, nested clean, and only the expected unstaged parent `path-planner` gitlink drift; 11B2 and 11B3 entry may show only that expected gitlink drift caused by already committed and reviewed nested slice work, while every other parent path and the nested tracked tree remain clean and nested HEAD is an exact descendant of the last sealed nested anchor. After the reviewed 11B1-11B3 milestone is integrated, and at every later post-integration entry, both tracked trees must again be clean with gitlink equality. Any other tracked change or lineage deviation stops implementation.
2. Write only the slice's RED tests. Use `--collect-only` to freeze the exact new nodeid set and delta before accepting RED.
3. RED may fail only on the newly frozen nodeids for the exact missing surface/behavior. Existing nodeids must stay green except the exact five explicitly superseded by 11B1.5 below; no new collection error or skip is allowed.
4. Commit tests-only RED, then implement the smallest complete approved contract and commit production-only GREEN. Never weaken RED to fit implementation.
5. Run the slice-focused suite only through the complete D-drive/import-isolation envelope below.
6. Obtain fresh spec and quality review. Slice 11B1.5 and slices 11B2 through 11B5 also require fresh adversarial safety/resource review. Any review fix begins with a reproducing RED.
7. Update exact test deltas from collection evidence; never predict counts before tests exist.

There is no registered Gate5 pytest marker. RED/GREEN evidence therefore uses the exact newly frozen nodeids or the listed test files, never a broad `-k gate5` approximation. Every collect-only, RED, GREEN, focused, named-v2, nested-full, root, and PPO pytest invocation uses its own fresh D-drive temp root and the complete isolation envelope below:

```powershell
$repoRoot = (Resolve-Path "D:/codex/worktrees/multiplatform-path-planner-v2").Path
$nestedRoot = (Resolve-Path "$repoRoot/path-planner").Path
$tempRoot = "D:/xunce/tmp/path_v2_g5b/<slice>/<fresh-id>"
if (Test-Path -LiteralPath $tempRoot) { throw "pytest temp root must be fresh" }
New-Item -ItemType Directory -Path "$tempRoot/mpl" -Force | Out-Null
$env:PYTHONNOUSERSITE = "1"
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
$env:TEMP = $tempRoot
$env:TMP = $tempRoot
$env:MPLCONFIGDIR = "$tempRoot/mpl"

# Nested tests use only the nested source tree.
$env:PYTHONPATH = "$nestedRoot/src"

# Root and PPO tests instead use both D-drive source trees, in this order.
$env:PYTHONPATH = "$repoRoot/src$([IO.Path]::PathSeparator)$nestedRoot/src"
```

Each nested pytest command first `Push-Location $nestedRoot`; each root/PPO command first `Push-Location $repoRoot`; every command restores the prior location in `finally`. Each pytest command also passes `-o addopts='' -p no:cacheprovider --basetemp "$tempRoot/basetemp"`. The two `PYTHONPATH` assignments are alternatives selected by suite, not combined sequential setup. Import-origin checks must prove both packages resolve inside this D-drive worktree before accepting broad evidence.

Shared files are serial-only. No two agents may edit `ballistics.py`, `hopper_authority.py`, `oracles/hopper.py`, `providers/hopper.py`, `hopper_route_validation.py`, `hopper_api.py`, any Hopper test file reused across slices, `api.py`, export files, or root runner/config files concurrently.

Static import ownership is frozen: `hopper_authority` may import only lower-level ballistics/profiles/contracts; `oracles.hopper` may import the authority and lower-level modules but never a provider, validator, or API module; `providers.hopper` may import the authority/oracle but must not statically import the route validator; `hopper_route_validation` may import provider types plus the authority/oracle; `hopper_api` may import contracts/authority/provider/validator but never `api.py`; and `api.py` may import `hopper_api`. The provider captures the trusted route validator only by a runtime import at the final-L2 call site, matching the existing legged cycle break. A deviation requires a new RED plus design/plan re-approval before code.

### Seven main serial slices plus the inserted 11B1.5 repair slice

#### 11B1 - Captured helpers and neutral resource authority

Files:

- Modify `path-planner/src/path_planner/v2/ballistics.py`.
- Modify `path-planner/tests/test_v2_ballistics.py`.
- Create `path-planner/src/path_planner/v2/hopper_authority.py`.
- Create `path-planner/tests/test_v2_hopper_authority.py`.

Scope:

- Add `sample_ballistic_arc_capped_v2(..., max_sample_count=...)` and `landing_zone_cells_capped_v2(..., max_candidate_count=...)` as exact explicit-cap entry points. This 11B1 historical baseline initially captures the generic ballistic helper; approved 11B1.5 narrowly replaces only the Hopper ballistic binding/ID/wrapper contract while preserving the generic helper unchanged.
- Existing public wrappers keep their exact signatures and behavior and delegate with the existing canonical global caps.
- Explicit caps reject bool/subclasses and are checked before allocation. Captured helpers never reread mutable module cap globals.
- Freeze the exact `HopperResourceAuthorityV2` record/singleton, helper identities/IDs, all caps and byte coefficients, 512 MiB hard accounted ceiling, `HOPPER_EXACT_MAX_INTEGER_BITS_V2=262_144`, 128-live-slot arena, exact-operation pre-admission API, and deterministic resource token before any oracle exists.
- Do not add the Section 2 fixture registry yet. Do not change exports, public dataclass shapes, numeric semantics, terrain behavior, or Gate5A test meaning.

Focused GREEN command:

```powershell
D:/conda_envs/lunar-explorer/python.exe -m pytest -o addopts='' -p no:cacheprovider `
  --basetemp "$tempRoot/basetemp" -q `
  tests/test_v2_ballistics.py `
  tests/test_v2_hopper_authority.py
```

#### 11B1.5 - A2 cadence/cardinal specialized-helper repair

This slice runs strictly after reviewed 11B1 and before any 11B2 work. It is a narrow approved supersession of the old Hopper generic-ballistic binding, not permission to weaken 11B1 resource, arena, cap, TOCTOU, or exception contracts.

RED files only:

- Modify `path-planner/tests/test_v2_ballistics.py`.
- Modify `path-planner/tests/test_v2_hopper_authority.py`.

The tests-only RED commit may update assertions inside exactly these five pre-existing nodeids because their old identity/ID/signature/rebind target is intentionally superseded:

1. `tests/test_v2_hopper_authority.py::test_hopper_resource_authority_freezes_exact_record_singleton_and_values`: generic ballistic identity/old ID becomes private specialized identity/new ID; all record fields/order/schema/caps/bytes remain asserted.
2. `tests/test_v2_hopper_authority.py::test_hopper_resource_authority_tokens_follow_exact_frozen_order`: both tokens freeze the new ID/identity and prove the old ID absent; token schema/order and all other values remain asserted.
3. `tests/test_v2_hopper_authority.py::test_captured_resource_helpers_use_sealed_caps_not_legacy_globals`: ballistic arguments become exact index/no-cadence and expected replay uses the specialized direct contract; fixed-cap and legacy-global isolation assertions remain.
4. `tests/test_v2_hopper_authority.py::test_captured_resource_helpers_fail_before_rebound_module_callable`: generic capped-helper rebinding becomes an explicit no-effect case, specialized module-binding drift is rejected before call, and landing behavior remains unchanged.
5. `tests/test_v2_hopper_authority.py::test_captured_resource_helpers_fail_before_every_token_field_drift`: only `invalid_ballistic_args` changes from the old raw-azimuth/cadence call shape to exact `(object(), 3.0, pi/4.0, 0, 1.62)` semantics; exhaustive canonical-singleton, subclass, every-token-field, equality-forger, and pre-operation rejection assertions remain unchanged.

No other existing nodeid may be deleted, renamed, parametrically narrowed, skipped, or have an assertion relaxed. Those five updates preserve their pre-existing substantive authority-record, token-order, fixed-cap, legacy-global isolation, pre-call TOCTOU/no-dynamic-dispatch, exhaustive token-drift, equality-forger, pre-operation rejection, and error-precedence assertions under the new specialized contract. Together with the nine new nodes, the final suite additionally freezes specialized normal/ordinary-exception post-call reseal, authority-drift precedence, and critical-exception propagation.

The RED assertions freeze these exact boundaries:

- Private helper signature is exactly `(start, speed_mps, elevation_rad, azimuth_index, g_mps2, *, max_sample_count)`; raw azimuth, direction, `dt_s`, extra positional cadence, and extra keyword cadence are absent/rejected. Exact-index type/range rejects bool, subclasses, and values outside `0..15`.
- `max_sample_count` remains an exact positive built-in `int`: bool, int subclasses, and coercible non-ints raise the exact-int `TypeError`; `0` and `-1` raise the positive-value `ValueError`; cap `1` rejects the required `sample_count=2` only after valid representability, while caps `2` and `100_000` pass. The specialized path preserves the existing post-representability cap audit order rather than allowing an invalid cap to mask an earlier endpoint/apex representability failure.
- For each world origin `0.0`, `1.0`, `10.0`, and `1_000_000.0`, exercise the complete `4*3*16 = 192` matrix with speeds `(1.5, 2.0, 2.5, 3.0)`, elevations `(pi/6.0, pi/4.0, pi/3.0)`, all sixteen indices, and `max_sample_count=100_000`. Require exact canonical directions, endpoints, `flight_time`, and exactly two `(0.0, flight_time)` samples. This does not waive separate true absorbed-displacement, underflow, overflow, origin, or apex failures.
- The all-sixteen/two-sample behavior test monkeypatches/delegates the existing `_exact_positive_ratio_ceil` and `_interior_sample_times` behavioral seams: every recorded `(flight_time, cadence)` pair must have identical `.hex()` words, interval count must be exact `1` under the explicit cap, and a forbidden `_bounded_local_repair` sentinel must receive zero calls. It does not hardcode a result tuple or freeze any new arbitrary-direction core name/signature.
- Private visibility checks prove the helper absent from wildcard import, top-level `path_planner.v2`, `oracles`, `providers`, and every relevant `__all__`/public attribute surface.
- Authority wrapper tests freeze exact index-only forwarding and output, fixed captured cap, normal and ordinary-exception completion reseal with authority-drift precedence, and propagation of `KeyboardInterrupt`, `MemoryError`, and `SystemExit` under the existing critical-exception rule.
- The generic regression sentinel freezes pre-A2 public/generic raw-libm behavior, including the `pi/2` landing x residual `0x1.5c9f64561d398p-53` at start zero, speed `2.0`, elevation `pi/4.0`, gravity `1.62`, and `dt_s=flight_time`; nonzero-origin absorption failure; composite invalid-input failure priority; and generic fine/`nextafter` cadence behavior as independently produced private adversarial samples.
- Tests must use behavioral seams. They may not inspect source/AST, freeze trig call counts, replace the sealed specialized helper, use approximate comparisons for exact contracts, freeze an arbitrary-direction core name/signature, or depend on the not-yet-created 11B2 oracle.

The exact new A2 failing nodeids are:

1. `tests/test_v2_ballistics.py::test_hopper_ballistic_helper_freezes_private_no_cadence_signature`.
2. `tests/test_v2_ballistics.py::test_hopper_ballistic_helper_rejects_invalid_exact_action_index`.
3. `tests/test_v2_ballistics.py::test_hopper_ballistic_helper_derives_all_16_exact_directions_and_two_samples`.
4. `tests/test_v2_ballistics.py::test_hopper_ballistic_helper_covers_192_actions_at_nonzero_and_large_origins`.
5. `tests/test_v2_ballistics.py::test_hopper_ballistic_helper_keeps_real_representability_failures_global`.
6. `tests/test_v2_ballistics.py::test_hopper_ballistic_helper_is_module_private_and_unexported`.
7. `tests/test_v2_hopper_authority.py::test_captured_hopper_ballistic_wrapper_freezes_index_only_signature_and_forwarding`.
8. `tests/test_v2_hopper_authority.py::test_captured_hopper_ballistic_wrapper_reseals_normal_and_ordinary_completion`.
9. `tests/test_v2_hopper_authority.py::test_captured_hopper_ballistic_wrapper_preserves_critical_exception_precedence`.

RED also adds `tests/test_v2_ballistics.py::test_generic_ballistic_helpers_keep_pre_a2_bytes_and_failure_precedence` as a regression sentinel that must already pass against the pre-GREEN production tree. After exact collect-only evidence, accepted RED is exactly the five superseded existing nodeids plus the nine new A2 nodeids above: `14 failed`, zero errors/skips, every other pre-existing nodeid green, and the new generic regression sentinel green. Any different failure set stops the slice.

GREEN files only:

- Modify `path-planner/src/path_planner/v2/ballistics.py`.
- Modify `path-planner/src/path_planner/v2/hopper_authority.py`.

GREEN implements the exact private signature, index type/range checks, 16 canonical azimuth words, four exact cardinal directions with positive zero, raw `sin/cos` for the other twelve, internal same-word flight-time cadence, exact two-sample postcondition, unchanged generic bytes/failure precedence, absence from all public exports, `sample_hopper_ballistic_arc_capped/v1` binding/token revision, and index-aware `_call_captured_ballistic_helper_v2` with fixed captured cap and mandatory entry/normal/ordinary-exception reseals. Specialized module-binding drift must be detected read-only; actual dispatch uses only the captured object; generic helper rebinding must not affect Hopper. Existing real representability, cap, error-priority, arena, six-counter, slot, bit, and byte contracts remain fail-closed.

Focused GREEN command:

```powershell
D:/conda_envs/lunar-explorer/python.exe -m pytest -o addopts='' -p no:cacheprovider `
  --basetemp "$tempRoot/basetemp" -q `
  tests/test_v2_ballistics.py `
  tests/test_v2_hopper_authority.py
```

After GREEN, obtain fresh spec, quality, and adversarial safety/resource review. Any review fix begins with a reproducing RED. Do not start 11B2 until those reviews pass, and do not integrate the parent gitlink: 11B1, 11B1.5, 11B2, and 11B3 remain nested commits until the complete reviewed oracle milestone.

#### 11B2 - Oracle contracts plus continuous launch and arc authority

Files:

- Create `path-planner/src/path_planner/v2/oracles/hopper.py`.
- Create `path-planner/tests/test_v2_hopper_oracle.py`.

Scope:

- Freeze exact frozen/slotted input/result carriers, reason universe, detail/evidence shape, canonical numeric helpers, six independent replay-work counters, deadline/profile/snapshot pre-audit, and no-later-call behavior.
- Implement the approved piecewise-constant closed-cell prism model, exact Euclidean XY disk overlap, full-vertical conservative product proxy, exact rational/Qsqrt comparisons, unbounded hard-cell columns, launch footprint, continuous interval coverage, candidate/work caps, stable semantic winner, deadline, and reseal checkpoints using only the A2-revised sealed 11B1 resource singleton.
- Every exact operation must enforce both the 262,144-bit pre-allocation limit and prospective 128-live-slot admission from its first GREEN; no temporary unbounded `Fraction`/big-int path is allowed.
- Oracle/candidate input carries the sealed profile and exact speed/elevation/azimuth indices, not raw duplicate direction or `dt_s`. It independently reconstructs A2 canonical action semantics and flight time; invokes only `_call_captured_ballistic_helper_v2`; requires exact `S=2`, final-time equality, normalized `(0,1)`, and direction/endpoint agreement before the one `[0,1]` interval; and treats any mismatch as global `hopper_numeric_contract_mismatch` with no later helper/action.
- The exact continuous parabola remains the clearance authority over the whole interval; two samples never authorize endpoint-only or 3D-line clearance. Private direct-core fine and `nextafter` cadence tests are partition-equivalence adversarial evidence only and must include a repair-sample sentinel and work-cap fail-closed coverage; they cannot enter production inputs or lineage.
- Keep ballistic build reservation `9_604_096`, exact-oracle reservation `36_200_448`, cap `100_000`, six counters, 262,144-bit ceiling, and 128-slot arena unchanged; only actual retained operational samples specialize to `96` bytes.
- Do not export the module, evaluate landing/stop, generate an action, search, or construct a public planning outcome.

Focused GREEN command runs `tests/test_v2_hopper_oracle.py`, `tests/test_v2_ballistics.py`, `tests/test_v2_hopper_authority.py`, `tests/test_v2_profiles.py`, `tests/test_v2_runtime.py`, `tests/test_v2_terrain.py`, and `tests/test_v2_geometry.py`. Fresh adversarial review must cover equality/nextafter boundaries, single-interval full-arc safety, private alternate partitions/repair sentinel, cardinal and near-cardinal directions, nonzero/large origins, grid edges/corners, unknown/OOB/hard cells, irrational roots, bit/work limits, forged helpers, and authority drift.

#### 11B3 - Landing, fixture model, pose, and stop authority

Files:

- Modify `path-planner/src/path_planner/v2/hopper_authority.py`.
- Modify `path-planner/tests/test_v2_hopper_authority.py`.
- Modify `path-planner/src/path_planner/v2/oracles/hopper.py`.
- Modify `path-planner/tests/test_v2_hopper_oracle.py`.

Scope:

- Add the approved Section 2 fixture-only registry/wrapper, exact parameter authority records/tokens and stop/energy evaluators to the already neutral authority module; implement per-hop unconditioned `0.99` prefix, full landing-cell-square plus footprint, same-height nominal recenter proxy, fixed yaw, touchdown/stop and relative-energy seams, exact reason precedence, and formal-ineligibility evidence.
- Default nullable Hopper profile remains unsupported. Test fixture success cannot become formal or physical capability evidence.
- Do not change `profiles.py` field shape and do not export the new modules yet.

Milestone:

- Run the cumulative 11B1-11B3 focused suite and fresh whole-oracle spec/quality/adversarial review.
- Only after review, integrate the reviewed nested HEAD into the parent with a gitlink-only commit.

#### 11B4 - Typed primitive and independent complete-route L2

Files:

- Create `path-planner/src/path_planner/v2/providers/hopper.py`.
- Create `path-planner/src/path_planner/v2/hopper_route_validation.py`.
- Create `path-planner/tests/test_v2_hopper_route_validation.py`.
- Modify `path-planner/tests/test_v2_hopper_oracle.py` only if a route replay boundary needs a new oracle regression.

Scope:

- Freeze Hopper state key, typed primitive, exact public/provenance fields, complete route carrier, route-state clamp, replay counters, cost/probability diagnostic, route digest, disabled cache evidence, success envelope, and independent full-route L2/reseal.
- Both provider-internal and API L2 consume the exact same A2-revised neutral resource singleton, specialized helper identity/ID/cap, 128-slot arena admission, and deterministic route/transient accounting. Each independently reconstructs speed/elevation/canonical azimuth/direction/flight time from sealed profile plus the primitive's three indices, accepts no carried `dt_s`, enforces exact two-sample replay, and validates the full continuous parabola before landing/stop.
- Freeze generic route-state identity `1+sum(S_i-1)`, operational `S_i=2`, child delta `1`, and exact `H+1` route replay states against the same subtraction-safe clamp. Separately freeze observation's endpoint-only `H+1` carrier and prove it does not authorize replay/safety. Keep worst-case helper/oracle reservations unchanged while using only `96` actual ballistic-retention bytes.
- This slice cannot claim complete L2 while any resource, cadence, direction, reseal, or reconstruction dependency is deferred.
- The large Hopper validator lives in its own module. `validation.py` is not changed in this slice.

Focused GREEN command:

```powershell
D:/conda_envs/lunar-explorer/python.exe -m pytest -o addopts='' -p no:cacheprovider `
  --basetemp "$tempRoot/basetemp" -q `
  tests/test_v2_hopper_route_validation.py `
  tests/test_v2_hopper_oracle.py `
  tests/test_v2_ballistics.py `
  tests/test_v2_contracts.py `
  tests/test_v2_serialization.py
```

After fresh review, integrate only the reviewed nested gitlink.

#### 11B5 - Deterministic provider search and resource authority

Files:

- Modify `path-planner/src/path_planner/v2/providers/hopper.py`.
- Create `path-planner/tests/test_v2_hopper_provider.py`.
- Modify `path-planner/tests/test_v2_hopper_route_validation.py` only for provider/L2 seam regressions.

Scope:

- Implement `plan()` capability preflight, exact fixture registry authority, fixed 192-action Dijkstra order, A2 index-derived azimuth/directions with exact cardinal `0/4/8/12`, no snap, deterministic tie-breaks, memory ledger and admission enforcement, per-hop work caps, operational `H+1` route-state/expansion/deadline caps, complete-route final L2, exact failure policy, and no partial success. The provider exposes the exact A2-revised singleton read-only; it may not create or substitute a second authority.
- Every action calls only the captured specialized ballistic path and carries no `dt_s`; all 192 actions remain available at nonzero/large origins. Local semantic rejection alone may advance to the next action; any true direction/representability/two-sample/cadence/numeric mismatch is a global abort and may not be skipped.
- Incomplete default profile must return typed unsupported capability before any Hopper ballistics/oracle/action/search call.

Focused GREEN command adds provider, search, runtime, cache, ballistics, authority, profiles, oracle, and route-L2 tests. Adversarial review must cover 192-action nonzero/large-origin direction reconstruction, generic-versus-specialized callable rebinding, mutable helper caps, memory phases, operational `H+1` versus observation authority, slot/bit/work limits, telemetry prefixes, deadline checkpoints, malformed outcomes, and repeated/hash-seed determinism.

After fresh review, integrate only the reviewed nested gitlink.

#### 11B6 - Explicit exports and API Hopper authority

Files:

- Create `path-planner/src/path_planner/v2/hopper_api.py`.
- Create `path-planner/tests/test_v2_hopper_api.py`.
- Modify `path-planner/src/path_planner/v2/oracles/__init__.py`.
- Modify `path-planner/src/path_planner/v2/providers/__init__.py`.
- Modify `path-planner/src/path_planner/v2/validation.py` only as a thin Hopper re-export.
- Modify `path-planner/src/path_planner/v2/__init__.py`.
- Modify `path-planner/src/path_planner/v2/api.py`.
- Modify `path-planner/tests/test_v2_api.py` only for the minimal dispatch/no-v1-fallback guards.
- Modify only these existing Hopper tests for explicit export/API seam regressions: `path-planner/tests/test_v2_hopper_authority.py`, `path-planner/tests/test_v2_hopper_oracle.py`, `path-planner/tests/test_v2_hopper_route_validation.py`, and `path-planner/tests/test_v2_hopper_provider.py`. No ballistics test or other Hopper test may change in 11B6; the new `path-planner/tests/test_v2_hopper_api.py` is listed separately above.

Scope:

- Preserve generic endpoint-safety precedence.
- Keep Hopper-specific token/audit/remap helpers in `hopper_api.py`; add explicit Hopper provider dispatch, composite authority sealing, provider-failure structural policy, independent claimed-success L2/reseal, exact outcome remapping, and postcondition deadline handling.
- Preserve wheel/legged behavior and v1 default surfaces. Never modify top-level `path_planner/__init__.py`, CLI, v1 adapter, PPO, executor, or canary code.

The public export delta is exact and may not grow during implementation:

- `oracles.__all__`: `HopperJumpCandidateV2`, `HopperValidationResultV2`, `validate_hopper_jump_l2`.
- `providers.__all__`: `HopperSearchStateV2`, `HopperJumpPrimitiveV2`, `HopperPrimitiveProviderV2`, `hopper_state_key_v2`, `nominal_hopper_search_state_v2`.
- Shared `validation.py` thin imports only: `HOPPER_ROUTE_VALIDATOR_ID_V2`, `HopperRouteProbabilityDiagnosticV2`, `validate_hopper_route_l2`.
- Top-level `path_planner.v2.__all__`: the preceding public symbols plus `HopperProviderAuthorityV2`, and no other Hopper symbol.

The following remain internal and must not enter any package `__all__`: `HopperParameterSetRecordV2`, `HopperResourceAuthorityV2`, `HopperRouteValidationResultV2`, `HopperSearchMemoryLedgerV2`, generic `sample_ballistic_arc_capped_v2`, private `_sample_hopper_ballistic_arc_capped_v2`, capped `landing_zone_cells_capped_v2`, fixture evaluators/factories, arena/ledger helpers, in-memory authority tokens, composite token builders, and everything in `hopper_api.py`. Exact export identity and absence tests are part of 11B6 RED.

Verification order:

1. Hopper API/provider/L2/oracle focused tests.
2. Existing wheel and legged API/provider/L2 focused tests.
3. Named v2 suite; expected prior baseline plus frozen Gate5B delta, with zero new skip.
4. Nested full suite once; expected `1641 + reviewed Gate5B delta` passed and the same 17 optional pydrake skips.
5. Fresh cross-platform spec/quality/adversarial review, then one gitlink-only parent integration.

#### 11B7 - Root Gate5 blocked-evidence snapshot

Files:

- Create `configs/xunce_path_v2_gate5_hopper_v1.json`.
- Modify `scripts/path_v2_gate_artifacts.py` only to support exact optional Gate-specific manifest metadata while preserving byte-equivalent existing calls.
- Modify `scripts/run_xunce_path_v2_gate_benchmark.py`.
- Modify `tests/test_xunce_path_v2_gate_benchmark.py`.
- Modify `tests/test_xunce_path_v2_g0_baseline_and_isolation.py` only for backward-compatibility and manifest-metadata RED guards.
- Modify `configs/stage_registry.json`.

Scope:

- Start only after parent gitlink equals the reviewed 11B6 nested HEAD and both tracked trees are clean.
- Freeze the six default Hopper proxy profile fields and `parameter_set_id` as exact null, with first blocker `freeze_hopper_simulation_proxy_profile_parameters`.
- Freeze `status=blocked`, `execution_class=blocked_profile_freeze`, `accepts_formal_inputs=false`, `formal_metrics_status=not_evaluated`, `formal_evidence_eligible=false`, `formal_row_count=0`, and all four release boundaries false.
- The shared atomic artifact helper's existing call behavior and Gate1-4 bytes remain unchanged. Gate5 passes exact manifest metadata for the frozen status/execution/blocker/eligibility/row-count/parameter-set values; missing, extra, conflicting, or caller-self-reported values fail closed.
- The output contains exactly eight regular artifacts: `config.json`, `summary.json`, `routing.json`, `results.jsonl`, `phase-state.jsonl`, `review.json`, `report.md`, and `manifest.json`. `summary.json`, `routing.json`, `review.json`, `manifest.json`, and `report.md` must agree on blocked status, `execution_class`, primary blocker, formal ineligibility, and zero formal rows; `results.jsonl` is empty and no fixture row may appear in a formal aggregate.
- Do not compute formal metrics, read formal data, publish a checkpoint, replace default policy, connect an executor, or start a canary.
- Developer tests use `tmp_path` and fake pytest. `scripts/run_stage.py --dry-run` is preview-only and writes nothing; the Gate benchmark runner's own `--dry-run` does write its exact artifact set and therefore may run only against a fresh non-formal directory under `D:/xunce/tmp/path_v2_g5/`. Neither path may call runner `main()` against `D:/xunce/out/path_v2/g5`.
- The checked-in Gate5 config freezes `temp_root="D:/xunce/tmp/path_v2_g5"`. For a real controller invocation, the runner verifies that base is on D, atomically derives a fresh `formal/<attempt-id>` child, freezes the resolved child in a trusted in-memory execution record before any optional subprocess, and later serializes it consistently into summary/review/manifest. A caller-supplied existing child, C-drive path, symlink/reparse escape, or path outside the base fails before output-root creation.
- Any runner-internal pytest subprocess uses nested cwd `D:/codex/worktrees/multiplatform-path-planner-v2/path-planner`, nested-only D-worktree `PYTHONPATH`, exact `PYTHONNOUSERSITE=1`, `PYTHONDONTWRITEBYTECODE=1`, `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, D-attempt `TEMP/TMP/MPLCONFIGDIR`, and arguments `-o addopts='' -p no:cacheprovider --basetemp <attempt>/basetemp --junitxml <attempt>/gate5.junit.xml`. The runner constructs and audits this environment rather than trusting inherited values; RED captures cwd/env/argv and rejects any C path or alternate import origin.
- If the profile-freeze first blocker short-circuits before pytest, the runner records the subprocess phase as `not_run_due_to_profile_freeze` and invokes no pytest helper. Every branch that does invoke pytest must satisfy the preceding envelope; fake-pytest tests cover both zero-call and exact-call cases.
- No Gate5 path may stat, enumerate, read, hash, parse, or otherwise probe `D:/xunce/inputs/path_v2/g5/` or any formal label/optimum/schedule dataset while `accepts_formal_inputs=false`. A filesystem spy RED proves the profile-freeze result is independent of formal-data presence or absence.
- Creating the real blocked snapshot under `D:/xunce/out/path_v2/g5` is a later controller-only execution step that requires separate explicit controller authorization after code review and root regression; it is not an automatic consequence of this slice or of implementation completion.

Root GREEN command:

```powershell
D:/conda_envs/lunar-explorer/python.exe -m pytest -o addopts='' -p no:cacheprovider `
  --basetemp "$tempRoot/basetemp" -q `
  tests/test_xunce_path_v2_gate_benchmark.py `
  tests/test_xunce_path_v2_g0_baseline_and_isolation.py
```

Expected: prior `322` root tests plus the frozen 11B7 delta pass; no failure/error/skip.

### Executable PPO Stage1 audit

Run from the parent repository root with the root/PPO isolation envelope above and a new `D:/xunce/tmp/path_v2_g5/ppo/<fresh-id>` temp root:

```powershell
$ppoRoot = "D:/xunce/tmp/path_v2_g5/ppo/<fresh-id>"
if (Test-Path -LiteralPath $ppoRoot) { throw "PPO temp root must be fresh" }
New-Item -ItemType Directory -Path "$ppoRoot/mpl" -Force | Out-Null
$env:TEMP = $ppoRoot
$env:TMP = $ppoRoot
$env:MPLCONFIGDIR = "$ppoRoot/mpl"
$env:PYTHONPATH = "$repoRoot/src$([IO.Path]::PathSeparator)$nestedRoot/src"

D:/conda_envs/lunar-explorer/python.exe -m pytest -o addopts='' `
  -p no:cacheprovider -q `
  tests/ppo_highres_frontier/test_stage1_smoke_env.py `
  --junitxml "$ppoRoot/stage1.junit.xml" `
  --basetemp "$ppoRoot/basetemp"
$ppoExit = $LASTEXITCODE
if ($ppoExit -notin @(0, 1)) { throw "unexpected PPO pytest exit code" }

@'
import json
from pathlib import Path

from scripts.run_xunce_path_v2_g0_baseline_and_isolation import parse_junit

config = json.loads(
    Path("configs/xunce_path_v2_g0_baseline_and_isolation_v1.json").read_text(
        encoding="utf-8"
    )
)
expected = {row["nodeid"] for row in config["ppo_stage1"]["known_failures"]}
observed = parse_junit(Path("D:/xunce/tmp/path_v2_g5/ppo/<fresh-id>/stage1.junit.xml"))
assert observed.tests == 56
assert observed.passed >= 43
assert observed.failures <= 13
assert (observed.errors, observed.skipped) == (0, 0)
assert len(expected) == 13
actual = set(observed.failed_nodeids)
assert actual <= expected
# This branch's checked-in Gate0 snapshot currently also requires exact identity.
# A strict subset is a baseline improvement, not a v2 regression, but it needs a
# separately approved Gate0 baseline refresh before the existing runner can pass.
assert actual == expected
assert observed.passed == 43
print("PPO Stage1 exact inherited failure set verified")
'@ | D:/conda_envs/lunar-explorer/python.exe -
```

Pytest exit code `1` is expected and acceptable only when this JUnit audit proves no failure outside the frozen allowlist. The goal-level non-regression rule is `passed>=43`, `failures<=13`, actual nodeids a subset, and zero errors/skips. The current checked-in Gate0 runner is deliberately stricter and also requires exact `43/13` identity; if inherited failures shrink, record that as external baseline improvement and obtain a separate Gate0 baseline-refresh approval rather than misclassifying it as a v2 regression or silently editing the allowlist. Console totals alone are not evidence.

### Final Gate5B integration regression

After all seven main slices, the inserted 11B1.5 slice, and every required review, but before any formal output write:

1. Prove parent/nested clean and gitlink equality; audit exact changed-file allowlists and `git diff --check`.
2. Re-run cumulative focused, named v2, nested full, and root Gate0/gate-runner suites using D-drive temp roots.
3. Re-run PPO Stage1 and prove the result is exactly `43 passed` plus the same frozen 13 failed nodeids, with zero errors/skips.
4. Prove v1 default imports/CLI/adapter/policy tests remain unchanged and v2 is still explicit opt-in.
5. Prove config/artifacts encode `publishes_checkpoint=false`, `replaces_default_policy=false`, `connects_real_executor=false`, and `starts_online_canary=false`.
6. Run `scripts/run_stage.py --dry-run` as a no-write registry preview, then run the benchmark runner's artifact-writing `--dry-run` only against a fresh non-formal `D:/xunce/tmp/path_v2_g5/dry_run/<fresh-id>` root and audit its exact artifact set.
7. Before formal execution, prove both exact formal root `D:/xunce/out/path_v2/g5` and the selected formal temp root `D:/xunce/tmp/path_v2_g5/formal/<fresh-id>` do not exist. Seal parent/nested HEAD, gitlink, config, runner, and approved static contract-file hashes before root creation and verify them again after execution. Do not stat/read/hash/parse any formal dataset as part of this seal.
8. Only then, and only after separate explicit controller authorization, may the controller run the Gate5 runner against the fresh exact formal root, using the fresh formal temp root, to produce the blocked snapshot. Cross-check the five human/machine artifacts listed above, exact null profile/parameter-set fields, zero formal rows, formal ineligibility, and all four false release boundaries. A blocked Gate5 is not a passed Gate5 and does not authorize Gate6 release claims.

### Forbidden substitutions

- Do not modify `contracts.py`, `profiles.py`, `providers/base.py`, `cache.py`, wheel/legged production or tests, top-level `path_planner/__init__.py`, CLI, v1 adapters, PPO, executor, canary, default-policy, or checkpoint code unless a new RED proves an unavoidable contract change and the plan/design is independently re-approved first.
- No discrete sample-only clearance, L-infinity final authority, bilinear terrain interpolation, bounded hard-obstacle height, implicit snap, or hidden touchdown-to-mean transition.
- No endpoint-only or 3D straight-line substitution for the full continuous parabola; exact two-sample cadence changes partition count only, never safety authority.
- No production `dt_s`, alternate cadence, direction-vector input, generic-ballistic fallback, cardinal epsilon snap, or post-hoc endpoint rewrite. Any future cadence/direction semantic change requires a version bump and new approval.
- No unapproved fixture value may become a default or formal/physical capability claim.
- No helper result, cached evidence, provider outcome, or route primitive may authorize its own safety without independent reseal/L2.
- No test count, pass claim, or formal artifact may be inferred from a narrow focused run.
- No broad cleanup, history rewrite, batch deletion, C-drive worktree access, or unrelated refactor is part of Gate5B.
