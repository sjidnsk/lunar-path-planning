# Midterm G2 Independent Truth Producer Design Addendum

## Decision

Reduced midterm G2 uses a two-stage evidence boundary:

```text
isolated truth producer
  -> immutable content-addressed truth bundle
  -> Task 7 one-way adapter/readiness
  -> Task 8 provider execution in a separate output root
```

The truth producer is a separate reviewed source package and process. It does
not import this repository, `path_planner`, any provider/oracle/L2 validator, or
Task 7. It uses its own integer fixed-point geometry, analytic predicates,
finite graphs, reason namespace, canonical encoder, and certificate verifier.
Provider execution starts only after `truth-freeze.json` is published and may
never write back to the truth bundle.

Project authorization is:
`720e11ef04ad2b57283421809a077ccf1f0b35167a9f482082241398ad0214d2`.
It authorizes project-internal technical independence and generation, but no
artifact is eligible until its exact IDs, source bundle, hashes, isolation
audits, raw retention, reproduction audit, and independent technical review are
bound by a separate immutable approval.

## Exact scale

The candidate truth bundle contains:

- `3334` primitive labels for wheel;
- `3334` primitive labels for legged;
- `3334` primitive labels for Hopper;
- exactly one independently solved small-map optimum per platform;
- exactly `43` unique requests per platform:
  - Standard: `23 reachable + 7 hard_reachable + 3 unreachable`;
  - Kilometer: `6 reachable + 2 hard_reachable + 2 unreachable`.

That is `10002` labels, `3` optima, and `129` unique requests. Five repeats are
created later by Task 8 and may change only submission order and repeat index.
Warm-up, cold-start, and worker-1 diagnostics remain non-formal.

## Process and import isolation

Production generation runs from a frozen D-drive source snapshot with an empty
project `PYTHONPATH`. The process records `sys.path`, loaded module origins,
Python/dependency identities, source snapshot SHA-256, command, working
directory, and environment allowlist. Any loaded module or path containing
`path_planner`, `lunar_exploration_ppo`, or `xunce_mid_dual_g2_inputs` blocks
publication.

Static audit rejects forbidden imports, dynamic import, `exec`, pickle,
subprocess calls into provider code, and unexplained high-similarity code
fragments. Case generation and truth evaluation are separate phases:

- generator rows contain no expected verdict, reachability, provider outcome,
  or target reason;
- the oracle consumes only frozen case bytes, parameter/profile records, and
  producer specification;
- provider outcome keys are forbidden in every truth-source row.

## Primitive truth

All equality and uncertainty rules are frozen:

- distance epsilon `0.001 m`;
- slope epsilon `0.01 deg`;
- height epsilon `0.001 m`;
- probability epsilon `0.000001`;
- upper limits pass on equality;
- closed obstacle contact fails;
- support and landing-probability lower limits pass on equality;
- an interval that cannot prove a side becomes `G2I_NUMERIC_UNDECIDED` and may
  not count toward a quota.

Wheel uses closed-form SE(2), closed swept-footprint/obstacle geometry, analytic
plane slope, and `G2I_W_*` reasons. Its exact strata are:
`700 nominal`, `450 slope`, `450 obstacle`, `300 knownness/boundary`,
`630 kinematic`, `404 anti-alias`, and `400 compound`.

Legged uses integer foothold/body polygons, analytic planes, step length/height,
support-hull margin, body sweep, a frozen phase automaton, and `G2I_L_*`
reasons. Its strata are:
`650 nominal`, `480 foothold`, `360 step length`, `360 step height`,
`480 support`, `420 body sweep`, `284 sequence/grid`, and `300 compound`.

Hopper consumes only the new complete candidate parameter record. It uses
independent no-drag lunar ballistic equations, analytic arc/cell intersection,
independent Gaussian landing integration, closed landing-disk geometry, and
independent stop/energy evaluation with `G2I_H_*` reasons. Its strata are:
`384 action lattice`, `516 nominal`, `360 launch`, `600 arc`,
`300 unknown/boundary`, `300 landing probability`,
`550 landing footprint/slope/height`, and `324 stop/model`.

Each platform must have at least 64 terrain hashes and a safe ratio from 35% to
65%. Wheel covers at least 72 heading bins and seven control families; every
leg ID occurs at least 700 times; every Hopper action tuple occurs at least
twice. IDs and full case hashes are globally unique.

## Optima and request truth

Each platform small map enumerates its complete finite node/action envelope.
The primary solver is heap-based UCS/Dijkstra. A structurally different
quadratic Dijkstra plus bounded exhaustive path enumeration independently
recomputes the optimum. The row retains the full edge list, distance labels,
predecessor tree, settled order, completeness counts, Bellman checks, and
certificate hashes; a scalar constant without reconstructable evidence is
invalid.

The request generator freezes terrain families, seeds, pool cardinalities,
hardness rules, and selection hashing before generation. Truth selection is
provider-blind. Reachability is a complete finite-graph search:

- reachable requires a settled goal and full path/edge certificate;
- unreachable requires exhausted open set, reachable-set hash, and a frontier
  cut with independent rejection reasons;
- hard reachable is defined only from detour/path length and independent safety
  slack, never provider outcome or timing.

Standard terrain is procedural simulation proxy. Kilometer terrain binds the
existing LOLA 20 m macro source bytes while any sub-20 m terrain is separately
identified as `synthetic_terrain_obstacle_proxy/v1`; it is never described as
measured or `physical_obstacle_cells`.

## Canonical bundle and retention

Truth source:

```text
D:/xunce/inputs/mid_dual/g2-truth/<candidate-id>/
  truth-freeze.json
  primitive-labels.jsonl
  small-map-optima.jsonl
  requests.jsonl
  certificates/
  terrain/
  source/
  raw/
  audits/
  reject-ledger.jsonl
  shard-manifests/
```

JSON/JSONL is UTF-8, LF, sorted-key canonical JSON without NaN, infinity,
negative zero, or duplicate keys. Terrain NPZ is fixed-order, ZIP_STORED,
fixed-metadata and pickle-free. Hashes are lowercase SHA-256 over
domain-separated, length-prefixed bytes. `input_set_id` and payload root derive
only from current bytes.

The bundle retains the producer specification/source/lock/licenses, parameter
records, LOLA bytes, generator pool, rejected cases and reasons, raw oracle
rows, full graphs, certificates, selection audit, import/static audits, and two
fresh-run byte comparisons. Publishing is data-first and manifest-last; resume
accepts only a contiguous hash-valid shard prefix with identical inputs.

Candidate attestations remain:

```text
technical_independence = T2_candidate
organizational_independence = project_internal
formal_evidence_eligible = false
```

Task 7 creates a separate execution bundle and one-to-one
`truth_request_sha256 -> provider_request_sha256` crosswalk. Only after a fresh
independent reviewer binds the exact source, implementation, payload,
reproduction, Hopper candidate, and authorization hashes may a separate
approved attestation be created. No candidate file is edited in place.
