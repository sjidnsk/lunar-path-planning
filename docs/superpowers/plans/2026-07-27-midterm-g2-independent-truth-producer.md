# Midterm G2 Independent Truth Producer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone deterministic truth producer that can generate and audit the exact reduced-G2 candidate evidence bundle without importing or observing Path Planner providers.

**Architecture:** A self-contained source package separates case generation, independent platform oracles, complete finite-graph solving, canonical packaging, and audit/reproduction. Production runs from a frozen D-drive copy with a cleared project import path; Task 7 consumes only the immutable published bundle.

**Tech Stack:** Python 3.11 standard library, NumPy only for canonical array containers, integer fixed-point geometry, Decimal/rational cost, SHA-256, deterministic ZIP/NPZ, pytest.

## Global Constraints

- Root seed is exactly `20260727`; every child seed is SHA-256 domain-derived and never uses Python `hash()`.
- Scale is exactly `3334 × 3` labels, `1 × 3` optima, and `43 × 3` requests.
- The producer must not import or invoke `path_planner`, `lunar_exploration_ppo`, Task 7, provider, oracle, L2, or timing code.
- Generator rows never contain truth; provider outcome/timing keys are forbidden in truth artifacts.
- Candidate Hopper input must be complete, generic, simulation-proxy-only, and formally ineligible.
- All raw rows, rejects, graphs, source snapshots, audits, certificates, and reproduction evidence are retained on D.
- Candidate output remains formally ineligible until a separate exact-artifact review and approval.

## File Map

Create under `independent/g2_t2_producer/`:

- `SPECIFICATION.json` — exact quotas, boundaries, reason precedence, terrain families, request mix, seeds, and canonical schemas.
- `producer-lock.json` — Python/dependency/encoding contract.
- `LICENSES.json` — standard-library/NumPy and retained-source license index.
- `run_producer.py` — phase CLI and process-isolation preflight.
- `producer/canonical.py` — canonical JSON/JSONL/hash/NPZ and content index.
- `producer/models.py` — neutral fixed-point cases/rows with forbidden-key validation.
- `producer/generate_cases.py` — truth-blind stratified cases.
- `producer/oracle_wheel.py` — independent wheel predicates.
- `producer/oracle_legged.py` — independent legged predicates.
- `producer/oracle_hopper.py` — independent ballistic/landing/stop/energy predicates.
- `producer/finite_graph.py` — complete graph enumeration, two solvers, Bellman/certificate verification.
- `producer/generate_requests.py` — provider-blind terrain pools, reachability, hardness, and selection.
- `producer/package_bundle.py` — shard merge, atomic truth freeze, source/raw retention.
- `producer/audit_bundle.py` — import/static/cardinality/join/determinism/reproduction audits.
- `producer/__init__.py` — package version only.
- `tests/test_counterexamples.py`
- `tests/test_boundaries.py`
- `tests/test_repeatability.py`
- `tests/test_full_candidate_bundle.py`

---

## Task P1: Canonical contracts and isolation

**Files:** `SPECIFICATION.json`, `producer-lock.json`, `LICENSES.json`,
`run_producer.py`, `producer/canonical.py`, `producer/models.py`,
`tests/test_boundaries.py`, `tests/test_repeatability.py`

- [ ] Write failing tests for exact quotas/schema, domain-separated hashes,
  canonical JSON/JSONL, fixed-metadata NPZ, no invalid floats, unique IDs,
  forbidden truth/provider keys, seed derivation, and project-module import
  rejection.
- [ ] Run those tests and verify RED.
- [ ] Implement only the frozen canonical/model/isolation primitives.
- [ ] Run tests twice under distinct `PYTHONHASHSEED` values and require
  byte-identical fixtures.

## Task P2: Truth-blind cases and independent primitive oracles

**Files:** `producer/generate_cases.py`, `producer/oracle_wheel.py`,
`producer/oracle_legged.py`, `producer/oracle_hopper.py`,
`tests/test_counterexamples.py`, `tests/test_boundaries.py`

- [ ] Add RED tests for every equality boundary, one-ULP/epsilon side,
  closed-contact rule, numeric-undecided route, reason precedence, and the exact
  wheel/legged/Hopper stratum cardinalities.
- [ ] Implement generator rows without truth keys and the independent fixed-point
  oracles described in the design addendum.
- [ ] Verify all `G2I_*` reason namespaces, safe-ratio/diversity rules, exact
  `3334` per platform, global case/hash uniqueness, and full Hopper 192-action
  coverage without importing provider constants.

## Task P3: Complete optima and request truth

**Files:** `producer/finite_graph.py`, `producer/generate_requests.py`,
`tests/test_counterexamples.py`, `tests/test_full_candidate_bundle.py`

- [ ] Add RED tests for complete node/action enumeration, nonnegative edge cost,
  two-solver agreement, Bellman certificate, unreachable frontier cut,
  provider-blind hard classification, exact platform/scale/outcome counts, and
  selection-seed immutability.
- [ ] Implement the three reconstructable small maps and both solver paths.
- [ ] Implement the 256 Standard + 96 Kilometer candidate terrain pool,
  independent reachability/certificates, frozen hardness rules, and hash-ranked
  exact selection.
- [ ] Verify exactly three optima, 129 globally unique requests, 114 reachable,
  15 unreachable, and no provider/timing outcome in source rows.

## Task P4: Recoverable packaging and full audits

**Files:** `producer/package_bundle.py`, `producer/audit_bundle.py`,
`run_producer.py`, `tests/test_repeatability.py`,
`tests/test_full_candidate_bundle.py`

- [ ] Add RED tests for data-first/manifest-last publication, contiguous-prefix
  resume, input drift, missing raw source, hash/file-name mismatch, duplicate or
  dangling join, partial quota, source mutation, loaded-module contamination,
  candidate formal-eligibility injection, and two-run byte mismatch.
- [ ] Implement phase CLI:
  `preflight`, `cases`, `labels`, `optima`, `requests`, `package`, `audit`,
  `reproduce`, and `all`.
- [ ] Retain all source/raw/reject/shard/certificate evidence and publish a
  content-addressed `truth-freeze.json` only after every audit passes.
- [ ] Ensure a fresh second root reproduces the same payload root and that no
  production process imported a project module.

## Task P5: Verification and candidate handoff

- [ ] Run the full producer tests:

  ```powershell
  & 'D:/conda_envs/lunar-explorer/python.exe' -m pytest `
    independent/g2_t2_producer/tests -vv -p no:cacheprovider `
    --basetemp 'D:/xunce/tmp/pytest-mid-dual/g2-producer'
  ```

- [ ] Copy the exact source package to a new content-addressed D-drive source
  root; clear project `PYTHONPATH`; run preflight and a small fixture smoke.
- [ ] Verify the exact allowed-file diff, UTF-8 decoding, source implementation
  hash, forbidden imports, no formal G2 calls, and no modification of any
  existing D-drive formal root.
- [ ] Commit only the independent producer package after the controller grants
  a serialized commit lease.

## Self-Review Checklist

- [ ] Truth generation cannot observe provider results.
- [ ] Case generation and oracle evaluation are distinct artifacts.
- [ ] Exact quotas and diversity checks cannot be met by row duplication.
- [ ] All optima and reachability labels have reconstructable certificates.
- [ ] Raw sources and rejects are retained and hash-bound.
- [ ] Kilometer microterrain is explicitly synthetic proxy, never physical truth.
- [ ] Candidate attestations remain formally ineligible.
- [ ] A separate independent artifact review is required before Task 7 approval.
