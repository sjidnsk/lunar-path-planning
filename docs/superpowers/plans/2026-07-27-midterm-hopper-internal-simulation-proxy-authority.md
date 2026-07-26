# Midterm Hopper Internal Simulation Proxy Authority Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fail-closed, opt-in implementation authority for the frozen generic internal Hopper computational simulation proxy required by reduced midterm G2/G3.

**Architecture:** Extend the path-planner authority registry with one additive implementation record and two exact evaluators while keeping formal-evidence approval outside the submodule. Provider, jump L2, route L2, and API seals resolve and bind the same record; Task 7 later binds candidate, approval, independent input, and artifact hashes before formal eligibility.

**Tech Stack:** Python 3.11, dataclasses, exact binary64 comparison, SHA-256 lineage, pytest, existing Path Planner v2 Hopper authority and L2 APIs.

## Global Constraints

- Authorization record SHA-256 is exactly `720e11ef04ad2b57283421809a077ccf1f0b35167a9f482082241398ad0214d2`.
- Parameter-set ID is exactly `hopper_generic_internal_computational_simulation_proxy_midterm_g2g3/v1`.
- New values are `0.375`, `0.750`, `0.125`, `0.625`, stop `speed<=2.5`, and energy `(speed/2.5)**2`.
- Existing action generation remains 192 actions and the `3.0 m/s` actions are not removed.
- Existing Gate5B fixture and default configuration remain byte-for-byte semantically unchanged and formally ineligible.
- `max_traversable_slope_deg=30.0`, `risk_weight=0.0`, landing mass `>=0.99`, and all current terrain/safety semantics remain unchanged.
- The submodule implementation record never grants formal eligibility.
- No hardware claim, checkpoint publication, default-policy replacement, executor connection, or canary start.

## File Map

### Modify

- `path-planner/src/path_planner/v2/profiles.py`
  - permit the explicit new capability revision while keeping exact Hopper
    profile validation and existing default revision.
- `path-planner/src/path_planner/v2/hopper_authority.py`
  - add the implementation record, exact stop/energy evaluators, strict registry,
    record matching, in-memory identity seal, and deterministic lineage.
- `path-planner/src/path_planner/v2/providers/hopper.py`
  - resolve the supported record and apply its exact evaluators without changing
    action enumeration or search.
- `path-planner/src/path_planner/v2/oracles/hopper.py`
  - use the same resolved record and exact evaluator authority for jump L2.
- `path-planner/src/path_planner/v2/hopper_route_validation.py`
  - replace fixture-only resolution with supported-record resolution and exact
    profile/record/evaluator matching.
- `path-planner/src/path_planner/v2/hopper_api.py`
  - bind the extended registry and evaluator identities in every existing seal.

### Create

- `path-planner/tests/test_v2_hopper_internal_sim_proxy_authority.py`
  - exact numeric, evaluator, behavior, fixture isolation, tamper, lineage, and
    API/L2 regression tests.

---

## Task H1: Add the generic internal Hopper implementation authority

**Files:**

- Modify: `path-planner/src/path_planner/v2/profiles.py`
- Modify: `path-planner/src/path_planner/v2/hopper_authority.py`
- Modify: `path-planner/src/path_planner/v2/providers/hopper.py`
- Modify: `path-planner/src/path_planner/v2/oracles/hopper.py`
- Modify: `path-planner/src/path_planner/v2/hopper_route_validation.py`
- Modify: `path-planner/src/path_planner/v2/hopper_api.py`
- Create: `path-planner/tests/test_v2_hopper_internal_sim_proxy_authority.py`

**Interfaces:**

- Consumes: existing `HopperProfileV2`, `HopperProviderAuthorityV2`,
  `HopperParameterSetRecordV2`, Hopper provider, jump L2, route L2, and API seal.
- Produces: one frozen implementation record retrievable by exact parameter-set
  ID; exact stop and energy evaluators; stable in-memory and deterministic
  lineage tokens. It does not produce a formally approved evidence record.

- [ ] **Step 1: Write exact record and fixture-isolation tests**

  Assert exact IDs, binary64 `.hex()` words, type/finite/positive checks,
  `launch_height >= body_radius + arc_margin`, strict sorted registry, no
  duplicates, unchanged Gate5B tuple/token, and unknown ID fail-closed behavior.

- [ ] **Step 2: Run the authority tests and verify RED**

  Run:

  ```powershell
  & 'D:/conda_envs/lunar-explorer/python.exe' -m pytest `
    tests/test_v2_hopper_internal_sim_proxy_authority.py -vv `
    -p no:cacheprovider `
    --basetemp 'D:/xunce/tmp/pytest-mid-dual/hopper-authority-red'
  ```

  Expected: collection or assertions fail because the new record and evaluators
  do not yet exist.

- [ ] **Step 3: Add the implementation record and exact evaluators**

  In `hopper_authority.py`, retain the current fixture object unchanged and add
  the frozen generic implementation record. The stop evaluator accepts exact
  finite positive `float` values at or below `2.5`; the energy evaluator returns
  the exact result of `(speed / 2.5) ** 2`. Reject `bool`, subclasses, NaN,
  infinity, and nonpositive speed. Record evaluator IDs and source SHA-256 in
  deterministic lineage; record evaluator object identities only in the
  in-memory seal.

- [ ] **Step 4: Add profile and shared exact-match support**

  In `profiles.py`, allow the new explicit capability revision in addition to
  the existing revision. Do not alter the default profile. Centralize the
  profile/record comparison so provider, jump L2, and route L2 compare exact
  IDs and binary64 words for base profile, capability revision, four numeric
  fields, stop condition, energy model, and parameter-set ID.

- [ ] **Step 5: Wire provider, jump L2, route L2, and API seals**

  Resolve only an exact registry record; invoke that record's trusted evaluators;
  preserve the 192-action generation order; ensure `3.0 m/s` fails at stop
  validation rather than disappearing. Keep provider and both L2 validators'
  independent recomputation and reseal checks. Upgrade only an authority/lineage
  schema ID whose serialized shape changes; do not rename unchanged safety
  validator semantics.

- [ ] **Step 6: Add behavioral and tamper tests**

  Cover `1.5/2.0/2.5` stop pass, `3.0` and `nextafter(2.5,+inf)` stop fail,
  frozen energy words, 192 actions, unchanged landing/slope/same-height/risk
  behavior, provider/L2 agreement, fixture isolation, one-ULP rejection, and
  post-construction evaluator/registry/profile/provider/L2 tamper failure.
  Verify deterministic lineage under at least two `PYTHONHASHSEED` values and
  absence of object addresses or `repr()` text.

- [ ] **Step 7: Run focused and full submodule verification**

  Run:

  ```powershell
  & 'D:/conda_envs/lunar-explorer/python.exe' -m pytest `
    tests/test_v2_hopper_internal_sim_proxy_authority.py `
    tests/test_platform_profile.py `
    tests/test_v2_formal_request_codec.py `
    tests/test_v2_midterm_timing.py -vv `
    -p no:cacheprovider `
    --basetemp 'D:/xunce/tmp/pytest-mid-dual/hopper-authority-focused'

  & 'D:/conda_envs/lunar-explorer/python.exe' -m pytest `
    -p no:cacheprovider `
    --basetemp 'D:/xunce/tmp/pytest-mid-dual/hopper-authority-full'
  ```

  Expected: all focused and full submodule tests pass with the existing fixture,
  profile defaults, action order, provider semantics, and L2 safety behavior
  unchanged.

- [ ] **Step 8: Review exact diff and commit**

  Confirm only the seven Task H1 files changed. Confirm no root Task 7 artifact,
  formal input, or formal output was generated. Commit the seven submodule files,
  then commit only the root `path-planner` gitlink after the controller grants
  the serialized commit lease.

## Self-Review Checklist

- [ ] The implementation record cannot set or infer formal evidence eligibility.
- [ ] Gate5B fixture/default behavior and tokens remain unchanged.
- [ ] The new capability is opt-in and generic, with no real hardware name.
- [ ] All six parameter semantics are exact and independently tested.
- [ ] Provider, jump L2, route L2, and API seal bind the same record/evaluators.
- [ ] Action generation remains 192 and safety contracts remain unchanged.
- [ ] One-ULP, identity, registry, profile, and lineage drift fail closed.
- [ ] Task 7 remains responsible for candidate, independent review/input, approval, and approved-record hashes.
