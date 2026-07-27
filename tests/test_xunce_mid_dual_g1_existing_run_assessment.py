from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts/xunce_mid_dual_g1_existing_run_assessment.py"
RUN_ID = "g1-formal-20260727-0650-cst"
SCALE_PROFILE = "midterm_reduced_w8x3_update80/v1"
RUNNER_ID = "run_xunce_mid_dual_g1_coverage/v1"
CHECKPOINT_SHA256 = (
    "35e04c86f9f973af028fb08f0175d42ab45378d09e1f2b96ee6aad6e4c12b5b5"
)
POLICY_STATE_SHA256 = (
    "3123e6adde99be41e3bd5cc2f3843068e5416892395926d759c2c6a243ffd381"
)
DENOMINATOR_SOURCE = "reachable_observable_free_highres_cells/v1"
DENOMINATOR_ALGORITHM = "exact_reachable_safe_pose_range_los/v1"
AUTHORITY_PATHS = (
    ".superpowers/sdd/2026-07-26-midterm-dual-gate-experiment/"
    "g1-existing-run-assessment-use-ruling.md",
    ".superpowers/sdd/2026-07-26-midterm-dual-gate-experiment/"
    "g1-lineage-salvage-review.md",
    ".superpowers/sdd/2026-07-26-midterm-dual-gate-experiment/"
    "task-11-consolidated-repair-brief.md",
)

_MODULE: Any | None = None


def _load_module() -> Any:
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    if not SCRIPT_PATH.is_file():
        pytest.fail("assessment module is not implemented")
    spec = importlib.util.spec_from_file_location(
        "xunce_mid_dual_g1_existing_run_assessment_under_test",
        SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _MODULE = module
    return module


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return _sha256(_canonical_bytes(value))


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def _jsonl_bytes(rows: list[dict[str, object]]) -> bytes:
    return b"".join(_canonical_bytes(row) + b"\n" for row in rows)


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _canon(path: Path) -> str:
    return path.resolve().as_posix()


def _scenario_id(split: str, index: int) -> str:
    prefix = "test" if split == "test_q24" else "unseen"
    return f"{prefix}/scenario-{index:04d}/standard-proxy/v1"


def _episode_id(split: str, index: int) -> str:
    return f"{split}-episode-{index:02d}"


def _make_proof(scenario_id: str, index: int, slope: float) -> dict[str, object]:
    return {
        "scenario_id": scenario_id,
        "coverage_denominator_source": DENOMINATOR_SOURCE,
        "coverage_denominator_algorithm": DENOMINATOR_ALGORITHM,
        "exact": True,
        "coverable_mask_sha256": _sha256(
            f"denominator:{scenario_id}".encode("utf-8")
        ),
        "coverable_cell_count": 1000 + index,
        "key": {"max_slope_deg": slope},
    }


def _reconstructed_input_audit(
    *,
    manifest_path: Path,
    manifest: dict[str, object],
    manifest_sha256: str,
) -> dict[str, object]:
    cohorts = manifest["cohorts"]
    assert isinstance(cohorts, dict)
    proofs_list = manifest["denominator_proofs"]
    assert isinstance(proofs_list, list)
    proofs = {str(item["scenario_id"]): item for item in proofs_list}
    jobs: list[dict[str, object]] = []
    for split in ("test_q24", "unseen24"):
        scenario_ids = cohorts[split]
        assert isinstance(scenario_ids, list)
        for index, scenario_id in enumerate(scenario_ids):
            proof = proofs[str(scenario_id)]
            jobs.append(
                {
                    "split": split,
                    "scenario_id": scenario_id,
                    "episode_id": _episode_id(split, index),
                    "episode_index": index,
                    "lane_id": f"lane-{index % 8}",
                    "denominator_sha256": proof["coverable_mask_sha256"],
                    "denominator_cell_count": proof["coverable_cell_count"],
                    "denominator_source": DENOMINATOR_SOURCE,
                    "denominator_algorithm": DENOMINATOR_ALGORITHM,
                }
            )
    return {
        "schema_version": "xunce-mid-dual-g1-input-audit/v1",
        "gate_id": "g1",
        "runner_id": RUNNER_ID,
        "scale_profile": SCALE_PROFILE,
        "status": "verified",
        "scenario_manifest_path": _canon(manifest_path),
        "scenario_manifest_schema_version": "mid-dual-scenario-freeze/v1",
        "scenario_manifest_sha256": manifest_sha256,
        "scenario_manifest_canonical_sha256": _canonical_sha256(manifest),
        "policy_blind_attestation_sha256": _canonical_sha256(
            manifest["policy_blind_attestation"]
        ),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "policy_state_sha256": POLICY_STATE_SHA256,
        "denominator_source": DENOMINATOR_SOURCE,
        "denominator_algorithm": DENOMINATOR_ALGORITHM,
        "formal_split_order": ["test_q24", "unseen24"],
        "formal_jobs": jobs,
        "test_c24_scenario_ids": list(cohorts["test_c24"]),
        "validation3_scenario_ids": list(cohorts["validation3"]),
        "replay3_scenario_ids": list(cohorts["replay3"]),
    }


def _common_row(
    *,
    split: str,
    phase_id: str,
    index: int,
    scenario_id: str,
    config_sha256: str,
    input_sha256: str,
    code_sha256: str,
    manifest_sha256: str,
) -> dict[str, object]:
    return {
        "gate_id": "g1",
        "runner_id": RUNNER_ID,
        "phase_id": phase_id,
        "phase_name": split,
        "scale_profile": SCALE_PROFILE,
        "run_id": RUN_ID,
        "split": split,
        "episode_id": _episode_id(split, index),
        "episode_index": index,
        "scenario_id": scenario_id,
        "lane_id": f"lane-{index % 8}",
        "source_sha256": code_sha256,
        "config_sha256": config_sha256,
        "input_sha256": input_sha256,
        "code_sha256": code_sha256,
        "scenario_manifest_sha256": manifest_sha256,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "policy_state_sha256": POLICY_STATE_SHA256,
    }


@dataclass(frozen=True)
class ExistingRunFixture:
    repository_root: Path
    source_root: Path
    preservation_root: Path
    scenario_manifest_path: Path
    stop_evidence_path: Path
    review_root: Path
    envelope_root: Path
    source_hashes: dict[str, str]
    authority_hashes: dict[str, str]
    preservation_metadata_sha256: str
    preservation_artifact_hashes: dict[str, str]
    scenario_manifest_sha256: str
    input_sha256: str
    config_semantic_sha256: str
    phase_state_sha256: str

    def install_scope(self, module: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(module, "REPOSITORY_ROOT", self.repository_root)
        monkeypatch.setattr(
            module,
            "AUTHORIZED_SOURCE_ROOT",
            _canon(self.source_root),
        )
        monkeypatch.setattr(
            module,
            "AUTHORIZED_PRESERVATION_ROOT",
            _canon(self.preservation_root),
        )
        monkeypatch.setattr(
            module,
            "AUTHORIZED_REVIEW_BASE_ROOT",
            _canon(self.review_root.parent),
        )
        monkeypatch.setattr(
            module,
            "AUTHORIZED_ENVELOPE_BASE_ROOT",
            _canon(self.envelope_root.parent),
        )
        monkeypatch.setattr(
            module,
            "AUTHORIZED_SOURCE_FILE_SHA256",
            dict(self.source_hashes),
        )
        monkeypatch.setattr(
            module,
            "AUTHORITY_ALLOWLIST",
            dict(self.authority_hashes),
        )
        monkeypatch.setattr(
            module,
            "AUTHORIZED_PRESERVATION_METADATA_SHA256",
            self.preservation_metadata_sha256,
        )
        monkeypatch.setattr(
            module,
            "AUTHORIZED_PRESERVATION_ARTIFACT_SHA256",
            dict(self.preservation_artifact_hashes),
        )
        monkeypatch.setattr(
            module,
            "AUTHORIZED_SCENARIO_MANIFEST_SHA256",
            self.scenario_manifest_sha256,
        )
        monkeypatch.setattr(
            module,
            "AUTHORIZED_INPUT_SHA256",
            self.input_sha256,
        )
        monkeypatch.setattr(
            module,
            "AUTHORIZED_CONFIG_SEMANTIC_SHA256",
            self.config_semantic_sha256,
        )
        monkeypatch.setattr(
            module,
            "AUTHORIZED_PHASE_STATE_SHA256",
            self.phase_state_sha256,
        )


def _build_fixture(
    tmp_path: Path,
    *,
    numerical_passed: bool = False,
    checkpoint_update: int = 80,
    slope: float = 30.0,
    safety_violation_count: int = 0,
    masked_action_count: int = 0,
    lane_drift: bool = False,
    p01_input_drift: bool = False,
) -> ExistingRunFixture:
    repository_root = tmp_path / "repo"
    source_root = tmp_path / "source" / RUN_ID
    preservation_root = tmp_path / "preservation" / (
        f"{RUN_ID}-active-snapshot-v1"
    )
    scenario_manifest_path = tmp_path / "inputs/scenarios/frozen/manifest.json"
    stop_evidence_path = tmp_path / "external-stop/stop-evidence.json"
    review_base = tmp_path / "review"
    envelope_base = tmp_path / "envelope"

    authority_hashes: dict[str, str] = {}
    for index, relative in enumerate(AUTHORITY_PATHS):
        payload = f"authority-{index}\n".encode("utf-8")
        _write(repository_root / relative, payload)
        authority_hashes[relative] = _sha256(payload)

    test_q = [_scenario_id("test_q24", index) for index in range(24)]
    unseen = [_scenario_id("unseen24", index) for index in range(24)]
    test_c = [f"test-c/scenario-{index:04d}/v1" for index in range(24)]
    validation = [f"validation/scenario-{index:04d}/v1" for index in range(3)]
    replay = [f"replay/scenario-{index:04d}/v1" for index in range(3)]
    used = [*test_q, *unseen, *test_c, *validation, *replay]
    extras = [f"extra/scenario-{index:04d}/v1" for index in range(364 - len(used))]
    all_scenarios = [*used, *extras]
    proofs = [
        _make_proof(scenario_id, index, slope)
        for index, scenario_id in enumerate(all_scenarios)
    ]
    scenario_manifest: dict[str, object] = {
        "schema_version": "mid-dual-scenario-freeze/v1",
        "completion_status": "complete",
        "cohorts": {
            "test_q24": test_q,
            "unseen24": unseen,
            "test_c24": test_c,
            "validation3": validation,
            "replay3": replay,
        },
        "denominator_proofs": proofs,
        "policy_blind_attestation": {
            "attestation_id": "mid-dual-policy-blind-source-attestation/v1",
            "forbidden_inputs_used": {
                "checkpoint": False,
                "policy": False,
                "result": False,
                "reward": False,
                "runtime": False,
            },
        },
    }
    manifest_bytes = _json_bytes(scenario_manifest)
    _write(scenario_manifest_path, manifest_bytes)
    scenario_manifest_sha256 = _sha256(manifest_bytes)
    input_audit = _reconstructed_input_audit(
        manifest_path=scenario_manifest_path,
        manifest=scenario_manifest,
        manifest_sha256=scenario_manifest_sha256,
    )
    input_sha256 = _canonical_sha256(input_audit)
    config_semantic_sha256 = _sha256(b"effective-config-semantic")
    code_sha256 = _sha256(b"runtime-code")
    config = {
        "schema_version": "xunce-mid-dual-g1-effective-config/v1",
        "gate_id": "g1",
        "runner_id": RUNNER_ID,
        "scale_profile": SCALE_PROFILE,
        "run_id": RUN_ID,
        "mode": "formal",
        "output_root": _canon(source_root),
        "required_phase_ids": ["p01", "p02", "p03", "p04", "p05", "p06", "p07"],
        "checkpoint": {
            "path": "D:/fixture/update-00000080/checkpoint.pt",
            "sha256": CHECKPOINT_SHA256,
            "policy_state_sha256": POLICY_STATE_SHA256,
            "update": checkpoint_update,
        },
        "scenario_manifest": {
            "path": _canon(scenario_manifest_path),
            "schema_version": "mid-dual-scenario-freeze/v1",
            "sha256": scenario_manifest_sha256,
        },
        "denominator": {
            "source": DENOMINATOR_SOURCE,
            "algorithm": DENOMINATOR_ALGORITHM,
            "integer_count_binding": True,
        },
        "execution": {
            "worker_count": 8,
            "lane_sizes": [3] * 8,
            "episodes_per_formal_split": 24,
            "max_steps": 128,
        },
        "bootstrap": {
            "confidence_level": 0.95,
            "resamples": 2000,
            "seed": 20260726,
            "unit": "episode",
        },
        "input_audit": {
            "path": "g1_input_audit.json",
            "schema_version": "xunce-mid-dual-g1-input-audit/v1",
            "sha256": input_sha256,
        },
        "input_sha256": input_sha256,
        "config_sha256": config_semantic_sha256,
        "code_sha256": code_sha256,
    }
    _write(source_root / "config.json", _json_bytes(config))

    proofs_by_id = {str(proof["scenario_id"]): proof for proof in proofs}
    phase_rows: dict[str, list[dict[str, object]]] = {}
    for split, phase_id, scenario_ids in (
        ("test_q24", "p03", test_q),
        ("unseen24", "p04", unseen),
    ):
        rows: list[dict[str, object]] = []
        for index, scenario_id in enumerate(scenario_ids):
            proof = proofs_by_id[scenario_id]
            denominator = int(proof["coverable_cell_count"])
            low = (
                not numerical_passed
                and (
                    (split == "test_q24" and index == 20)
                    or (split == "unseen24" and index in {3, 10, 23})
                )
            )
            final_count = denominator // 10 if low else denominator
            common = _common_row(
                split=split,
                phase_id=phase_id,
                index=index,
                scenario_id=scenario_id,
                config_sha256=config_semantic_sha256,
                input_sha256=input_sha256,
                code_sha256=code_sha256,
                manifest_sha256=scenario_manifest_sha256,
            )
            coverage = {
                **common,
                "row_kind": "coverage_episode",
                "schema_version": "xunce-mid-dual-g1-coverage-episode/v1",
                "denominator_source": DENOMINATOR_SOURCE,
                "denominator_algorithm": DENOMINATOR_ALGORITHM,
                "denominator_sha256": proof["coverable_mask_sha256"],
                "denominator_cell_count": denominator,
                "initial_covered_cell_count": 0,
                "final_covered_cell_count": final_count,
                "coverage": final_count / denominator,
                "elapsed_ms": 1.0,
                "steps_executed": 1,
                "termination_reason": "budget_done" if low else "success_done",
                "safety_violation_count": (
                    safety_violation_count if index == 0 else 0
                ),
                "masked_action_count": (
                    masked_action_count if index == 0 else 0
                ),
            }
            if lane_drift and split == "unseen24" and index == 23:
                coverage["lane_id"] = "lane-0"
            rows.append(coverage)
        for index, scenario_id in enumerate(scenario_ids):
            common = _common_row(
                split=split,
                phase_id=phase_id,
                index=index,
                scenario_id=scenario_id,
                config_sha256=config_semantic_sha256,
                input_sha256=input_sha256,
                code_sha256=code_sha256,
                manifest_sha256=scenario_manifest_sha256,
            )
            join_key = _sha256(f"{split}:{index}:0".encode("utf-8"))
            decision_sha = _sha256(f"decision:{split}:{index}".encode("utf-8"))
            rows.append(
                {
                    **common,
                    "row_kind": "decision",
                    "schema_version": "xunce-mid-dual-g1-decision/v1",
                    "step_index": 0,
                    "trace": {
                        "join_key": join_key,
                        "decision_sha256": decision_sha,
                    },
                }
            )
        for index, scenario_id in enumerate(scenario_ids):
            common = _common_row(
                split=split,
                phase_id=phase_id,
                index=index,
                scenario_id=scenario_id,
                config_sha256=config_semantic_sha256,
                input_sha256=input_sha256,
                code_sha256=code_sha256,
                manifest_sha256=scenario_manifest_sha256,
            )
            rows.append(
                {
                    **common,
                    "row_kind": "planner_call",
                    "schema_version": "xunce-mid-dual-g1-planner-call/v1",
                    "step_index": 0,
                    "trace": {
                        "join_key": _sha256(f"{split}:{index}:0".encode("utf-8")),
                        "decision_sha256": _sha256(
                            f"decision:{split}:{index}".encode("utf-8")
                        ),
                        "invalid_action": False,
                        "safety_violation": False,
                    },
                }
            )
        phase_rows[phase_id] = rows

    empty_sha = _sha256(b"")
    phase_audits = {
        "p01": {
            "schema_version": "xunce-mid-dual-g1-phase-audit/v1",
            "gate_id": "g1",
            "runner_id": RUNNER_ID,
            "phase_id": "p01",
            "phase_name": "preflight",
            "status": "passed",
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "policy_state_sha256": POLICY_STATE_SHA256,
            "scenario_manifest_sha256": scenario_manifest_sha256,
            "input_sha256": (
                _sha256(b"drifted-input")
                if p01_input_drift
                else input_sha256
            ),
            "code_sha256": code_sha256,
        },
        "p02": {
            "schema_version": "xunce-mid-dual-g1-phase-audit/v1",
            "gate_id": "g1",
            "runner_id": RUNNER_ID,
            "phase_id": "p02",
            "phase_name": "validation_dry_run",
            "status": "passed",
        },
        "p03": {
            "schema_version": "xunce-mid-dual-g1-phase-audit/v1",
            "gate_id": "g1",
            "runner_id": RUNNER_ID,
            "phase_id": "p03",
            "phase_name": "test_q24",
            "cohort": "test_q24",
            "status": "complete",
            "coverage_episode_count": 24,
            "decision_count": 24,
            "planner_call_count": 24,
        },
        "p04": {
            "schema_version": "xunce-mid-dual-g1-phase-audit/v1",
            "gate_id": "g1",
            "runner_id": RUNNER_ID,
            "phase_id": "p04",
            "phase_name": "unseen24",
            "cohort": "unseen24",
            "status": "complete",
            "coverage_episode_count": 24,
            "decision_count": 24,
            "planner_call_count": 24,
        },
    }
    results_bytes = {
        "p01": b"",
        "p02": b"",
        "p03": _jsonl_bytes(phase_rows["p03"]),
        "p04": _jsonl_bytes(phase_rows["p04"]),
    }
    phase_state: list[dict[str, object]] = []
    attempts: list[dict[str, object]] = []
    for phase_id in ("p01", "p02", "p03", "p04"):
        rows_rel = f"phases/{phase_id}/a01/results.jsonl"
        audit_rel = f"phases/{phase_id}/a01/audit.json"
        row_sha = _sha256(results_bytes[phase_id])
        assert row_sha == empty_sha if phase_id in {"p01", "p02"} else True
        _write(source_root / rows_rel, results_bytes[phase_id])
        _write(source_root / audit_rel, _json_bytes(phase_audits[phase_id]))
        phase_state.append(
            {
                "attempt_id": "a01",
                "phase_id": phase_id,
                "row_sha256": row_sha,
                "rows_path": rows_rel,
            }
        )
        attempts.append(
            {
                "attempt_id": "a01",
                "audit_path": audit_rel,
                "phase_id": phase_id,
                "row_sha256": row_sha,
                "rows_path": rows_rel,
                "status": "accepted",
            }
        )
    phase_state_bytes = _jsonl_bytes(phase_state)
    attempts_bytes = _jsonl_bytes(attempts)
    _write(source_root / "phase-state.jsonl", phase_state_bytes)
    _write(source_root / "phase-attempts.jsonl", attempts_bytes)
    phase_state_sha256 = _sha256(phase_state_bytes)

    source_relatives = [
        "config.json",
        "phase-state.jsonl",
        "phase-attempts.jsonl",
        *[
            f"phases/{phase_id}/a01/{name}"
            for phase_id in ("p01", "p02", "p03", "p04")
            for name in ("audit.json", "results.jsonl")
        ],
    ]
    source_hashes = {
        relative: _sha256((source_root / relative).read_bytes())
        for relative in source_relatives
    }

    raw_process_bytes = b"Get-CimInstance filtered G1 process result: []\r\n"
    stop_evidence = {
        "schema_version": "xunce-mid-dual-g1-existing-run-stop-evidence/v1",
        "run_id": RUN_ID,
        "source_root": _canon(source_root),
        "phase_state_sha256": phase_state_sha256,
        "p04_results_sha256": source_hashes["phases/p04/a01/results.jsonl"],
        "p04_audit_sha256": source_hashes["phases/p04/a01/audit.json"],
        "p04_accepted_at": "2026-07-27T12:54:05.988918+08:00",
        "observed_at": "2026-07-27T13:00:24.417739+08:00",
        "parent_pid": 17212,
        "observed_worker_pids": [
            48712,
            44764,
            44180,
            39500,
            40300,
            36024,
            36532,
            45828,
        ],
        "process_query_raw_bytes_hex": raw_process_bytes.hex(),
        "process_query_raw_sha256": _sha256(raw_process_bytes),
        "matching_parent_process_count": 0,
        "matching_worker_process_count": 0,
        "p05_phase_present": False,
    }
    _write(stop_evidence_path, _json_bytes(stop_evidence))

    preservation_artifact_hashes: dict[str, str] = {}
    for index, name in enumerate(
        (
            "root-commit.tar",
            "path-planner-commit.tar",
            "runtime-working-tree.tar",
            "runtime-file-inventory.json",
            "runtime-files.txt",
            "root-status.txt",
            "path-planner-status.txt",
        )
    ):
        payload = f"preserved-{index}\n".encode("utf-8")
        _write(preservation_root / name, payload)
        preservation_artifact_hashes[name] = _sha256(payload)
    preservation_metadata = {
        "schema_version": (
            "xunce-mid-dual-active-runtime-preservation-snapshot/v1"
        ),
        "run_id": RUN_ID,
        "formal_root": _canon(source_root),
        "artifacts": dict(preservation_artifact_hashes),
        "formal_evidence_eligible": False,
        "eligibility_reason": (
            "preservation_only_pending_independent_dependency_closure_audit"
        ),
    }
    preservation_metadata_bytes = _json_bytes(preservation_metadata)
    _write(
        preservation_root / "snapshot-metadata.json",
        preservation_metadata_bytes,
    )
    preservation_metadata_sha256 = _sha256(preservation_metadata_bytes)

    review_root = review_base / (
        f"g1-existing-run-review-v1-{phase_state_sha256}"
    )
    envelope_root = envelope_base / (
        f"g1-existing-run-assessment-v1-{phase_state_sha256}"
    )
    return ExistingRunFixture(
        repository_root=repository_root,
        source_root=source_root,
        preservation_root=preservation_root,
        scenario_manifest_path=scenario_manifest_path,
        stop_evidence_path=stop_evidence_path,
        review_root=review_root,
        envelope_root=envelope_root,
        source_hashes=source_hashes,
        authority_hashes=authority_hashes,
        preservation_metadata_sha256=preservation_metadata_sha256,
        preservation_artifact_hashes=preservation_artifact_hashes,
        scenario_manifest_sha256=scenario_manifest_sha256,
        input_sha256=input_sha256,
        config_semantic_sha256=config_semantic_sha256,
        phase_state_sha256=phase_state_sha256,
    )


def _create(module: Any, fixture: ExistingRunFixture) -> Path:
    return module.create_existing_run_assessment(
        source_root=fixture.source_root,
        preservation_root=fixture.preservation_root,
        stop_evidence_path=fixture.stop_evidence_path,
        review_root=fixture.review_root,
        envelope_root=fixture.envelope_root,
    )


def test_fixed_allowlist_binds_only_the_user_authorized_existing_run() -> None:
    module = _load_module()
    assert (
        module.AUTHORIZED_SOURCE_ROOT
        == "D:/xunce/out/mid_dual/g1/g1-formal-20260727-0650-cst"
    )
    assert module.AUTHORIZED_RUN_ID == RUN_ID
    assert (
        module.AUTHORIZED_PRESERVATION_ROOT
        == "D:/xunce/out/mid_dual/g1-lineage-preserve/"
        "g1-formal-20260727-0650-cst-active-snapshot-v1"
    )
    assert module.AUTHORITY_ALLOWLIST == {
        AUTHORITY_PATHS[0]: (
            "9d02d80aacf0dbff5046d1fb149f5044ddfc6bfd42faaf0b553746947f89331d"
        ),
        AUTHORITY_PATHS[1]: (
            "1ff2af3c3eb2fe61e02754aa04c00702f1aee082bfbfd8050f3b5031dd1393c4"
        ),
        AUTHORITY_PATHS[2]: (
            "cc444b1fd3446f1fbe0a57e7d9115f601f9c7ff8f26e70edd0094100f840edec"
        ),
    }


def test_authorized_numerical_failure_is_sealed_and_verifies_without_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    fixture = _build_fixture(tmp_path, numerical_passed=False)
    fixture.install_scope(module, monkeypatch)
    before = {
        path.relative_to(fixture.source_root).as_posix(): _sha256(path.read_bytes())
        for path in fixture.source_root.rglob("*")
        if path.is_file()
    }

    created = _create(module, fixture)
    verified = module.verify_existing_run_assessment(
        envelope_root=created,
        expected_source_root=fixture.source_root,
    )

    assert created == fixture.envelope_root
    assert verified.assessment_use_status == "authorized_existing_run"
    assert verified.strict_prestart_lineage_status == "not_satisfied"
    assert verified.lineage_limitation_acknowledged is True
    assert verified.numerical_gate_status == "recomputed_from_raw_rows"
    assert verified.numerical_result_status == "failed"
    assert verified.review_metrics["splits"]["test_q24"]["coverage_80_count"] == 23
    assert verified.review_metrics["splits"]["unseen24"]["coverage_80_count"] == 21
    assert verified.review_metrics["formal_episode_count"] == 48
    envelope = json.loads(
        (created / "assessment-use-envelope.json").read_text(encoding="utf-8")
    )
    assert envelope["source_run"]["p05_replay_status"] == "not_run_by_ruling"
    assert envelope["source_run"]["native_final_manifest_status"] == "not_expected"
    assert envelope["source_run"]["p06_p07_finalization_status"] == (
        "not_run_by_ruling"
    )
    snapshot = json.loads(
        (
            fixture.review_root / "source-phase-snapshot.json"
        ).read_text(encoding="utf-8")
    )
    assert snapshot["reconstructed_input_audit"]["evidence_kind"] == (
        "reconstructed_from_frozen_scenario_manifest/v1"
    )
    assert snapshot["reconstructed_input_audit"]["source_root_artifact_status"] == (
        "absent_expected"
    )
    after = {
        path.relative_to(fixture.source_root).as_posix(): _sha256(path.read_bytes())
        for path in fixture.source_root.rglob("*")
        if path.is_file()
    }
    assert after == before
    assert not (fixture.source_root / "g1_input_audit.json").exists()
    assert not (fixture.source_root / "phases/p05").exists()
    all_output_text = "\n".join(
        path.read_text(encoding="utf-8")
        for root in (fixture.review_root, fixture.envelope_root)
        for path in root.rglob("*.json")
    )
    assert "replay_join" not in all_output_text


def test_complete_numerical_pass_is_legal_despite_strict_lineage_limitation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    fixture = _build_fixture(tmp_path, numerical_passed=True)
    fixture.install_scope(module, monkeypatch)

    verified = module.verify_existing_run_assessment(
        envelope_root=_create(module, fixture),
        expected_source_root=fixture.source_root,
    )

    assert verified.numerical_result_status == "passed"
    assert verified.strict_prestart_lineage_status == "not_satisfied"
    assert verified.review_metrics["midterm_reduced_passed"] is True
    assert verified.review_metrics["final_threshold_reduced_passed"] is True


@pytest.mark.parametrize(
    ("fixture_options", "reason"),
    (
        ({"checkpoint_update": 79}, "existing_run_contract_invalid"),
        ({"slope": 29.5}, "existing_run_input_manifest_invalid"),
        ({"safety_violation_count": 1}, "existing_run_raw_rows_invalid"),
        ({"masked_action_count": 1}, "existing_run_raw_rows_invalid"),
        ({"lane_drift": True}, "existing_run_raw_rows_invalid"),
        ({"p01_input_drift": True}, "existing_run_phase_snapshot_invalid"),
    ),
)
def test_semantic_contract_drift_blocks_before_any_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixture_options: dict[str, object],
    reason: str,
) -> None:
    module = _load_module()
    fixture = _build_fixture(tmp_path, **fixture_options)
    fixture.install_scope(module, monkeypatch)

    with pytest.raises(module.ExistingRunAssessmentBlocked, match=reason):
        _create(module, fixture)

    assert not fixture.review_root.exists()
    assert not fixture.envelope_root.exists()


def test_scope_cannot_be_reused_for_another_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    fixture = _build_fixture(tmp_path)
    fixture.install_scope(module, monkeypatch)
    other = tmp_path / "source/g1-formal-another-run"
    other.mkdir(parents=True)

    with pytest.raises(
        module.ExistingRunAssessmentBlocked,
        match="existing_run_scope_mismatch",
    ):
        module.create_existing_run_assessment(
            source_root=other,
            preservation_root=fixture.preservation_root,
            stop_evidence_path=fixture.stop_evidence_path,
            review_root=fixture.review_root,
            envelope_root=fixture.envelope_root,
        )

    assert not fixture.review_root.exists()
    assert not fixture.envelope_root.exists()


@pytest.mark.parametrize(
    "unexpected_relative",
    (
        "g1_input_audit.json",
        "summary.json",
        "routing.json",
        "report.md",
        "manifest.json",
        "phases/p05/a01/results.jsonl",
        "phases/p06/a01/results.jsonl",
        "phases/p07/a01/results.jsonl",
    ),
)
def test_fabricated_input_or_native_finalization_artifact_blocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unexpected_relative: str,
) -> None:
    module = _load_module()
    fixture = _build_fixture(tmp_path)
    fixture.install_scope(module, monkeypatch)
    _write(fixture.source_root / unexpected_relative, b"fabricated\n")

    with pytest.raises(
        module.ExistingRunAssessmentBlocked,
        match="existing_run_unexpected_artifact",
    ):
        _create(module, fixture)

    assert not fixture.review_root.exists()
    assert not fixture.envelope_root.exists()


def test_nonzero_process_stop_evidence_blocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    fixture = _build_fixture(tmp_path)
    fixture.install_scope(module, monkeypatch)
    stop = json.loads(fixture.stop_evidence_path.read_text(encoding="utf-8"))
    stop["matching_worker_process_count"] = 1
    _write(fixture.stop_evidence_path, _json_bytes(stop))

    with pytest.raises(
        module.ExistingRunAssessmentBlocked,
        match="existing_run_stop_evidence_invalid",
    ):
        _create(module, fixture)

    assert not fixture.review_root.exists()
    assert not fixture.envelope_root.exists()


def test_authority_and_preservation_drift_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    fixture = _build_fixture(tmp_path)
    fixture.install_scope(module, monkeypatch)
    authority = fixture.repository_root / AUTHORITY_PATHS[0]
    authority.write_bytes(authority.read_bytes() + b"drift")

    with pytest.raises(
        module.ExistingRunAssessmentBlocked,
        match="existing_run_authority_invalid",
    ):
        _create(module, fixture)

    assert not fixture.review_root.exists()
    assert not fixture.envelope_root.exists()

    fixture = _build_fixture(tmp_path / "preservation-case")
    fixture.install_scope(module, monkeypatch)
    artifact = fixture.preservation_root / "runtime-files.txt"
    artifact.write_bytes(artifact.read_bytes() + b"drift")
    with pytest.raises(
        module.ExistingRunAssessmentBlocked,
        match="existing_run_preservation_invalid",
    ):
        _create(module, fixture)
    assert not fixture.review_root.exists()
    assert not fixture.envelope_root.exists()


def test_manifest_or_live_source_drift_is_detected_on_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    fixture = _build_fixture(tmp_path)
    fixture.install_scope(module, monkeypatch)
    created = _create(module, fixture)
    review = fixture.review_root / "review.json"
    review.write_bytes(review.read_bytes() + b" ")

    with pytest.raises(
        module.ExistingRunAssessmentBlocked,
        match="existing_run_review_manifest_invalid",
    ):
        module.verify_existing_run_assessment(
            envelope_root=created,
            expected_source_root=fixture.source_root,
        )

    fixture = _build_fixture(tmp_path / "source-drift-case")
    fixture.install_scope(module, monkeypatch)
    created = _create(module, fixture)
    source_config = fixture.source_root / "config.json"
    source_config.write_bytes(source_config.read_bytes() + b" ")
    with pytest.raises(
        module.ExistingRunAssessmentBlocked,
        match="existing_run_source_hash_mismatch",
    ):
        module.verify_existing_run_assessment(
            envelope_root=created,
            expected_source_root=fixture.source_root,
        )


def test_existing_output_roots_are_never_overwritten(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    fixture = _build_fixture(tmp_path)
    fixture.install_scope(module, monkeypatch)
    _create(module, fixture)
    envelope_before = {
        path.relative_to(fixture.envelope_root).as_posix(): path.read_bytes()
        for path in fixture.envelope_root.rglob("*")
        if path.is_file()
    }

    with pytest.raises(
        module.ExistingRunAssessmentBlocked,
        match="existing_run_output_exists",
    ):
        _create(module, fixture)

    assert envelope_before == {
        path.relative_to(fixture.envelope_root).as_posix(): path.read_bytes()
        for path in fixture.envelope_root.rglob("*")
        if path.is_file()
    }


def test_review_copy_cannot_be_verified_from_an_unapproved_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    fixture = _build_fixture(tmp_path)
    fixture.install_scope(module, monkeypatch)
    _create(module, fixture)
    copied_review = tmp_path / "unapproved-review-copy"
    shutil.copytree(fixture.review_root, copied_review)

    with pytest.raises(
        module.ExistingRunAssessmentBlocked,
        match="existing_run_review_manifest_invalid",
    ):
        module._verify_review(
            review_root=copied_review,
            expected_source_root=fixture.source_root.resolve(),
            verifier_source_sha256=_sha256(SCRIPT_PATH.read_bytes()),
        )


def test_module_is_stdlib_recompute_and_does_not_import_producer_stack() -> None:
    _load_module()
    tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    forbidden_fragments = {
        "run_xunce_mid_dual_g1_coverage",
        "xunce_mid_dual_artifacts",
        "collector",
        "lunar_exploration_ppo.eval.standard",
        "lunar_exploration_ppo.eval.midterm_reduced",
        "torch",
        "numpy",
        "subprocess",
    }
    assert not {
        name
        for name in imported
        if any(fragment in name for fragment in forbidden_fragments)
    }
