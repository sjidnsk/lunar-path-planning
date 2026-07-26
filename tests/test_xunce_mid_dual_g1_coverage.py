"""缩减规模 G1 显式 24 场景适配层合同。"""

from __future__ import annotations

import ast
import hashlib
import inspect
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lunar_exploration_ppo.env.standard_training import build_standard_catalog
from lunar_exploration_ppo.eval.evaluator import EvaluationSummary
from lunar_exploration_ppo.utils.artifact_io import ArtifactStore


@lru_cache(maxsize=1)
def _catalog():
    return build_standard_catalog(verify_hashes=False)


def _cohorts() -> dict[str, list[str]]:
    by_split = {
        split: [
            f"{record.scenario_id}/standard-proxy/v1"
            for record in _catalog().records
            if record.split == split
        ]
        for split in ("validation", "test", "unseen")
    }
    return {
        "test_q24": by_split["test"][:24],
        "test_c24": by_split["test"][24:48],
        "unseen24": by_split["unseen"][:24],
        "g3_test_q5": by_split["test"][:5],
        "g3_unseen5": by_split["unseen"][:5],
        "validation3": by_split["validation"][:3],
        "replay3": by_split["test"][:3],
    }


def _manifest_payload() -> dict[str, object]:
    cohorts = _cohorts()
    selected_ids = sorted(
        set(cohorts["test_q24"])
        | set(cohorts["test_c24"])
        | set(cohorts["unseen24"])
        | set(cohorts["validation3"])
    )
    hash_value = "a" * 64
    proofs = [
        {
            "scenario_id": scenario_id,
            "scenario_hash": hashlib.sha256(scenario_id.encode("utf-8")).hexdigest(),
            "split": scenario_id.split("/", 1)[0],
            "coverable_mask_sha256": hashlib.sha256(
                f"mask:{scenario_id}".encode("utf-8")
            ).hexdigest(),
            "coverable_cell_count": 4096,
            "algorithm_id": "exact_reachable_safe_pose_range_los/v1",
            "exact": True,
            "coverage_denominator_source": (
                "reachable_observable_free_highres_cells/v1"
            ),
            "coverage_denominator_algorithm": (
                "exact_reachable_safe_pose_range_los/v1"
            ),
            "semantic_alias_proven": True,
        }
        for scenario_id in selected_ids
    ]
    return {
        "schema_version": "mid-dual-scenario-freeze/v1",
        "completion_status": "complete",
        "config_sha256": hash_value,
        "descriptor_catalog_sha256": hash_value,
        "source_manifest_sha256": hash_value,
        "stage6_coverage_manifest_sha256": hash_value,
        "stage6_catalog_sha256": _catalog().sha256,
        "reconstruction_index_sha256": hash_value,
        "denominator_proofs_sha256": hash_value,
        "descriptor_generator": {"id": "fixture", "version": "v1"},
        "policy_blind_attestation": {"attestation_id": "fixture"},
        "source_pools": {
            "validation": hash_value,
            "test": hash_value,
            "unseen": hash_value,
        },
        "cohort_sizes": {
            "test_q24": 24,
            "test_c24": 24,
            "unseen24": 24,
            "g3_test_q5": 5,
            "g3_unseen5": 5,
            "validation3": 3,
            "replay3": 3,
        },
        "cohorts": cohorts,
        "denominator_proofs": proofs,
        "bundle_files": [
            {
                "path": name,
                "sha256": hashlib.sha256(name.encode("utf-8")).hexdigest(),
                "size_bytes": len(name),
            }
            for name in (
                "config.json",
                "descriptors.jsonl",
                "source-manifest.json",
                "reconstruction-index.jsonl",
                "denominator-proofs.jsonl",
            )
        ],
    }


def _load_fixture_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from lunar_exploration_ppo.eval import midterm_reduced

    root = tmp_path / "frozen"
    root.mkdir()
    payload = ArtifactStore.canonical_json_bytes(_manifest_payload())
    (root / "manifest.json").write_bytes(payload)
    monkeypatch.setattr(midterm_reduced, "_verify_frozen_bundle", lambda _root: True)
    return midterm_reduced.load_frozen_scenario_manifest(
        bundle_root=root,
        expected_manifest_sha256=hashlib.sha256(payload).hexdigest(),
    )


def _install_identity_factory(
    monkeypatch: pytest.MonkeyPatch,
    midterm_reduced,
):
    built: list[str] = []

    class IdentityFactory:
        def __init__(self, catalog) -> None:
            assert catalog is _catalog()

        def build(self, record):
            scenario_id = f"{record.scenario_id}/standard-proxy/v1"
            built.append(scenario_id)
            return SimpleNamespace(
                scenario_id=scenario_id,
                scenario_hash=hashlib.sha256(
                    scenario_id.encode("utf-8")
                ).hexdigest(),
            )

    monkeypatch.setattr(midterm_reduced, "StandardScenarioFactory", IdentityFactory)
    return built


def test_reduced_builder_accepts_only_explicit_frozen_24_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.eval import midterm_reduced

    frozen = _load_fixture_manifest(tmp_path, monkeypatch)
    _install_identity_factory(monkeypatch, midterm_reduced)
    jobs = midterm_reduced.build_midterm_reduced_jobs(
        catalog=_catalog(),
        frozen_manifest=frozen,
        cohort="test_q24",
        evaluation_seed_start=2026072600,
    )

    assert len(jobs) == 24
    assert tuple(job.scenario_id for job in jobs) == tuple(
        scenario_id.removesuffix("/standard-proxy/v1")
        for scenario_id in _cohorts()["test_q24"]
    )
    assert tuple(job.episode_index for job in jobs) == tuple(range(24))
    assert len({job.scenario_id for job in jobs}) == 24
    assert "scenario_ids" not in inspect.signature(
        midterm_reduced.build_midterm_reduced_jobs
    ).parameters

    tampered = _manifest_payload()
    tampered["cohorts"]["test_q24"][1] = tampered["cohorts"]["test_q24"][0]
    payload = ArtifactStore.canonical_json_bytes(tampered)
    bad_root = tmp_path / "tampered"
    bad_root.mkdir()
    (bad_root / "manifest.json").write_bytes(payload)
    with pytest.raises(
        midterm_reduced.MidtermReducedEvaluationError,
        match="cohort",
    ):
        midterm_reduced.load_frozen_scenario_manifest(
            bundle_root=bad_root,
            expected_manifest_sha256=hashlib.sha256(payload).hexdigest(),
        )


def test_reduced_partition_is_eight_lanes_of_three(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.eval import midterm_reduced, standard

    frozen = _load_fixture_manifest(tmp_path, monkeypatch)
    _install_identity_factory(monkeypatch, midterm_reduced)
    jobs = midterm_reduced.build_midterm_reduced_jobs(
        catalog=_catalog(),
        frozen_manifest=frozen,
        cohort="unseen24",
        evaluation_seed_start=2026072700,
    )
    lanes = standard.partition_standard_evaluation_jobs(jobs)

    assert len(lanes) == 8
    assert tuple(len(lane) for lane in lanes) == (3,) * 8
    assert tuple(
        job.episode_index for lane in lanes for job in lane
    ) == (
        0,
        8,
        16,
        1,
        9,
        17,
        2,
        10,
        18,
        3,
        11,
        19,
        4,
        12,
        20,
        5,
        13,
        21,
        6,
        14,
        22,
        7,
        15,
        23,
    )


def test_reduced_builder_cannot_build_64_then_slice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.eval import midterm_reduced, standard

    frozen = _load_fixture_manifest(tmp_path, monkeypatch)
    _install_identity_factory(monkeypatch, midterm_reduced)

    def reject_canonical_builder(*_args, **_kwargs):
        raise AssertionError("reduced schedule attempted canonical build-and-slice")

    monkeypatch.setattr(
        standard,
        "build_standard_evaluation_schedule",
        reject_canonical_builder,
    )
    jobs = midterm_reduced.build_midterm_reduced_jobs(
        catalog=_catalog(),
        frozen_manifest=frozen,
        cohort="test_q24",
        evaluation_seed_start=2026072600,
    )

    assert len(jobs) == 24
    assert tuple(job.scenario_id for job in jobs) == tuple(
        scenario_id.removesuffix("/standard-proxy/v1")
        for scenario_id in _cohorts()["test_q24"]
    )


def test_reduced_builder_reconstructs_full_stage6_identity_and_hash_one_to_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.eval import midterm_reduced

    frozen = _load_fixture_manifest(tmp_path, monkeypatch)
    built = _install_identity_factory(monkeypatch, midterm_reduced)
    plan = midterm_reduced.build_midterm_reduced_job_plan(
        catalog=_catalog(),
        frozen_manifest=frozen,
        cohort="test_q24",
        evaluation_seed_start=2026072600,
    )

    assert tuple(built) == tuple(_cohorts()["test_q24"])
    assert tuple(plan.frozen_scenario_id_by_record_id.values()) == tuple(
        _cohorts()["test_q24"]
    )
    assert tuple(plan.record_id_by_frozen_scenario_id.values()) == tuple(
        scenario_id.removesuffix("/standard-proxy/v1")
        for scenario_id in _cohorts()["test_q24"]
    )
    assert all(
        plan.scenario_hash_by_record_id[record_id]
        == hashlib.sha256(frozen_id.encode("utf-8")).hexdigest()
        for record_id, frozen_id in plan.frozen_scenario_id_by_record_id.items()
    )

    first_frozen_id = _cohorts()["test_q24"][0]
    bad_proofs = dict(frozen.denominator_proofs)
    bad_proofs[first_frozen_id] = {
        **dict(bad_proofs[first_frozen_id]),
        "scenario_hash": "f" * 64,
    }
    corrupted = midterm_reduced.FrozenScenarioManifest(
        bundle_root=frozen.bundle_root,
        manifest_sha256=frozen.manifest_sha256,
        stage6_catalog_sha256=frozen.stage6_catalog_sha256,
        cohorts=frozen.cohorts,
        denominator_proofs=bad_proofs,
    )
    with pytest.raises(
        midterm_reduced.MidtermReducedEvaluationError,
        match="scenario identity",
    ):
        midterm_reduced.build_midterm_reduced_job_plan(
            catalog=_catalog(),
            frozen_manifest=corrupted,
            cohort="test_q24",
            evaluation_seed_start=2026072600,
        )


def test_frozen_manifest_hash_and_semantic_verification_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.eval import midterm_reduced

    root = tmp_path / "frozen"
    root.mkdir()
    payload = ArtifactStore.canonical_json_bytes(_manifest_payload())
    (root / "manifest.json").write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()

    monkeypatch.setattr(midterm_reduced, "_verify_frozen_bundle", lambda _root: False)
    with pytest.raises(
        midterm_reduced.MidtermReducedEvaluationError,
        match="verification",
    ):
        midterm_reduced.load_frozen_scenario_manifest(
            bundle_root=root,
            expected_manifest_sha256=digest,
        )

    monkeypatch.setattr(midterm_reduced, "_verify_frozen_bundle", lambda _root: True)
    with pytest.raises(
        midterm_reduced.MidtermReducedEvaluationError,
        match="SHA-256",
    ):
        midterm_reduced.load_frozen_scenario_manifest(
            bundle_root=root,
            expected_manifest_sha256="b" * 64,
        )


def test_frozen_manifest_rejects_unsafe_descriptor_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.eval import midterm_reduced
    from lunar_exploration_ppo.utils.path_security import PathSecurityError

    root = tmp_path / "frozen"
    root.mkdir()
    payload = ArtifactStore.canonical_json_bytes(_manifest_payload())
    (root / "manifest.json").write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()

    monkeypatch.setattr(midterm_reduced, "_verify_frozen_bundle", lambda _root: True)

    def reject_unsafe_read(*_args: object, **_kwargs: object) -> object:
        raise PathSecurityError("test manifest is a reparse point")

    monkeypatch.setattr(
        midterm_reduced,
        "secure_read_bytes",
        reject_unsafe_read,
        raising=False,
    )
    with pytest.raises(
        midterm_reduced.MidtermReducedEvaluationError,
        match="unsafe",
    ):
        midterm_reduced.load_frozen_scenario_manifest(
            bundle_root=root,
            expected_manifest_sha256=digest,
        )


def test_update80_loader_is_fixed_cuda_fp32_and_rechecks_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.eval import midterm_reduced

    assert tuple(
        inspect.signature(midterm_reduced.load_midterm_update80_policy).parameters
    ) == ()
    source = Path(midterm_reduced.__file__).read_text(encoding="utf-8")
    imports = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    imported_modules = {
        alias.name
        for node in imports
        for alias in (
            node.names
            if isinstance(node, ast.Import)
            else [ast.alias(name=node.module or "")]
        )
    }
    assert "lunar_exploration_ppo.ppo.trainer" not in imported_modules
    assert not any("update" in name for name in imported_modules)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(
        midterm_reduced.MidtermReducedEvaluationError,
        match="CUDA",
    ):
        midterm_reduced.load_midterm_update80_policy()

    captured: dict[str, object] = {}
    policy = nn.Linear(2, 1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        midterm_reduced,
        "_sha256_file",
        lambda path: midterm_reduced.UPDATE80_CHECKPOINT_SHA256,
    )
    monkeypatch.setattr(
        midterm_reduced,
        "_policy_state_sha256",
        lambda _policy: midterm_reduced.UPDATE80_POLICY_STATE_SHA256,
    )
    monkeypatch.setattr(
        midterm_reduced,
        "_validate_cuda_fp32_policy",
        lambda _policy: None,
    )

    def load_policy(**kwargs):
        captured.update(kwargs)
        return policy

    monkeypatch.setattr(
        midterm_reduced,
        "load_stage4_policy_for_standard",
        load_policy,
    )
    assert midterm_reduced.load_midterm_update80_policy() is policy
    assert captured == {
        "checkpoint_path": midterm_reduced.UPDATE80_CHECKPOINT_PATH,
        "checkpoint_sha256": (
            "35e04c86f9f973af028fb08f0175d42ab45378d09e1f2b96ee6aad6e4c12b5b5"
        ),
        "policy_state_sha256": (
            "3123e6adde99be41e3bd5cc2f3843068e5416892395926d759c2c6a243ffd381"
        ),
        "device": "cuda",
    }


def test_denominator_audit_and_path_steps_to_80_99_are_derived_from_trace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.eval import midterm_reduced, standard

    frozen = _load_fixture_manifest(tmp_path, monkeypatch)
    _install_identity_factory(monkeypatch, midterm_reduced)
    job_plan = midterm_reduced.build_midterm_reduced_job_plan(
        catalog=_catalog(),
        frozen_manifest=frozen,
        cohort="test_q24",
        evaluation_seed_start=2026072600,
    )
    scenario_ids = tuple(job.scenario_id for job in job_plan.jobs)
    episode_rows = tuple(
        {
            "episode_id": f"test-q24-{index:02d}",
            "episode_index": index,
            "scenario_id": scenario_id,
            "lane_id": f"lane-{index % 8}",
            "initial_coverage_rate": 0.25,
            "final_coverage": 0.99,
            "steps_executed": 2,
        }
        for index, scenario_id in enumerate(scenario_ids)
    )
    step_rows = tuple(
        row
        for index, scenario_id in enumerate(scenario_ids)
        for row in (
            {
                "join_key": f"{scenario_id}:0",
                "episode_id": f"test-q24-{index:02d}",
                "episode_index": index,
                "scenario_id": scenario_id,
                "step_index": 0,
                "coverage_rate": 0.80,
                "cumulative_path_length_m": 1.25,
            },
            {
                "join_key": f"{scenario_id}:1",
                "episode_id": f"test-q24-{index:02d}",
                "episode_index": index,
                "scenario_id": scenario_id,
                "step_index": 1,
                "coverage_rate": 0.99,
                "cumulative_path_length_m": 2.50,
            },
        )
    )
    execution = standard.StandardEvaluationExecution(
        summary=EvaluationSummary(
            method="ppo_policy",
            scale_profile="Standard v1",
            episodes=(),
            metrics={"episode_count": 24},
            bootstrap_audit={},
            fairness_audit={},
        ),
        episode_rows=episode_rows,
        decision_rows=(),
        step_rows=step_rows,
    )

    result = midterm_reduced.derive_midterm_reduced_trace_summary(
        execution=execution,
        frozen_manifest=frozen,
        cohort="test_q24",
        job_plan=job_plan,
    )

    assert result["schema_version"] == "midterm-reduced-trace-summary/v1"
    assert result["split"] == "test"
    assert result["episode_count"] == 24
    assert result["denominator_audit"] == {
        "source": "reachable_observable_free_highres_cells/v1",
        "algorithm": "exact_reachable_safe_pose_range_los/v1",
        "proof_count": 24,
        "nonempty_count": 24,
        "exact_count": 24,
        "semantic_alias_proven_count": 24,
        "passed": True,
    }
    assert result["episodes"][0]["steps_to_80"] == 1
    assert result["episodes"][0]["path_length_to_80_m"] == pytest.approx(1.25)
    assert result["episodes"][0]["steps_to_99"] == 2
    assert result["episodes"][0]["path_length_to_99_m"] == pytest.approx(2.50)
    assert result["final_coverage"]["mean"] == pytest.approx(0.99)
    assert result["final_coverage"]["coverage_80_count"] == 24
    assert result["final_coverage"]["coverage_99_count"] == 24
