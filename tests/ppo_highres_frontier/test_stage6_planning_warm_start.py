"""Stage 6 U74 -> child U75 planning-safety warm-start 合同。"""

from __future__ import annotations

import copy
from contextlib import nullcontext
import hashlib
import importlib
import importlib.util
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/ppo_highres_frontier_stage6_v1.json"
PARENT_STAGE_ROOT = Path(
    "D:/xunce/out/ppo_frontier/"
    "s6-standard-single-r1-20260718T220434Z/s6"
)
SCRIPT = ROOT / "scripts/create_ppo_stage6_planning_warm_start.py"
RUNNER = ROOT / "scripts/run_ppo_stage6_standard.py"
DESIGN = ROOT / (
    "docs/superpowers/specs/"
    "2026-07-23-ppo-stage6-planning-unknown-buffer-design-addendum.md"
)
PLAN = ROOT / (
    "docs/superpowers/plans/"
    "2026-07-23-ppo-stage6-planning-unknown-buffer-warm-start.md"
)


def _module():
    return importlib.import_module(
        "lunar_exploration_ppo.workflows.stage6_planning_warm_start"
    )


def _artifact() -> dict[str, object]:
    return {
        "schema_version": "stage6_planning_u74_warm_start/v1",
        "mode": "child_lineage_warm_start_not_exact_resume/v1",
        "parent": {
            "run_id": "s6-standard-single-r1-20260718T220434Z",
            "seed": 20260716,
            "update": 74,
            "checkpoint_sha256": (
                "ce9ea8cb047b5acc0ffc4f5d63084f3d54312cf55ecb0b92b8fc20bf43791553"
            ),
            "manifest_sha256": (
                "6c0e40d2dd9ea489f4c67a6a43868dc80751f07961f5db8ccf5c804213e340c5"
            ),
            "complete_sha256": (
                "6cd6e6210283fc71a5205a727f82c07ff448458d4ac06ad8080277e2502960ef"
            ),
            "policy_state_sha256": (
                "c1650b7bed6fa22387ac5b5fdb9d80a6aa0baf3a1d425e868d81586070c5db20"
            ),
            "config_sha256": (
                "7b1c37673c105c34e7bd71d04d4f508d3a167cdf77426b0b35c7aff86ba8adae"
            ),
            "lineage_sha256": (
                "c7a0a7bf98b5aa4a39094ec01a2a0031d87ad8fc71060323474b20ebb134530c"
            ),
            "resource_accepted": True,
            "receipt_transaction_key": "0073:20260716:update:074",
            "source_repair_last_ordinal": 6,
        },
        "discarded_u75_attempt1": {
            "transaction_key": "0074:20260716:update:075",
            "attempt": 1,
            "segment_index": 8,
            "only_phase": "pre",
            "accepted": False,
            "discarded": True,
        },
        "child": {
            "run_id": "s6-planning-u74-child-r1",
            "seed": 20260716,
            "first_update": 75,
            "final_update": 100,
            "new_semantics_update_count": 26,
            "effective_config_sha256": "a" * 64,
        },
        "state_transfer": {
            "imported": [
                "model_state_dict",
                "optimizer_state_dict",
                "normalization_stats",
                "rng_state",
                "scenario_sampler_state",
            ],
            "discarded": [
                "vector_env_states",
                "best_record",
                "eval_metrics",
                "u75_attempt1_rollout",
                "u75_attempt1_candidate_snapshot",
                "u75_attempt1_old_logprob",
                "u75_attempt1_episode_state",
            ],
            "child_vector_env_reset_count": 8,
            "child_best_validation_updates": [80, 90, 100],
        },
        "bindings": {
            "design_sha256": "b" * 64,
            "plan_sha256": "c" * 64,
            "source_set_sha256": "d" * 64,
            "spec_review_sha256": "e" * 64,
            "quality_review_sha256": "f" * 64,
            "launch_authorization_sha256": "1" * 64,
        },
    }


def _child_journal_bindings(checkpoint_sha256: str) -> dict[str, object]:
    from test_stage6_machine_preflight import (
        _environment_sha256,
        _valid_environment_identity,
    )

    environment = _valid_environment_identity()
    reviewed_tree = "1" * 40
    return {
        "config_sha256": "a" * 64,
        "source_set_sha256": "2" * 64,
        "prospective_tree_sha256": hashlib.sha256(
            reviewed_tree.encode("ascii")
        ).hexdigest(),
        "data_sha256": "4" * 64,
        "environment_identity": environment,
        "environment_sha256": _environment_sha256(environment),
        "stage5_gate_sha256": "5" * 64,
        "formal_run_id": "s6-standard-single-r1-20260723T010203Z",
        "changed_path_set_sha256": "7" * 64,
        "review_authorization_record_sha256": "8" * 64,
        "authorization_file_sha256": "9" * 64,
        "review_identity_sha256": "b" * 64,
        "reviewed_prospective_git_tree": reviewed_tree,
        "frozen_diff_sha256": "c" * 64,
        "spec_review_sha256": "d" * 64,
        "quality_review_sha256": "e" * 64,
        "checkpoint_sha256": checkpoint_sha256,
    }


def _child_acceptance_immutable(checkpoint_sha256: str) -> dict[str, object]:
    immutable = _child_journal_bindings(checkpoint_sha256)
    immutable.update(
        {
            "coverage_cache_manifest_path": "D:/xunce/review/manifest.json",
            "coverage_cache_manifest_sha256": "f" * 64,
            "coverage_cache_manifest_size_bytes": 1,
            "coverage_cache_root": "D:/xunce/cache/ppo_frontier",
            "coverage_cache_entry_set_sha256": "0" * 64,
            "coverage_cache_runtime_mode": "persistent_exact_manifest_read_only/v1",
            "coverage_cache_formal_audit_sha256": "1" * 64,
            "coverage_cache_formal_audit_size_bytes": 1,
        }
    )
    return immutable


def _transaction_checkpoint(transaction: object):
    from lunar_exploration_ppo.ppo.standard_training import (
        StandardTransactionCheckpoint,
    )

    key = str(transaction.key)  # type: ignore[attr-defined]

    def digest(label: str) -> str:
        return hashlib.sha256(f"{key}:{label}".encode("utf-8")).hexdigest()

    return StandardTransactionCheckpoint(
        transaction_key=key,
        seed=int(transaction.seed),  # type: ignore[attr-defined]
        update=int(transaction.update),  # type: ignore[attr-defined]
        checkpoint_sha256=digest("checkpoint"),
        complete_marker_sha256=digest("complete"),
        policy_state_sha256=digest("policy"),
    )


def _commit_child_transaction(
    *,
    journal: object,
    receipt_index: object,
    transactions: tuple[object, ...],
    transaction: object,
) -> object:
    from lunar_exploration_ppo.ppo.standard_training import (
        commit_checkpointed_transaction,
    )

    checkpoint = _transaction_checkpoint(transaction)
    assert receipt_index.append_once(checkpoint) is True  # type: ignore[attr-defined]
    commit_checkpointed_transaction(
        journal=journal,
        transactions=transactions,
        transaction=transaction,
        checkpoint=checkpoint,
        bindings=_child_journal_bindings(checkpoint.checkpoint_sha256),
        receipt_rows=receipt_index.verify(),  # type: ignore[attr-defined]
    )
    return checkpoint


def _receipt_rows_for_transactions(
    path: Path,
    transactions: tuple[object, ...],
):
    from lunar_exploration_ppo.ppo.standard_training import CheckpointReceiptIndex

    index = CheckpointReceiptIndex(path)
    for transaction in transactions:
        assert index.append_once(_transaction_checkpoint(transaction)) is True
    return index.verify()


def _canonical_jsonl_bytes(rows: list[dict[str, object]]) -> bytes:
    return b"".join(
        (
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        for row in rows
    )


def _rehash_journal_rows(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    previous = str(rows[0]["previous_record_hash"])
    result: list[dict[str, object]] = []
    for row in rows:
        event = {
            "state": row["state"],
            "previous_record_hash": previous,
            "bindings": row["bindings"],
        }
        record_hash = hashlib.sha256(
            (
                json.dumps(
                    event,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8")
        ).hexdigest()
        result.append({**event, "record_hash": record_hash})
        previous = record_hash
    return result


def test_warm_start_module_is_independent_from_source_repair_ordinal_chain() -> None:
    assert (
        importlib.util.find_spec(
            "lunar_exploration_ppo.workflows.stage6_planning_warm_start"
        )
        is not None
    )
    module = _module()
    assert module.WARM_START_SCHEMA == "stage6_planning_u74_warm_start/v1"
    assert module.PARENT_UPDATE == 74
    assert module.CHILD_FIRST_UPDATE == 75
    assert module.CHILD_FINAL_UPDATE == 100
    assert not hasattr(module, "SOURCE_REPAIR_ORDINAL")


def test_child_effective_config_hash_binds_new_planning_semantics() -> None:
    module = _module()
    from lunar_exploration_ppo.configs.stage6 import load_stage6_config

    first = module.planning_effective_config_sha256(CONFIG.read_bytes())
    second = module.planning_effective_config_sha256(CONFIG.read_bytes())
    config = load_stage6_config(CONFIG)

    assert first == second
    assert len(first) == 64
    assert first != module.PARENT_CONFIG_SHA256
    assert config.planning_unknown_buffer_m == 0.75
    assert config.reset_local_safety_scan_range_m == 0.75
    assert config.reset_local_safety_scan_fov_deg == 360.0
    assert config.reset_local_safety_scan_ray_angle_step_deg == 1.0


def test_child_effective_config_has_one_canonical_persistable_snapshot() -> None:
    module = _module()

    first = module.planning_effective_config_bytes(CONFIG.read_bytes())
    second = module.planning_effective_config_bytes(CONFIG.read_bytes())
    value = json.loads(first.decode("utf-8"))

    assert first == second
    assert first.endswith(b"\n")
    assert first == (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    assert hashlib.sha256(first).hexdigest() == (
        module.planning_effective_config_sha256(CONFIG.read_bytes())
    )
    assert value["base_config_sha256"] == module.PARENT_CONFIG_SHA256
    assert value["planning_safety"] == {
        "planning_unknown_buffer_m": 0.75,
        "reset_scan_order": [
            "reset_local_safety",
            "reset_exploration",
        ],
        "reset_local_safety_scan": {
            "fov_deg": 360.0,
            "range_m": 0.75,
            "ray_angle_step_deg": 1.0,
        },
    }
    assert value["warm_start"]["schema_version"] == module.WARM_START_SCHEMA
    assert value["warm_start"]["parent_update"] == 74
    assert value["warm_start"]["child_updates"] == list(range(75, 101))


def test_child_effective_config_recovers_frozen_base_config() -> None:
    module = _module()

    effective = module.planning_effective_config_bytes(CONFIG.read_bytes())
    snapshot = module.parse_planning_effective_config_bytes(effective)

    assert snapshot.effective_config_bytes == effective
    assert snapshot.effective_config_sha256 == hashlib.sha256(
        effective
    ).hexdigest()
    assert snapshot.base_config_sha256 == module.PARENT_CONFIG_SHA256
    assert snapshot.base_config.planning_unknown_buffer_m == 0.75
    assert snapshot.base_config.reset_local_safety_scan_range_m == 0.75
    assert snapshot.base_config.reset_local_safety_scan_fov_deg == 360.0
    assert snapshot.base_config.reset_local_safety_scan_ray_angle_step_deg == 1.0


def test_execution_identity_uses_effective_config_bytes_not_parent_base() -> None:
    module = _module()
    from lunar_exploration_ppo.workflows import stage6

    effective = module.planning_effective_config_bytes(CONFIG.read_bytes())
    identity = stage6.stage6_execution_identity(
        repo_root=ROOT,
        config_path=CONFIG,
        effective_config_bytes=effective,
    )

    assert identity["config_sha256"] == hashlib.sha256(effective).hexdigest()
    assert identity["config_sha256"] != module.PARENT_CONFIG_SHA256


def test_warm_start_context_binds_exact_effective_config_bytes() -> None:
    module = _module()
    effective = module.planning_effective_config_bytes(CONFIG.read_bytes())
    artifact = _artifact()
    artifact["child"]["effective_config_sha256"] = hashlib.sha256(  # type: ignore[index]
        effective
    ).hexdigest()

    context = module.validate_planning_warm_start_artifact(
        artifact,
        expected_child_run_id="s6-planning-u74-child-r1",
        expected_child_config_bytes=effective,
    )

    assert context.child_effective_config_bytes == effective
    assert context.child_effective_config_sha256 == hashlib.sha256(
        effective
    ).hexdigest()


def test_warm_start_artifact_freezes_child_lineage_and_state_transfer() -> None:
    module = _module()

    context = module.validate_planning_warm_start_artifact(
        _artifact(),
        expected_child_run_id="s6-planning-u74-child-r1",
        expected_child_config_sha256="a" * 64,
    )

    assert context.child_first_update == 75
    assert context.child_final_update == 100
    assert context.new_semantics_update_count == 26
    assert context.parent_run_id != context.child_run_id
    assert context.child_effective_config_sha256 != context.parent_config_sha256
    assert context.imported_state_names == (
        "model_state_dict",
        "optimizer_state_dict",
        "normalization_stats",
        "rng_state",
        "scenario_sampler_state",
    )
    assert "vector_env_states" in context.discarded_state_names
    assert "best_record" in context.discarded_state_names
    assert "eval_metrics" in context.discarded_state_names
    assert context.child_best_validation_updates == (80, 90, 100)


def test_loaded_warm_start_artifact_detects_later_byte_drift(tmp_path: Path) -> None:
    module = _module()
    path = tmp_path / "planning-warm-start.json"
    path.write_bytes(
        (
            json.dumps(
                _artifact(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    )
    context, artifact_sha256 = module.load_planning_warm_start_artifact(
        path,
        expected_child_run_id="s6-planning-u74-child-r1",
        expected_child_config_sha256="a" * 64,
    )
    assert context.artifact_sha256 == artifact_sha256
    context.require_current("before drift")

    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(module.Stage6PlanningWarmStartError, match="changed"):
        context.require_current("after drift")


def test_warm_start_expected_bindings_use_current_real_evidence() -> None:
    module = _module()
    record = {
        "source_set_sha256": "2" * 64,
        "spec_review_sha256": "3" * 64,
        "quality_review_sha256": "4" * 64,
        "authorization_file_sha256": "5" * 64,
    }
    handle = SimpleNamespace(canonical_record=lambda: dict(record))

    bindings = module.planning_warm_start_expected_bindings(
        repo_root=ROOT,
        review_authorization_handle=handle,
    )

    assert bindings == {
        "design_sha256": hashlib.sha256(DESIGN.read_bytes()).hexdigest(),
        "plan_sha256": hashlib.sha256(PLAN.read_bytes()).hexdigest(),
        "source_set_sha256": "2" * 64,
        "spec_review_sha256": "3" * 64,
        "quality_review_sha256": "4" * 64,
        "launch_authorization_sha256": "5" * 64,
    }


@pytest.mark.parametrize(
    "field",
    [
        "design_sha256",
        "plan_sha256",
        "source_set_sha256",
        "spec_review_sha256",
        "quality_review_sha256",
        "launch_authorization_sha256",
    ],
)
def test_warm_start_artifact_rejects_each_well_formed_evidence_drift(
    field: str,
) -> None:
    module = _module()
    effective = module.planning_effective_config_bytes(CONFIG.read_bytes())
    artifact = _artifact()
    artifact["child"]["effective_config_sha256"] = hashlib.sha256(  # type: ignore[index]
        effective
    ).hexdigest()
    expected = dict(artifact["bindings"])  # type: ignore[arg-type]
    artifact["bindings"][field] = "0" * 64  # type: ignore[index]

    with pytest.raises(
        module.Stage6PlanningWarmStartError,
        match="evidence binding",
    ):
        module.validate_planning_warm_start_artifact(
            artifact,
            expected_child_run_id="s6-planning-u74-child-r1",
            expected_child_config_bytes=effective,
            expected_bindings=expected,
        )


def test_planning_warm_start_artifact_is_part_of_exact_input_pin() -> None:
    from lunar_exploration_ppo.workflows import stage6

    warm = Path("D:/xunce/review/planning-warm-start.json")
    requests = stage6._stage6_input_pin_requests(
        repo_root=ROOT,
        config_path=CONFIG,
        stage5_authority=SimpleNamespace(snapshots=()),
        review_authorization_handle=SimpleNamespace(evidence_paths=()),
        planning_warm_start_path=warm,
    )

    assert ("planning-warm-start:artifact", warm) in requests
    parent_labels = {
        label
        for label, _path in requests
        if label.startswith("planning-warm-start:parent:")
    }
    assert parent_labels == {
        "planning-warm-start:parent:checkpoint",
        "planning-warm-start:parent:manifest",
        "planning-warm-start:parent:complete",
        "planning-warm-start:parent:config",
        "planning-warm-start:parent:job-state",
        "planning-warm-start:parent:receipt-index",
        "planning-warm-start:parent:lineage-audit",
        "planning-warm-start:parent:resource-audit",
        "planning-warm-start:parent:source-repair-ordinal1",
        "planning-warm-start:parent:source-repair-ordinal2",
        "planning-warm-start:parent:source-repair-ordinal3",
        "planning-warm-start:parent:source-repair-ordinal4",
        "planning-warm-start:parent:source-repair-ordinal5",
        "planning-warm-start:parent:source-repair-ordinal6",
    }


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("parent", "run_id"), "wrong-run", "parent"),
        (("parent", "seed"), 7, "seed"),
        (("parent", "update"), 73, "update"),
        (("parent", "checkpoint_sha256"), "0" * 64, "checkpoint"),
        (("parent", "manifest_sha256"), "0" * 64, "manifest"),
        (("parent", "complete_sha256"), "0" * 64, "complete"),
        (("parent", "config_sha256"), "0" * 64, "config"),
        (("parent", "lineage_sha256"), "0" * 64, "lineage"),
        (("parent", "resource_accepted"), False, "accepted"),
        (("discarded_u75_attempt1", "discarded"), False, "discard"),
        (("discarded_u75_attempt1", "only_phase"), "post", "pre"),
        (("child", "first_update"), 76, "first"),
        (("child", "effective_config_sha256"), "0" * 64, "config"),
        (("state_transfer", "child_vector_env_reset_count"), 0, "reset"),
        (("state_transfer", "child_best_validation_updates"), [10, 80], "best"),
    ],
)
def test_warm_start_artifact_fails_closed_on_parent_child_or_discard_drift(
    path: tuple[str, str],
    value: object,
    match: str,
) -> None:
    module = _module()
    artifact = copy.deepcopy(_artifact())
    artifact[path[0]][path[1]] = value  # type: ignore[index]

    with pytest.raises(module.Stage6PlanningWarmStartError, match=match):
        module.validate_planning_warm_start_artifact(
            artifact,
            expected_child_run_id="s6-planning-u74-child-r1",
            expected_child_config_sha256="a" * 64,
        )


def test_checkpoint_state_selection_imports_only_approved_parent_state() -> None:
    module = _module()
    payload = {
        "model_state_dict": {"weight": "model"},
        "optimizer_state_dict": {"state": "optimizer"},
        "normalization_stats": {"count": 74},
        "rng_state": {"python": "rng"},
        "scenario_sampler_state": {"workers": [{"cursor": 9}] * 8},
        "vector_env_states": [{"episode": index} for index in range(8)],
        "best_record": {"update": 10},
        "eval_metrics": {"validation": {"update": 70}},
    }

    selected = module.select_parent_warm_start_state(payload)

    assert set(selected) == {
        "model_state_dict",
        "optimizer_state_dict",
        "normalization_stats",
        "rng_state",
        "scenario_sampler_state",
    }
    assert selected["scenario_sampler_state"] == payload["scenario_sampler_state"]
    assert "vector_env_states" not in selected
    assert "best_record" not in selected
    assert "eval_metrics" not in selected


def test_child_schedule_starts_at_u75_and_validates_only_u80_u90_u100() -> None:
    module = _module()
    from lunar_exploration_ppo.configs.stage6 import load_stage6_config

    transactions = module.build_planning_child_transactions(
        load_stage6_config(CONFIG)
    )

    assert len(transactions) == 26
    assert tuple(transaction.sequence for transaction in transactions) == tuple(
        range(26)
    )
    assert transactions[0].key == "0000:20260716:update:075"
    assert transactions[-1].key == "0025:20260716:update:100"
    assert transactions[0].key != module.DISCARDED_U75_TRANSACTION_KEY
    assert tuple(
        transaction.update
        for transaction in transactions
        if transaction.validation_episodes
    ) == (80, 90, 100)
    assert all(transaction.update != 74 for transaction in transactions)


def test_empty_child_root_commits_real_u75_transaction(tmp_path: Path) -> None:
    module = _module()
    from lunar_exploration_ppo.configs.stage6 import load_stage6_config
    from lunar_exploration_ppo.ppo.standard_training import CheckpointReceiptIndex
    from lunar_exploration_ppo.workflows.stage6 import Stage6StateJournal

    transactions = module.build_planning_child_transactions(
        load_stage6_config(CONFIG)
    )
    journal = Stage6StateJournal(tmp_path / "child/job-state.jsonl")
    receipts = CheckpointReceiptIndex(tmp_path / "child/checkpoints/index.jsonl")

    checkpoint = _commit_child_transaction(
        journal=journal,
        receipt_index=receipts,
        transactions=transactions,
        transaction=transactions[0],
    )

    assert checkpoint.transaction_key == "0000:20260716:update:075"
    assert tuple(row["state"] for row in journal.verify()) == (
        "seed_20260716_training_update_75",
    )


def test_child_u80_exact_resume_uses_child_local_prefix(tmp_path: Path) -> None:
    module = _module()
    from lunar_exploration_ppo.configs.stage6 import load_stage6_config
    from lunar_exploration_ppo.ppo.standard_training import (
        CheckpointReceiptIndex,
        reconcile_checkpointed_resume,
    )
    from lunar_exploration_ppo.workflows.stage6 import Stage6StateJournal

    transactions = module.build_planning_child_transactions(
        load_stage6_config(CONFIG)
    )
    journal = Stage6StateJournal(tmp_path / "child/job-state.jsonl")
    receipts = CheckpointReceiptIndex(tmp_path / "child/checkpoints/index.jsonl")
    checkpoint = None
    for transaction in transactions[:6]:
        checkpoint = _commit_child_transaction(
            journal=journal,
            receipt_index=receipts,
            transactions=transactions,
            transaction=transaction,
        )
    assert checkpoint is not None

    remaining, appended = reconcile_checkpointed_resume(
        transactions=transactions,
        journal=journal,
        checkpoint=checkpoint,
        bindings=_child_journal_bindings(checkpoint.checkpoint_sha256),
        receipt_rows=receipts.verify(),
    )

    assert appended == ()
    assert remaining[0].update == 81
    assert tuple(transaction.update for transaction in remaining) == tuple(
        range(81, 101)
    )


def test_child_partial_u80_transaction_is_completed_not_skipped(
    tmp_path: Path,
) -> None:
    module = _module()
    from lunar_exploration_ppo.configs.stage6 import load_stage6_config
    from lunar_exploration_ppo.ppo.standard_training import (
        CheckpointReceiptIndex,
        reconcile_checkpointed_resume,
    )
    from lunar_exploration_ppo.workflows.stage6 import Stage6StateJournal

    transactions = module.build_planning_child_transactions(
        load_stage6_config(CONFIG)
    )
    journal = Stage6StateJournal(tmp_path / "child/job-state.jsonl")
    receipts = CheckpointReceiptIndex(tmp_path / "child/checkpoints/index.jsonl")
    for transaction in transactions[:5]:
        _commit_child_transaction(
            journal=journal,
            receipt_index=receipts,
            transactions=transactions,
            transaction=transaction,
        )
    transaction = transactions[5]
    checkpoint = _transaction_checkpoint(transaction)
    assert receipts.append_once(checkpoint) is True
    bindings = _child_journal_bindings(checkpoint.checkpoint_sha256)
    journal.append(transaction.commit_states[0], bindings)

    remaining, appended = reconcile_checkpointed_resume(
        transactions=transactions,
        journal=journal,
        checkpoint=checkpoint,
        bindings=bindings,
        receipt_rows=receipts.verify(),
    )

    assert tuple(row["state"] for row in appended) == (
        "seed_20260716_validating_update_80",
    )
    assert remaining[0].update == 81


def test_child_u100_real_commit_completes_exact_child_schedule(
    tmp_path: Path,
) -> None:
    module = _module()
    from lunar_exploration_ppo.configs.stage6 import load_stage6_config
    from lunar_exploration_ppo.ppo.standard_training import (
        CheckpointReceiptIndex,
        completed_transaction_keys_from_journal,
    )
    from lunar_exploration_ppo.workflows.stage6 import Stage6StateJournal

    transactions = module.build_planning_child_transactions(
        load_stage6_config(CONFIG)
    )
    journal = Stage6StateJournal(tmp_path / "child/job-state.jsonl")
    receipts = CheckpointReceiptIndex(tmp_path / "child/checkpoints/index.jsonl")
    for transaction in transactions:
        _commit_child_transaction(
            journal=journal,
            receipt_index=receipts,
            transactions=transactions,
            transaction=transaction,
        )

    assert len(receipts.verify()) == 26
    assert completed_transaction_keys_from_journal(
        journal.verify(),
        transactions,
    ) == tuple(transaction.key for transaction in transactions)
    assert journal.verify()[-1]["state"] == "seed_20260716_complete"


def test_parent_u75_receipt_is_not_a_completed_child_transaction(
    tmp_path: Path,
) -> None:
    module = _module()
    from lunar_exploration_ppo.configs.stage6 import load_stage6_config
    from lunar_exploration_ppo.ppo.standard_training import (
        CheckpointReceiptIndex,
        StandardTransactionCheckpoint,
        StandardTrainingError,
        commit_checkpointed_transaction,
    )
    from lunar_exploration_ppo.workflows.stage6 import Stage6StateJournal

    transactions = module.build_planning_child_transactions(
        load_stage6_config(CONFIG)
    )
    journal = Stage6StateJournal(tmp_path / "child/job-state.jsonl")
    receipts = CheckpointReceiptIndex(tmp_path / "child/checkpoints/index.jsonl")
    parent_receipt = StandardTransactionCheckpoint(
        transaction_key=module.DISCARDED_U75_TRANSACTION_KEY,
        seed=20260716,
        update=75,
        checkpoint_sha256="1" * 64,
        complete_marker_sha256="2" * 64,
        policy_state_sha256="3" * 64,
    )
    assert receipts.append_once(parent_receipt) is True
    child_checkpoint = _transaction_checkpoint(transactions[0])

    with pytest.raises(StandardTrainingError, match="exact prefix"):
        commit_checkpointed_transaction(
            journal=journal,
            transactions=transactions,
            transaction=transactions[0],
            checkpoint=child_checkpoint,
            bindings=_child_journal_bindings(
                child_checkpoint.checkpoint_sha256
            ),
            receipt_rows=receipts.verify(),
        )


def test_planning_child_acceptance_profile_drives_26_receipt_reports() -> None:
    module = _module()
    from lunar_exploration_ppo.ppo.standard_training import (
        build_stage6_acceptance_artifacts,
    )

    profile = module.build_planning_child_acceptance_profile()
    immutable = _child_acceptance_immutable("1" * 64)
    immutable.pop("checkpoint_sha256")
    artifacts = build_stage6_acceptance_artifacts(
        global_best={
            "seed": 20260716,
            "update": 100,
            "success_rate_under_fixed_step_budget": 0.5,
            "mean_final_coverage": 0.9,
            "checkpoint_ref": "seed-20260716/update-00000100",
        },
        performance_advantage_established=False,
        performance_claim="performance advantage not established",
        ppo_ci95_low=0.1,
        gain_over_cost_ci95_high=0.2,
        final_evaluation_count=10,
        final_episode_count=640,
        checkpoint_receipt_count=26,
        immutable_bindings=immutable,
        acceptance_profile=profile,
    )

    assert profile["child_updates"] == list(range(75, 101))
    assert profile["child_validation_updates"] == [80, 90, 100]
    assert artifacts["summary"]["parent_semantics_update_count"] == 74
    assert artifacts["summary"]["child_new_semantics_update_count"] == 26
    assert artifacts["routing"]["parent_semantics_update_count"] == 74
    assert artifacts["routing"]["child_new_semantics_update_count"] == 26
    for report in artifacts["reports"].values():
        assert (
            "74 parent-semantics updates + 26 child new-semantics updates"
            in report.decode("utf-8")
        )
        assert "100 child new-semantics updates" not in report.decode("utf-8")


def test_stage6_acceptance_reports_warm_start_and_child_repair_separately() -> None:
    module = _module()
    from lunar_exploration_ppo.ppo.standard_training import (
        build_stage6_acceptance_artifacts,
    )

    immutable = _child_acceptance_immutable("1" * 64)
    immutable.pop("checkpoint_sha256")
    recovery_acceptance = {
        "schema_version": (
            "stage6_planning_child_recovery_capability_acceptance/v1"
        ),
        "input_snapshot_sha256": "2" * 64,
        "parent_artifact_sha256": "3" * 64,
        "continuation_artifact_sha256": "4" * 64,
        "current_verified_review_authorization": {},
        "terminal_complete": False,
        "resume_cursor": {"next_update": 85},
        "lineage_epochs": [{}],
        "journal_binding_epochs": [{}],
        "journal_prefixes": {},
    }
    child_binding = {
        "schema_version": (
            "stage6_planning_child_recovery_semantic_binding/v1"
        ),
        "capability_sha256": "1" * 64,
        "input_snapshot_sha256": "2" * 64,
        "parent_artifact_sha256": "3" * 64,
        "continuation_artifact_sha256": "4" * 64,
        "acceptance_binding": recovery_acceptance,
    }

    artifacts = build_stage6_acceptance_artifacts(
        global_best={
            "seed": 20260716,
            "update": 100,
            "success_rate_under_fixed_step_budget": 0.5,
            "mean_final_coverage": 0.9,
            "checkpoint_ref": "seed-20260716/update-00000100",
        },
        performance_advantage_established=False,
        performance_claim="performance advantage not established",
        ppo_ci95_low=0.1,
        gain_over_cost_ci95_high=0.2,
        final_evaluation_count=10,
        final_episode_count=640,
        checkpoint_receipt_count=26,
        immutable_bindings=immutable,
        acceptance_profile=module.build_planning_child_acceptance_profile(),
        planning_child_source_repair_binding=child_binding,
    )

    for section in ("summary", "routing"):
        assert artifacts[section]["acceptance_profile"] == (
            module.build_planning_child_acceptance_profile()
        )
        assert (
            artifacts[section]["planning_child_recovery"]
            == child_binding
        )
        assert "source_repair" not in artifacts[section]


def test_stage6_child_repair_acceptance_rejects_classic_source_repair() -> None:
    module = _module()
    from lunar_exploration_ppo.ppo.standard_training import (
        StandardTrainingError,
        build_stage6_acceptance_artifacts,
    )

    immutable = _child_acceptance_immutable("1" * 64)
    immutable.pop("checkpoint_sha256")
    with pytest.raises(
        StandardTrainingError,
        match="child source-repair.*classic",
    ):
        build_stage6_acceptance_artifacts(
            global_best={
                "seed": 20260716,
                "update": 100,
                "success_rate_under_fixed_step_budget": 0.5,
                "mean_final_coverage": 0.9,
                "checkpoint_ref": "seed-20260716/update-00000100",
            },
            performance_advantage_established=False,
            performance_claim="performance advantage not established",
            ppo_ci95_low=0.1,
            gain_over_cost_ci95_high=0.2,
            final_evaluation_count=10,
            final_episode_count=640,
            checkpoint_receipt_count=26,
            immutable_bindings=immutable,
            acceptance_profile=module.build_planning_child_acceptance_profile(),
            source_repair_binding={"schema_version": "invalid"},
            planning_child_source_repair_binding={
                "schema_version": (
                    "stage6_planning_child_source_repair_acceptance/v1"
                )
            },
        )


@pytest.mark.parametrize("receipt_count", [25, 27])
def test_planning_child_acceptance_profile_rejects_wrong_receipt_count(
    receipt_count: int,
) -> None:
    module = _module()
    from lunar_exploration_ppo.ppo.standard_training import (
        StandardTrainingError,
        build_stage6_acceptance_artifacts,
    )

    immutable = _child_acceptance_immutable("1" * 64)
    immutable.pop("checkpoint_sha256")
    with pytest.raises(StandardTrainingError, match="acceptance payload"):
        build_stage6_acceptance_artifacts(
            global_best={
                "seed": 20260716,
                "update": 100,
                "success_rate_under_fixed_step_budget": 0.5,
                "mean_final_coverage": 0.9,
                "checkpoint_ref": "seed-20260716/update-00000100",
            },
            performance_advantage_established=False,
            performance_claim="performance advantage not established",
            ppo_ci95_low=0.1,
            gain_over_cost_ci95_high=0.2,
            final_evaluation_count=10,
            final_episode_count=640,
            checkpoint_receipt_count=receipt_count,
            immutable_bindings=immutable,
            acceptance_profile=module.build_planning_child_acceptance_profile(),
        )


def test_planning_child_terminal_replay_accepts_exact_26_receipts(
    tmp_path: Path,
) -> None:
    module = _module()
    from lunar_exploration_ppo.configs.stage6 import load_stage6_config
    from lunar_exploration_ppo.ppo.standard_training import CheckpointReceiptIndex
    from lunar_exploration_ppo.workflows.stage6 import Stage6StateJournal

    config = load_stage6_config(CONFIG)
    transactions = module.build_planning_child_transactions(config)
    journal = Stage6StateJournal(tmp_path / "child/job-state.jsonl")
    receipts = CheckpointReceiptIndex(tmp_path / "child/checkpoints/index.jsonl")
    for transaction in transactions:
        _commit_child_transaction(
            journal=journal,
            receipt_index=receipts,
            transactions=transactions,
            transaction=transaction,
        )
    immutable = _child_journal_bindings("1" * 64)
    immutable.pop("checkpoint_sha256")

    result = module.verify_planning_child_terminal_replay(
        config=config,
        receipt_rows=receipts.verify(),
        journal_rows=journal.verify(),
        immutable_bindings=immutable,
        validation_transaction_keys=tuple(
            transaction.key
            for transaction in transactions
            if transaction.validation_episodes
        ),
        audit_transaction_keys=tuple(
            transaction.key
            for transaction in transactions
            if transaction.update == 100
        ),
        acceptance_profile=module.build_planning_child_acceptance_profile(),
    )

    assert result["passed"] is True
    assert result["parent_semantics_update_count"] == 74
    assert result["child_new_semantics_update_count"] == 26
    assert result["last_child_update"] == 100


@pytest.mark.parametrize(
    "case",
    [
        "25_receipts",
        "27_receipts",
        "missing_u75",
        "missing_u100",
        "mixed_parent_receipt",
        "extra_validation",
    ],
)
def test_planning_child_terminal_replay_fails_closed(
    case: str,
    tmp_path: Path,
) -> None:
    module = _module()
    from lunar_exploration_ppo.configs.stage6 import load_stage6_config
    from lunar_exploration_ppo.ppo.standard_training import (
        CheckpointReceiptIndex,
        StandardTransactionCheckpoint,
    )
    from lunar_exploration_ppo.workflows.stage6 import Stage6StateJournal

    config = load_stage6_config(CONFIG)
    transactions = module.build_planning_child_transactions(config)
    journal = Stage6StateJournal(tmp_path / "journal/job-state.jsonl")
    committed_receipts = CheckpointReceiptIndex(
        tmp_path / "journal/checkpoints/index.jsonl"
    )
    for transaction in transactions:
        _commit_child_transaction(
            journal=journal,
            receipt_index=committed_receipts,
            transactions=transactions,
            transaction=transaction,
        )
    selected = transactions
    if case == "25_receipts":
        selected = transactions[:12] + transactions[13:]
    elif case == "missing_u75":
        selected = transactions[1:]
    elif case == "missing_u100":
        selected = transactions[:-1]
    receipt_rows = _receipt_rows_for_transactions(
        tmp_path / case / "index.jsonl",
        selected,
    )
    if case == "27_receipts":
        index = CheckpointReceiptIndex(tmp_path / case / "index.jsonl")
        extra = StandardTransactionCheckpoint(
            transaction_key="0026:20260716:update:100",
            seed=20260716,
            update=100,
            checkpoint_sha256="4" * 64,
            complete_marker_sha256="5" * 64,
            policy_state_sha256="6" * 64,
        )
        assert index.append_once(extra) is True
        receipt_rows = index.verify()
    elif case == "mixed_parent_receipt":
        index = CheckpointReceiptIndex(tmp_path / case / "mixed-index.jsonl")
        parent = StandardTransactionCheckpoint(
            transaction_key=module.DISCARDED_U75_TRANSACTION_KEY,
            seed=20260716,
            update=75,
            checkpoint_sha256="7" * 64,
            complete_marker_sha256="8" * 64,
            policy_state_sha256="9" * 64,
        )
        assert index.append_once(parent) is True
        for transaction in transactions[1:]:
            assert index.append_once(_transaction_checkpoint(transaction)) is True
        receipt_rows = index.verify()
    validation_keys = tuple(
        transaction.key
        for transaction in transactions
        if transaction.validation_episodes
    )
    if case == "extra_validation":
        validation_keys = (*validation_keys, transactions[0].key)
    immutable = _child_journal_bindings("1" * 64)
    immutable.pop("checkpoint_sha256")

    with pytest.raises(
        module.Stage6PlanningWarmStartError,
        match="terminal",
    ):
        module.verify_planning_child_terminal_replay(
            config=config,
            receipt_rows=receipt_rows,
            journal_rows=journal.verify(),
            immutable_bindings=immutable,
            validation_transaction_keys=validation_keys,
            audit_transaction_keys=(transactions[-1].key,),
            acceptance_profile=module.build_planning_child_acceptance_profile(),
        )


def test_training_schedule_uses_child_transactions_and_warm_runtime_contract() -> None:
    module = _module()
    from lunar_exploration_ppo.configs.stage6 import load_stage6_config
    from lunar_exploration_ppo.ppo import standard_training

    config = load_stage6_config(CONFIG)

    class Backend:
        def __init__(self) -> None:
            self.transactions = module.build_planning_child_transactions(config)
            self.updates: list[int] = []

        def prepare_seed(self, seed, transactions):
            assert tuple(transactions) == self.transactions
            return {
                "seed": seed,
                "policy": object(),
                "optimizer": object(),
                "initial_policy_sha256": module.PARENT_POLICY_STATE_SHA256,
                "optimizer_fresh": False,
                "independent_seed_origin": False,
                "continue_from_current_state": False,
                "planning_warm_start": True,
                "parent_update": 74,
                "fresh_vector_env_reset_required": True,
                "sampler_seed_derivation_version": (
                    "stage6_seed_times_16_plus_lane/v1"
                ),
                "sampler_seeds": standard_training.derive_standard_sampler_seeds(
                    seed
                ),
                "runtime_token": "planning-warm-start",
            }, transactions

        def run_update(self, runtime, transaction):
            assert runtime["continue_from_current_state"] is False
            self.updates.append(transaction.update)

        def finish_seed(self, runtime):
            return standard_training.ValidationRecord(
                runtime["seed"], 100, 0.5, 0.6, "update-00000100"
            )

        def freeze_global_best(self, record):
            return record

        def checkpoint_write_count(self):
            return len(self.updates)

        def run_final_evaluation(self, record, split, method):
            return {"episode_count": 64}

        def finalize(self, record, evaluations):
            return {"updates": tuple(self.updates)}

    backend = Backend()
    result = standard_training.run_standard_training_schedule(config, backend)

    assert result["updates"] == tuple(range(75, 101))


@pytest.mark.parametrize(
    ("update", "expected_periodic_updates"),
    ((75, ()), (100, (100,))),
)
def test_warm_start_checkpoint_retention_preserves_schedule_and_capacity(
    tmp_path: Path,
    update: int,
    expected_periodic_updates: tuple[int, ...],
) -> None:
    from lunar_exploration_ppo.configs.stage6 import load_stage6_config
    from lunar_exploration_ppo.ppo import standard_training

    config = load_stage6_config(CONFIG)
    calls: list[dict[str, object]] = []

    class Retention:
        def __init__(self, root: Path, *, schema_version: str) -> None:
            del root, schema_version

        def apply(self, **kwargs: object) -> object:
            calls.append(dict(kwargs))
            return SimpleNamespace(kept_updates=(), removed_updates=())

    backend = object.__new__(standard_training.StandardProductionBackend)
    backend.config = config
    backend._planning_warm_start = object()
    backend._source_repair = None
    backend._components = {"CheckpointRetentionManager": Retention}
    backend._execution_operation = lambda label: nullcontext(label)
    backend._require_execution_capability_current = (
        lambda *args, **kwargs: None
    )

    backend._apply_checkpoint_retention(
        checkpoint_root=tmp_path / "checkpoints",
        update=update,
        best_record={"update": update},
    )

    assert calls == [
        {
            "latest_update": update,
            "periodic_updates": expected_periodic_updates,
            "best_update": update,
            "periodic_keep_count": config.training.periodic_keep_count,
        }
    ]


def test_warm_start_creation_script_builds_canonical_child_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = importlib.util.spec_from_file_location("stage6_warm_start_creator", SCRIPT)
    assert spec is not None and spec.loader is not None
    creator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(creator)
    monkeypatch.setattr(
        creator,
        "verify_parent_u74_bundle",
        lambda path: SimpleNamespace(
            resource_accepted=True,
            u75_attempt1_discarded=True,
        ),
    )
    output = tmp_path / "planning-warm-start.json"
    authorization = tmp_path / "launch-authorization.json"
    authorization.write_bytes(b'{"authorized":true}\n')
    hashes = {
        "design_sha256": "b" * 64,
        "plan_sha256": "c" * 64,
        "source_set_sha256": "d" * 64,
        "spec_review_sha256": "e" * 64,
        "quality_review_sha256": "f" * 64,
        "launch_authorization_sha256": "1" * 64,
    }
    effective = _module().planning_effective_config_bytes(CONFIG.read_bytes())
    review_handle = SimpleNamespace(
        canonical_record=lambda: {
            "source_set_sha256": hashes["source_set_sha256"],
            "spec_review_sha256": hashes["spec_review_sha256"],
            "quality_review_sha256": hashes["quality_review_sha256"],
            "authorization_file_sha256": hashes[
                "launch_authorization_sha256"
            ],
        },
        require_current=lambda label: None,
    )
    authorization_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        creator,
        "verify_stage6_review_launch_authorization",
        lambda **kwargs: (
            authorization_calls.append(dict(kwargs)) or review_handle
        ),
        raising=False,
    )
    monkeypatch.setattr(
        creator,
        "planning_warm_start_expected_bindings",
        lambda **kwargs: hashes,
        raising=False,
    )

    artifact = creator.create_planning_warm_start_artifact(
        output_path=output,
        parent_stage_root=PARENT_STAGE_ROOT,
        child_run_id="s6-planning-u74-child-r1",
        repo_root=ROOT,
        config_path=CONFIG,
        review_authorization_path=authorization,
    )

    assert "child_effective_config_sha256" not in inspect.signature(
        creator.create_planning_warm_start_artifact
    ).parameters
    assert not set(hashes).intersection(
        inspect.signature(
            creator.create_planning_warm_start_artifact
        ).parameters
    )
    assert authorization_calls == [
        {
            "authorization_path": authorization,
            "repo_root": ROOT,
            "config_path": CONFIG,
            "expected_run_id": "s6-planning-u74-child-r1",
            "effective_config_bytes": effective,
        }
    ]
    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert output.read_bytes() == (
        json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _module().validate_planning_warm_start_artifact(
        artifact,
        expected_child_run_id="s6-planning-u74-child-r1",
        expected_child_config_bytes=effective,
        expected_bindings=hashes,
    )

    existing_payload = output.read_bytes()
    with pytest.raises(FileExistsError):
        creator.create_planning_warm_start_artifact(
            output_path=output,
            parent_stage_root=PARENT_STAGE_ROOT,
            child_run_id="s6-planning-u74-child-r1",
            repo_root=ROOT,
            config_path=CONFIG,
            review_authorization_path=authorization,
        )
    assert output.read_bytes() == existing_payload

    failed_output = tmp_path / "failed-planning-warm-start.json"

    class FailingArtifactStore:
        def __init__(self, root: str | Path) -> None:
            self.root = Path(root)

        def write_bytes_exclusive(
            self,
            relative_path: str | Path,
            payload: bytes,
        ) -> Path:
            del relative_path, payload
            raise OSError("injected exclusive publication failure")

    monkeypatch.setattr(
        creator,
        "ArtifactStore",
        FailingArtifactStore,
        raising=False,
    )
    with pytest.raises(OSError, match="exclusive publication failure"):
        creator.create_planning_warm_start_artifact(
            output_path=failed_output,
            parent_stage_root=PARENT_STAGE_ROOT,
            child_run_id="s6-planning-u74-child-r1",
            repo_root=ROOT,
            config_path=CONFIG,
            review_authorization_path=authorization,
        )
    assert not failed_output.exists()


@pytest.mark.parametrize("resume", (False, True))
def test_warm_workflow_passes_exact_effective_config_to_preflight_and_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    resume: bool,
) -> None:
    module = _module()
    from lunar_exploration_ppo.configs.stage6 import load_stage6_config
    from lunar_exploration_ppo.ppo import standard_training
    from lunar_exploration_ppo.workflows import stage6 as stage6_workflow
    from lunar_exploration_ppo.workflows import stage6_terminal_recovery
    from test_stage6_workflow import (
        FORMAL_STAGE6_RUN_ID,
        _active_execution_capability,
    )

    effective = module.planning_effective_config_bytes(CONFIG.read_bytes())
    artifact = _artifact()
    artifact["child"]["run_id"] = FORMAL_STAGE6_RUN_ID  # type: ignore[index]
    artifact["child"]["effective_config_sha256"] = hashlib.sha256(  # type: ignore[index]
        effective
    ).hexdigest()
    warm_path = tmp_path / "planning-warm-start.json"
    warm_path.write_bytes(
        (
            json.dumps(
                artifact,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    )
    context, artifact_sha256 = module.load_planning_warm_start_artifact(
        warm_path,
        expected_child_run_id=FORMAL_STAGE6_RUN_ID,
        expected_child_config_bytes=effective,
    )
    parent = module.VerifiedParentU74(
        parent_update=module.PARENT_UPDATE,
        checkpoint_sha256=module.PARENT_CHECKPOINT_SHA256,
        manifest_sha256=module.PARENT_MANIFEST_SHA256,
        complete_sha256=module.PARENT_COMPLETE_SHA256,
        policy_state_sha256=module.PARENT_POLICY_STATE_SHA256,
        config_sha256=module.PARENT_CONFIG_SHA256,
        lineage_sha256=module.PARENT_LINEAGE_SHA256,
        resource_accepted=True,
        u75_attempt1_discarded=True,
        checkpoint_payload={},
    )
    preflight_calls: list[dict[str, object]] = []
    training_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        stage6_terminal_recovery,
        "detect_stage6_terminal_recovery",
        lambda **kwargs: {"status": "no_terminal", "reason": ""},
    )

    def preflight(**kwargs: object) -> dict[str, object]:
        preflight_calls.append(dict(kwargs))
        stage = Path(kwargs["base_output_root"]) / str(kwargs["run_id"]) / "s6"
        (stage / "preflight").mkdir(parents=True, exist_ok=True)
        (stage / "preflight/audit.json").write_bytes(b"{}\n")
        return {"passed": True}

    monkeypatch.setattr(
        stage6_workflow,
        (
            "_verify_stage6_machine_preflight_for_resume_at_root"
            if resume
            else "_run_stage6_machine_preflight"
        ),
        preflight,
    )

    def execute(**kwargs: object) -> dict[str, object]:
        training_calls.append(dict(kwargs))
        stage = Path(kwargs["run_root"]) / "s6"
        stage.mkdir(parents=True, exist_ok=True)
        return {
            "stage_root": str(stage),
            "summary": {"state": "training"},
            "routing": {"route": "resume"},
        }

    monkeypatch.setattr(
        standard_training,
        "execute_standard_training",
        execute,
    )
    with _active_execution_capability(
        tmp_path / "authority",
        monkeypatch,
        effective_config_bytes=effective,
        planning_warm_start_path=warm_path,
    ) as active:
        if resume:
            preflight_root = Path(active["run_root"]) / "s6/preflight"
            preflight_root.mkdir(parents=True)
            (preflight_root / "audit.json").write_bytes(b"{}\n")
        result = stage6_workflow._run_stage6_workflow_locked(
            config=load_stage6_config(CONFIG),
            config_path=CONFIG,
            run_id=FORMAL_STAGE6_RUN_ID,
            stage5_gate_path=tmp_path / "gate.json",
            base=Path(active["run_root"]).parent,
            run_root=Path(active["run_root"]),
            repo=ROOT,
            run_preexisted=resume,
            review_authorization_handle=active["review_handle"],
            stage5_authority=active["stage5_handle"],
            input_pin=active["pin"],
            execution_identity=active["identity"],
            execution_capability=active["capability"],
            run_lease=active["lease"],
            planning_warm_start_context=context,
            planning_warm_start_sha256=artifact_sha256,
            verified_parent_u74=parent,
        )

    assert result.stage_root == Path(active["run_root"]) / "s6"
    assert len(preflight_calls) == 1
    assert preflight_calls[0]["effective_config_bytes"] == effective
    assert len(training_calls) == 1
    assert training_calls[0]["planning_warm_start_context"] is context
    assert (
        training_calls[0]["planning_warm_start_sha256"]
        == artifact_sha256
    )


def test_final_machine_verifier_delegates_warm_transaction_graph_to_child_replay() -> None:
    from lunar_exploration_ppo.workflows import stage6 as stage6_workflow

    source = inspect.getsource(
        stage6_workflow._verify_stage6_machine_acceptance_from_bound_graph
    )

    assert "verify_planning_child_terminal_replay(" in source
    assert "require_dual_scan=planning_warm_start" in source


def test_formal_runner_requires_and_forwards_planning_warm_start(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    spec = importlib.util.spec_from_file_location("stage6_warm_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        runner,
        "run_stage6_workflow",
        lambda **kwargs: (
            calls.append(kwargs)
            or SimpleNamespace(
                run_id=kwargs["run_id"],
                stage_root=tmp_path / "s6",
                summary={},
                routing={},
            )
        ),
    )
    warm = tmp_path / "planning-warm-start.json"
    child_repair = tmp_path / "planning-child-source-repair.json"

    assert runner.main(
        [
            "--run-id",
            "s6-standard-single-r1-20260723T010203Z",
            "--review-authorization",
            str(tmp_path / "authorization.json"),
            "--planning-warm-start",
            str(warm),
            "--planning-child-source-repair",
            str(child_repair),
        ]
    ) == 0
    assert calls[0]["planning_warm_start_path"] == warm
    assert (
        calls[0]["planning_child_source_repair_path"]
        == child_repair
    )
    action = next(
        item
        for item in runner.build_parser()._actions
        if "--planning-warm-start" in item.option_strings
    )
    assert action.required is True
    child_action = next(
        item
        for item in runner.build_parser()._actions
        if "--planning-child-source-repair" in item.option_strings
    )
    assert child_action.required is False


@pytest.mark.skipif(
    not PARENT_STAGE_ROOT.is_dir(),
    reason="frozen parent U74 stage root is unavailable",
)
def test_parent_u74_job_state_replays_exact_74_transaction_prefix() -> None:
    module = _module()

    result = module.verify_parent_u74_journal_snapshots(
        config_bytes=(PARENT_STAGE_ROOT / "config.json").read_bytes(),
        job_state_bytes=(PARENT_STAGE_ROOT / "job-state.jsonl").read_bytes(),
        receipt_index_bytes=(
            PARENT_STAGE_ROOT / "checkpoints/index.jsonl"
        ).read_bytes(),
        lineage_audit_bytes=(
            PARENT_STAGE_ROOT / "lineage_audit.json"
        ).read_bytes(),
        source_repair_ordinal6_bytes=(
            PARENT_STAGE_ROOT
            / "source-repair-sensor-acceleration.json"
        ).read_bytes(),
    )

    assert result["completed_transaction_count"] == 74
    assert result["receipt_count"] == 74
    assert result["last_transaction_key"] == module.PARENT_RECEIPT_KEY
    assert result["next_update"] == 75


@pytest.mark.skipif(
    not PARENT_STAGE_ROOT.is_dir(),
    reason="frozen parent U74 stage root is unavailable",
)
@pytest.mark.parametrize(
    "case",
    [
        "truncated_u74",
        "appended_u75",
        "last_state",
        "immutable",
        "checkpoint",
    ],
)
def test_parent_u74_job_state_rejects_prefix_and_binding_drift(
    case: str,
) -> None:
    module = _module()
    rows = [
        json.loads(line)
        for line in (
            PARENT_STAGE_ROOT / "job-state.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    if case == "truncated_u74":
        rows.pop()
    elif case == "appended_u75":
        appended = copy.deepcopy(rows[-1])
        appended["state"] = "seed_20260716_training_update_75"
        appended["bindings"]["checkpoint_sha256"] = "f" * 64
        rows.append(appended)
    elif case == "last_state":
        rows[-1]["state"] = "seed_20260716_training_update_73"
    elif case == "immutable":
        rows[-1]["bindings"]["config_sha256"] = "f" * 64
    elif case == "checkpoint":
        rows[-1]["bindings"]["checkpoint_sha256"] = "f" * 64
    rows = _rehash_journal_rows(rows)

    with pytest.raises(
        module.Stage6PlanningWarmStartError,
        match="parent U74 journal",
    ):
        module.verify_parent_u74_journal_snapshots(
            config_bytes=(PARENT_STAGE_ROOT / "config.json").read_bytes(),
            job_state_bytes=_canonical_jsonl_bytes(rows),
            receipt_index_bytes=(
                PARENT_STAGE_ROOT / "checkpoints/index.jsonl"
            ).read_bytes(),
            lineage_audit_bytes=(
                PARENT_STAGE_ROOT / "lineage_audit.json"
            ).read_bytes(),
            source_repair_ordinal6_bytes=(
                PARENT_STAGE_ROOT
                / "source-repair-sensor-acceleration.json"
            ).read_bytes(),
        )


def test_unpinned_parent_capture_uses_secure_descriptor_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    from lunar_exploration_ppo.utils import path_security

    stage = tmp_path / "s6"
    stage.mkdir()
    member = stage / "checkpoint.pt"
    member.write_bytes(b"immutable-parent-snapshot")
    label = "planning-warm-start:parent:checkpoint"
    monkeypatch.setattr(
        module,
        "parent_u74_input_pin_requests",
        lambda _parent_stage_root: ((label, member),),
    )

    secure_calls: list[tuple[Path, Path, str]] = []

    def observed_secure_read(
        path: str | Path,
        *,
        base: str | Path | None = None,
        label: str = "artifact file",
    ):
        secure_calls.append((Path(path), Path(base), label))
        return path_security.secure_read_bytes(
            path,
            base=base,
            label=label,
        )

    monkeypatch.setattr(
        module,
        "secure_read_bytes",
        observed_secure_read,
        raising=False,
    )

    def reject_naked_read(_path: Path) -> bytes:
        raise AssertionError("naked Path.read_bytes is forbidden")

    monkeypatch.setattr(Path, "read_bytes", reject_naked_read)

    assert module._capture_parent_u74_inputs(
        stage,
        input_pin=None,
    ) == {label: b"immutable-parent-snapshot"}
    assert secure_calls == [(member, stage, label)]


@pytest.mark.skipif(
    not PARENT_STAGE_ROOT.is_dir(),
    reason="frozen parent U74 stage root is unavailable",
)
def test_actual_parent_u74_bundle_verification_is_read_only_and_fail_closed() -> None:
    module = _module()
    members = (
        PARENT_STAGE_ROOT / "checkpoints/seed-20260716/update-00000074/checkpoint.pt",
        PARENT_STAGE_ROOT / "checkpoints/seed-20260716/update-00000074/manifest.json",
        PARENT_STAGE_ROOT / "checkpoints/seed-20260716/update-00000074/complete.json",
        PARENT_STAGE_ROOT / "checkpoints/index.jsonl",
        PARENT_STAGE_ROOT / "job-state.jsonl",
        PARENT_STAGE_ROOT / "resource_audit.jsonl",
        PARENT_STAGE_ROOT / "source-repair-sensor-acceleration.json",
    )
    before = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in members
    }

    verified = module.verify_parent_u74_bundle(PARENT_STAGE_ROOT)

    assert verified.parent_update == 74
    assert verified.checkpoint_sha256 == module.PARENT_CHECKPOINT_SHA256
    assert verified.manifest_sha256 == module.PARENT_MANIFEST_SHA256
    assert verified.complete_sha256 == module.PARENT_COMPLETE_SHA256
    assert hashlib.sha256(verified.checkpoint_bytes).hexdigest() == (
        module.PARENT_CHECKPOINT_SHA256
    )
    assert hashlib.sha256(verified.manifest_bytes).hexdigest() == (
        module.PARENT_MANIFEST_SHA256
    )
    assert hashlib.sha256(verified.complete_bytes).hexdigest() == (
        module.PARENT_COMPLETE_SHA256
    )
    assert verified.resource_accepted is True
    assert verified.u75_attempt1_discarded is True
    assert {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in members
    } == before


@pytest.mark.skipif(
    not PARENT_STAGE_ROOT.is_dir(),
    reason="frozen parent U74 stage root is unavailable",
)
def test_parent_u74_runtime_restore_consumes_verified_snapshot_without_rediscovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    from lunar_exploration_ppo.ppo import checkpoint as checkpoint_module
    from lunar_exploration_ppo.policy.cross_attention import (
        CrossAttentionFrontierPolicy,
    )

    verified = module.verify_parent_u74_bundle(PARENT_STAGE_ROOT)
    policy = CrossAttentionFrontierPolicy().to(device="cpu", dtype=torch.float32)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=3.0e-4)

    def reject_path_rediscovery(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("parent path was rediscovered after snapshot capture")

    monkeypatch.setattr(
        module,
        "verify_parent_u74_bundle",
        reject_path_rediscovery,
    )
    monkeypatch.setattr(module, "_sha256", reject_path_rediscovery)
    monkeypatch.setattr(
        checkpoint_module,
        "_restore_rng_state",
        lambda _state: None,
    )

    state = module.restore_parent_u74_runtime(
        verified_parent=verified,
        policy=policy,
        optimizer=optimizer,
    )

    assert state.update_step == 74
    assert state.policy_state_sha256 == module.PARENT_POLICY_STATE_SHA256
    assert state.model_state_restored is True


@pytest.mark.skipif(
    not PARENT_STAGE_ROOT.is_dir() or not torch.cuda.is_available(),
    reason="frozen parent U74 or CUDA is unavailable",
)
def test_actual_parent_u74_runtime_restores_only_approved_state() -> None:
    module = _module()
    from lunar_exploration_ppo.configs.stage6 import load_stage6_config
    from lunar_exploration_ppo.workflows.stage6 import (
        CANONICAL_STAGE4_CHECKPOINT,
        load_stage4_policy_for_standard,
    )

    config = load_stage6_config(CONFIG)
    policy = load_stage4_policy_for_standard(
        checkpoint_path=CANONICAL_STAGE4_CHECKPOINT,
        checkpoint_sha256=config.stage5_authority.checkpoint_sha256,
        policy_state_sha256=config.stage5_authority.policy_state_sha256,
        device=config.device,
    )
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=config.ppo.learning_rate,
        eps=config.ppo.adam_eps,
        weight_decay=config.ppo.weight_decay,
    )

    verified = module.verify_parent_u74_bundle(PARENT_STAGE_ROOT)
    state = module.restore_parent_u74_runtime(
        verified_parent=verified,
        policy=policy,
        optimizer=optimizer,
    )

    assert state.update_step == 74
    assert state.policy_state_sha256 == module.PARENT_POLICY_STATE_SHA256
    assert len(state.scenario_sampler_state["workers"]) == 8
    assert state.model_state_restored is True
    assert state.optimizer_state_restored is True
    assert state.rng_state_restored is True
    assert state.parent_vector_env_states_discarded is True
    assert state.parent_best_record_discarded is True
    assert state.parent_eval_metrics_discarded is True
