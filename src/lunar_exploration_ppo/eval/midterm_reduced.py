"""中期双门槛 G1 的显式 24 场景只读评价适配层。"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import struct
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, stdev
from types import MappingProxyType
from typing import Literal

import numpy as np
import torch
from torch import nn

from lunar_exploration_ppo.configs.stage6 import SafetyContract
from lunar_exploration_ppo.env.scenario_catalog import (
    StandardScenarioCatalog,
    StandardScenarioFactory,
)
from lunar_exploration_ppo.eval.standard import (
    STANDARD_EVALUATION_WORKERS,
    StandardEvaluationExecution,
    StandardEvaluationJob,
    partition_standard_evaluation_jobs,
    run_standard_evaluation_jobs,
)
from lunar_exploration_ppo.utils.artifact_io import ArtifactStore
from lunar_exploration_ppo.utils.path_security import (
    PathSecurityError,
    require_plain_path,
    secure_read_bytes,
)
from lunar_exploration_ppo.workflows.stage6 import (
    load_stage4_policy_for_standard,
)


SCALE_PROFILE = "midterm_reduced_w8x3_update80/v1"
UPDATE80_CHECKPOINT_PATH = (
    "D:/xunce/out/ppo_frontier/"
    "s6-standard-single-r1-20260724T000124Z/s6/"
    "checkpoints/seed-20260716/update-00000080/checkpoint.pt"
)
UPDATE80_CHECKPOINT_SHA256 = (
    "35e04c86f9f973af028fb08f0175d42ab45378d09e1f2b96ee6aad6e4c12b5b5"
)
UPDATE80_POLICY_STATE_SHA256 = (
    "3123e6adde99be41e3bd5cc2f3843068e5416892395926d759c2c6a243ffd381"
)
FROZEN_MANIFEST_SCHEMA = "mid-dual-scenario-freeze/v1"
DENOMINATOR_SOURCE = "reachable_observable_free_highres_cells/v1"
DENOMINATOR_ALGORITHM = "exact_reachable_safe_pose_range_los/v1"
_TRACE_SUMMARY_SCHEMA = "midterm-reduced-trace-summary/v1"
_COHORT_SIZES = MappingProxyType(
    {
        "test_q24": 24,
        "test_c24": 24,
        "unseen24": 24,
        "g3_test_q5": 5,
        "g3_unseen5": 5,
        "validation3": 3,
        "replay3": 3,
    }
)
_G1_COHORT_SPLITS = MappingProxyType(
    {
        "test_q24": "test",
        "test_c24": "test",
        "unseen24": "unseen",
    }
)
_ALL_COHORT_SPLITS = MappingProxyType(
    {
        **_G1_COHORT_SPLITS,
        "g3_test_q5": "test",
        "g3_unseen5": "unseen",
        "validation3": "validation",
        "replay3": "test",
    }
)
_BUNDLE_DATA_FILES = {
    "config.json",
    "descriptors.jsonl",
    "source-manifest.json",
    "reconstruction-index.jsonl",
    "denominator-proofs.jsonl",
}
_MANIFEST_FIELDS = {
    "schema_version",
    "completion_status",
    "config_sha256",
    "descriptor_catalog_sha256",
    "source_manifest_sha256",
    "stage6_coverage_manifest_sha256",
    "stage6_catalog_sha256",
    "reconstruction_index_sha256",
    "denominator_proofs_sha256",
    "descriptor_generator",
    "policy_blind_attestation",
    "source_pools",
    "cohort_sizes",
    "cohorts",
    "denominator_proofs",
    "bundle_files",
}


class MidtermReducedEvaluationError(RuntimeError):
    """缩减规模评价无法保持冻结输入或 update80 合同时抛出。"""


@dataclass(frozen=True, slots=True)
class FrozenScenarioManifest:
    bundle_root: Path
    manifest_sha256: str
    stage6_catalog_sha256: str
    cohorts: Mapping[str, tuple[str, ...]]
    denominator_proofs: Mapping[str, Mapping[str, object]]


@dataclass(frozen=True, slots=True)
class MidtermReducedEvaluation:
    frozen_manifest: FrozenScenarioManifest
    jobs: tuple[StandardEvaluationJob, ...]
    execution: StandardEvaluationExecution
    trace_summary: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class MidtermReducedJobPlan:
    jobs: tuple[StandardEvaluationJob, ...]
    record_id_by_frozen_scenario_id: Mapping[str, str]
    frozen_scenario_id_by_record_id: Mapping[str, str]
    scenario_hash_by_record_id: Mapping[str, str]


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"non-finite value {value!r}")


def _verify_frozen_bundle(bundle_root: str | Path) -> bool:
    """调用 Task 3 的独立全量复算器，不在本适配层复制选择逻辑。"""

    scripts_dir = Path(__file__).resolve().parents[3] / "scripts"
    inserted = str(scripts_dir) not in sys.path
    if inserted:
        sys.path.insert(0, str(scripts_dir))
    try:
        module = importlib.import_module("freeze_xunce_mid_dual_scenarios")
        verifier = getattr(module, "verify_frozen_bundle", None)
        return bool(verifier(bundle_root)) if callable(verifier) else False
    finally:
        if inserted:
            try:
                sys.path.remove(str(scripts_dir))
            except ValueError:
                pass


def _strict_manifest_payload(payload: bytes) -> dict[str, object]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, ValueError) as exc:
        raise MidtermReducedEvaluationError(
            "frozen scenario manifest is invalid JSON"
        ) from exc
    if (
        not isinstance(value, dict)
        or set(value) != _MANIFEST_FIELDS
        or ArtifactStore.canonical_json_bytes(value) != payload
    ):
        raise MidtermReducedEvaluationError(
            "frozen scenario manifest schema or canonical bytes drifted"
        )
    return value


def _validate_manifest_bundle_index(value: object) -> None:
    if not isinstance(value, list) or len(value) != len(_BUNDLE_DATA_FILES):
        raise MidtermReducedEvaluationError(
            "frozen scenario manifest bundle index is incomplete"
        )
    indexed: set[str] = set()
    for row in value:
        if (
            not isinstance(row, Mapping)
            or set(row) != {"path", "sha256", "size_bytes"}
            or row.get("path") not in _BUNDLE_DATA_FILES
            or not _is_sha256(row.get("sha256"))
            or type(row.get("size_bytes")) is not int
            or row["size_bytes"] < 0
            or row["path"] in indexed
        ):
            raise MidtermReducedEvaluationError(
                "frozen scenario manifest bundle index drifted"
            )
        indexed.add(str(row["path"]))
    if indexed != _BUNDLE_DATA_FILES:
        raise MidtermReducedEvaluationError(
            "frozen scenario manifest bundle index is incomplete"
        )


def _validated_cohorts(value: object) -> dict[str, tuple[str, ...]]:
    if not isinstance(value, Mapping) or set(value) != set(_COHORT_SIZES):
        raise MidtermReducedEvaluationError(
            "frozen scenario cohort schema drifted"
        )
    cohorts: dict[str, tuple[str, ...]] = {}
    for name, expected_count in _COHORT_SIZES.items():
        rows = value[name]
        expected_split = _ALL_COHORT_SPLITS[name]
        if (
            not isinstance(rows, list)
            or len(rows) != expected_count
            or any(
                not isinstance(item, str)
                or not item.startswith(f"{expected_split}/")
                for item in rows
            )
            or len(set(rows)) != expected_count
        ):
            raise MidtermReducedEvaluationError(
                f"frozen scenario cohort {name} drifted"
            )
        cohorts[name] = tuple(rows)
    if (
        set(cohorts["test_q24"]).intersection(cohorts["test_c24"])
        or not set(cohorts["g3_test_q5"]).issubset(cohorts["test_q24"])
        or not set(cohorts["g3_unseen5"]).issubset(cohorts["unseen24"])
        or not set(cohorts["replay3"]).issubset(cohorts["test_q24"])
    ):
        raise MidtermReducedEvaluationError(
            "frozen scenario cohort isolation drifted"
        )
    return cohorts


def _validated_denominator_proofs(
    value: object,
    *,
    required_ids: set[str],
) -> dict[str, Mapping[str, object]]:
    if not isinstance(value, list) or not value:
        raise MidtermReducedEvaluationError(
            "frozen denominator proofs are missing"
        )
    proofs: dict[str, Mapping[str, object]] = {}
    for row in value:
        scenario_id = row.get("scenario_id") if isinstance(row, Mapping) else None
        if (
            not isinstance(row, Mapping)
            or not isinstance(scenario_id, str)
            or not scenario_id
            or scenario_id in proofs
            or not _is_sha256(row.get("scenario_hash"))
            or not _is_sha256(row.get("coverable_mask_sha256"))
            or type(row.get("coverable_cell_count")) is not int
            or row["coverable_cell_count"] <= 0
            or row.get("algorithm_id") != DENOMINATOR_ALGORITHM
            or row.get("coverage_denominator_source") != DENOMINATOR_SOURCE
            or row.get("coverage_denominator_algorithm")
            != DENOMINATOR_ALGORITHM
            or row.get("exact") is not True
            or row.get("semantic_alias_proven") is not True
        ):
            raise MidtermReducedEvaluationError(
                "frozen denominator proof drifted"
            )
        proofs[scenario_id] = MappingProxyType(dict(row))
    if not required_ids.issubset(proofs):
        raise MidtermReducedEvaluationError(
            "frozen denominator proofs do not cover every selected scenario"
        )
    return proofs


def load_frozen_scenario_manifest(
    *,
    bundle_root: str | Path,
    expected_manifest_sha256: str,
) -> FrozenScenarioManifest:
    """验证 Task 3 完成 bundle、固定 manifest 哈希和所有 G1 分母证明。"""

    if not _is_sha256(expected_manifest_sha256):
        raise MidtermReducedEvaluationError(
            "expected frozen manifest SHA-256 is invalid"
        )
    try:
        root = require_plain_path(
            bundle_root,
            leaf_kind="directory",
            label="frozen scenario bundle root",
        )
        manifest_path = root / "manifest.json"
        payload = secure_read_bytes(
            manifest_path,
            base=root,
            label="frozen scenario manifest",
        ).payload
    except (OSError, PathSecurityError) as exc:
        raise MidtermReducedEvaluationError(
            "frozen scenario manifest is missing or unsafe"
        ) from exc
    if hashlib.sha256(payload).hexdigest() != expected_manifest_sha256:
        raise MidtermReducedEvaluationError(
            "frozen scenario manifest SHA-256 or identity drifted"
        )
    try:
        verified = _verify_frozen_bundle(root)
    except Exception as exc:
        raise MidtermReducedEvaluationError(
            "frozen scenario bundle verification failed"
        ) from exc
    if verified is not True:
        raise MidtermReducedEvaluationError(
            "frozen scenario bundle verification failed"
        )
    try:
        after_payload = secure_read_bytes(
            manifest_path,
            base=root,
            label="frozen scenario manifest",
        ).payload
        if after_payload != payload:
            raise MidtermReducedEvaluationError(
                "frozen scenario manifest changed after verification"
            )
    except (OSError, PathSecurityError) as exc:
        raise MidtermReducedEvaluationError(
            "frozen scenario manifest changed after verification"
        ) from exc

    manifest = _strict_manifest_payload(payload)
    hash_fields = (
        "config_sha256",
        "descriptor_catalog_sha256",
        "source_manifest_sha256",
        "stage6_coverage_manifest_sha256",
        "stage6_catalog_sha256",
        "reconstruction_index_sha256",
        "denominator_proofs_sha256",
    )
    if (
        manifest["schema_version"] != FROZEN_MANIFEST_SCHEMA
        or manifest["completion_status"] != "complete"
        or any(not _is_sha256(manifest[name]) for name in hash_fields)
        or manifest["cohort_sizes"] != dict(_COHORT_SIZES)
        or not isinstance(manifest["descriptor_generator"], Mapping)
        or not isinstance(manifest["policy_blind_attestation"], Mapping)
        or not isinstance(manifest["source_pools"], Mapping)
        or set(manifest["source_pools"]) != {"validation", "test", "unseen"}
        or any(
            not _is_sha256(item)
            for item in manifest["source_pools"].values()
        )
    ):
        raise MidtermReducedEvaluationError(
            "frozen scenario manifest completion contract drifted"
        )
    _validate_manifest_bundle_index(manifest["bundle_files"])
    cohorts = _validated_cohorts(manifest["cohorts"])
    required_ids = set().union(*cohorts.values())
    proofs = _validated_denominator_proofs(
        manifest["denominator_proofs"],
        required_ids=required_ids,
    )
    return FrozenScenarioManifest(
        bundle_root=root,
        manifest_sha256=expected_manifest_sha256,
        stage6_catalog_sha256=str(manifest["stage6_catalog_sha256"]),
        cohorts=MappingProxyType(cohorts),
        denominator_proofs=MappingProxyType(proofs),
    )


def build_midterm_reduced_jobs(
    *,
    catalog: StandardScenarioCatalog,
    frozen_manifest: FrozenScenarioManifest,
    cohort: Literal["test_q24", "test_c24", "unseen24"],
    evaluation_seed_start: int,
) -> tuple[StandardEvaluationJob, ...]:
    """从冻结顺序直接构建 24 个 job；不调用 canonical-64 builder。"""

    return build_midterm_reduced_job_plan(
        catalog=catalog,
        frozen_manifest=frozen_manifest,
        cohort=cohort,
        evaluation_seed_start=evaluation_seed_start,
    ).jobs


def build_midterm_reduced_job_plan(
    *,
    catalog: StandardScenarioCatalog,
    frozen_manifest: FrozenScenarioManifest,
    cohort: Literal["test_q24", "test_c24", "unseen24"],
    evaluation_seed_start: int,
) -> MidtermReducedJobPlan:
    """重建完整 proxy identity 后，形成 record ID 到 frozen ID 的一对一计划。"""

    if (
        not isinstance(catalog, StandardScenarioCatalog)
        or not isinstance(frozen_manifest, FrozenScenarioManifest)
        or cohort not in _G1_COHORT_SPLITS
        or type(evaluation_seed_start) is not int
        or catalog.sha256 != frozen_manifest.stage6_catalog_sha256
    ):
        raise MidtermReducedEvaluationError(
            "reduced Standard schedule binding drifted"
        )
    scenario_ids = frozen_manifest.cohorts[cohort]
    if len(scenario_ids) != 24 or len(set(scenario_ids)) != 24:
        raise MidtermReducedEvaluationError(
            "reduced frozen cohort must contain exactly 24 unique IDs"
        )
    expected_split = _G1_COHORT_SPLITS[cohort]
    selected_ids = set(scenario_ids)
    candidate_records = tuple(
        record
        for record in catalog.records
        if record.split == expected_split
        and f"{record.scenario_id}/standard-proxy/v1" in selected_ids
    )
    if len(candidate_records) != 24:
        raise MidtermReducedEvaluationError(
            "reduced frozen identities do not map one-to-one to catalog records"
        )
    factory = StandardScenarioFactory(catalog)
    records_by_frozen_id: dict[str, object] = {}
    record_id_by_frozen_id: dict[str, str] = {}
    frozen_id_by_record_id: dict[str, str] = {}
    scenario_hash_by_record_id: dict[str, str] = {}
    for record in candidate_records:
        try:
            scenario = factory.build(record)
        except Exception as exc:
            raise MidtermReducedEvaluationError(
                "reduced frozen scenario identity reconstruction failed"
            ) from exc
        proof = frozen_manifest.denominator_proofs.get(scenario.scenario_id)
        if (
            scenario.scenario_id not in selected_ids
            or not isinstance(proof, Mapping)
            or proof.get("scenario_hash") != scenario.scenario_hash
            or scenario.scenario_id in records_by_frozen_id
            or record.scenario_id in frozen_id_by_record_id
        ):
            raise MidtermReducedEvaluationError(
                "reduced frozen scenario identity or hash drifted"
            )
        records_by_frozen_id[scenario.scenario_id] = record
        record_id_by_frozen_id[scenario.scenario_id] = record.scenario_id
        frozen_id_by_record_id[record.scenario_id] = scenario.scenario_id
        scenario_hash_by_record_id[record.scenario_id] = scenario.scenario_hash
    if set(records_by_frozen_id) != selected_ids:
        raise MidtermReducedEvaluationError(
            "reduced frozen scenario identity is missing or ambiguous"
        )
    jobs: list[StandardEvaluationJob] = []
    for index, frozen_scenario_id in enumerate(scenario_ids):
        record = records_by_frozen_id.get(frozen_scenario_id)
        if record is None or record.split != expected_split:
            raise MidtermReducedEvaluationError(
                "reduced frozen scenario is absent from its bound catalog split"
            )
        jobs.append(
            StandardEvaluationJob(
                split=expected_split,  # type: ignore[arg-type]
                episode_index=index,
                scenario_id=record.scenario_id,
                scenario_seed=int(record.scenario_seed_hex, 16),
                terrain_seed=int(record.terrain_seed_hex, 16),
                start_pose_seed=int(record.start_pose_seed_hex, 16),
                evaluation_seed=evaluation_seed_start + index,
                theta_source="policy_theta_mu/v1",
            )
        )
    frozen_jobs = tuple(jobs)
    lanes = partition_standard_evaluation_jobs(frozen_jobs)
    if (
        len(lanes) != STANDARD_EVALUATION_WORKERS
        or tuple(len(lane) for lane in lanes) != (3,) * 8
    ):
        raise MidtermReducedEvaluationError(
            "reduced Standard schedule is not eight lanes of three"
        )
    return MidtermReducedJobPlan(
        jobs=frozen_jobs,
        record_id_by_frozen_scenario_id=MappingProxyType(
            {
                frozen_id: record_id_by_frozen_id[frozen_id]
                for frozen_id in scenario_ids
            }
        ),
        frozen_scenario_id_by_record_id=MappingProxyType(
            {
                record_id_by_frozen_id[frozen_id]: frozen_id
                for frozen_id in scenario_ids
            }
        ),
        scenario_hash_by_record_id=MappingProxyType(
            {
                record_id_by_frozen_id[frozen_id]: scenario_hash_by_record_id[
                    record_id_by_frozen_id[frozen_id]
                ]
                for frozen_id in scenario_ids
            }
        ),
    )


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise MidtermReducedEvaluationError(
            "update80 checkpoint is missing or unreadable"
        ) from exc
    return digest.hexdigest()


def _policy_state_sha256(policy: nn.Module) -> str:
    """与 Stage 6 loader 独立地复算 state_dict 身份。"""

    if not isinstance(policy, nn.Module):
        raise MidtermReducedEvaluationError(
            "update80 loader did not return a torch module"
        )
    digest = hashlib.sha256()
    for name, tensor in sorted(policy.state_dict().items()):
        if not isinstance(tensor, torch.Tensor):
            raise MidtermReducedEvaluationError(
                "update80 policy state contains a non-tensor value"
            )
        contiguous = tensor.detach().contiguous().cpu()
        raw = contiguous.numpy().tobytes(order="C")
        for value in (
            name.encode("utf-8"),
            str(contiguous.dtype).encode("ascii"),
            np.asarray(tuple(contiguous.shape), dtype="<i8").tobytes(),
            raw,
        ):
            digest.update(struct.pack("<Q", len(value)))
            digest.update(value)
    return digest.hexdigest()


def _validate_cuda_fp32_policy(policy: nn.Module) -> None:
    if not isinstance(policy, nn.Module):
        raise MidtermReducedEvaluationError(
            "update80 loader did not return a torch module"
        )
    tensors = tuple(policy.parameters()) + tuple(policy.buffers())
    if (
        not tensors
        or any(tensor.device.type != "cuda" for tensor in tensors)
        or any(
            tensor.is_floating_point() and tensor.dtype != torch.float32
            for tensor in tensors
        )
    ):
        raise MidtermReducedEvaluationError(
            "update80 policy must remain CUDA FP32 without CPU fallback"
        )


def load_midterm_update80_policy() -> nn.Module:
    """仅加载固定 update80，并在适配层再次核对文件和 policy 身份。"""

    if not torch.cuda.is_available():
        raise MidtermReducedEvaluationError(
            "midterm reduced evaluation requires CUDA; CPU fallback is forbidden"
        )
    before_hash = _sha256_file(UPDATE80_CHECKPOINT_PATH)
    if before_hash != UPDATE80_CHECKPOINT_SHA256:
        raise MidtermReducedEvaluationError(
            "update80 checkpoint SHA-256 drifted"
        )
    try:
        policy = load_stage4_policy_for_standard(
            checkpoint_path=UPDATE80_CHECKPOINT_PATH,
            checkpoint_sha256=UPDATE80_CHECKPOINT_SHA256,
            policy_state_sha256=UPDATE80_POLICY_STATE_SHA256,
            device="cuda",
        )
    except Exception as exc:
        raise MidtermReducedEvaluationError(
            "update80 checkpoint loader rejected the fixed binding"
        ) from exc
    if not isinstance(policy, nn.Module):
        raise MidtermReducedEvaluationError(
            "update80 loader did not return a torch module"
        )
    _validate_cuda_fp32_policy(policy)
    if (
        _policy_state_sha256(policy) != UPDATE80_POLICY_STATE_SHA256
        or _sha256_file(UPDATE80_CHECKPOINT_PATH)
        != UPDATE80_CHECKPOINT_SHA256
    ):
        raise MidtermReducedEvaluationError(
            "update80 checkpoint or policy-state identity drifted after load"
        )
    return policy


def _threshold_crossing(
    *,
    initial_coverage: float,
    rows: tuple[Mapping[str, object], ...],
    threshold: float,
) -> tuple[int | None, float | None]:
    if initial_coverage >= threshold:
        return 0, 0.0
    for row in rows:
        if float(row["coverage_rate"]) >= threshold:
            return (
                int(row["step_index"]) + 1,
                float(row["cumulative_path_length_m"]),
            )
    return None, None


def derive_midterm_reduced_trace_summary(
    *,
    execution: StandardEvaluationExecution,
    frozen_manifest: FrozenScenarioManifest,
    cohort: Literal["test_q24", "test_c24", "unseen24"],
    job_plan: MidtermReducedJobPlan,
) -> dict[str, object]:
    """只从内存逐步证据与冻结分母证明复算 80%/99% 指标。"""

    if (
        not isinstance(execution, StandardEvaluationExecution)
        or not isinstance(frozen_manifest, FrozenScenarioManifest)
        or not isinstance(job_plan, MidtermReducedJobPlan)
        or cohort not in _G1_COHORT_SPLITS
    ):
        raise MidtermReducedEvaluationError(
            "reduced trace summary inputs drifted"
        )
    expected_frozen_ids = frozen_manifest.cohorts[cohort]
    expected_ids = tuple(job.scenario_id for job in job_plan.jobs)
    if (
        len(job_plan.jobs) != 24
        or tuple(
            job_plan.record_id_by_frozen_scenario_id.get(frozen_id)
            for frozen_id in expected_frozen_ids
        )
        != expected_ids
        or tuple(
            job_plan.frozen_scenario_id_by_record_id.get(record_id)
            for record_id in expected_ids
        )
        != expected_frozen_ids
    ):
        raise MidtermReducedEvaluationError(
            "reduced trace job-plan identity binding drifted"
        )
    episode_rows = tuple(execution.episode_rows)
    step_rows = tuple(execution.step_rows)
    if (
        len(episode_rows) != 24
        or tuple(row.get("scenario_id") for row in episode_rows)
        != expected_ids
        or tuple(row.get("episode_index") for row in episode_rows)
        != tuple(range(24))
        or len({row.get("episode_id") for row in episode_rows}) != 24
    ):
        raise MidtermReducedEvaluationError(
            "reduced episode trace is incomplete or reordered"
        )
    lane_sizes: dict[str, int] = {}
    for row in episode_rows:
        lane_id = row.get("lane_id")
        if not isinstance(lane_id, str) or not lane_id:
            raise MidtermReducedEvaluationError(
                "reduced episode lane binding is missing"
            )
        lane_sizes[lane_id] = lane_sizes.get(lane_id, 0) + 1
    if len(lane_sizes) != 8 or tuple(sorted(lane_sizes.values())) != (3,) * 8:
        raise MidtermReducedEvaluationError(
            "reduced episode lanes are not eight groups of three"
        )

    grouped: dict[int, list[Mapping[str, object]]] = {
        index: [] for index in range(24)
    }
    join_keys: set[str] = set()
    for row in step_rows:
        episode_index = row.get("episode_index")
        join_key = row.get("join_key")
        if (
            type(episode_index) is not int
            or episode_index not in grouped
            or not isinstance(join_key, str)
            or not join_key
            or join_key in join_keys
            or row.get("scenario_id") != expected_ids[episode_index]
        ):
            raise MidtermReducedEvaluationError(
                "reduced step trace join drifted"
            )
        join_keys.add(join_key)
        grouped[episode_index].append(row)

    episode_summaries: list[dict[str, object]] = []
    final_coverages: list[float] = []
    for index, episode in enumerate(episode_rows):
        rows = tuple(
            sorted(grouped[index], key=lambda row: int(row["step_index"]))
        )
        if tuple(row.get("step_index") for row in rows) != tuple(
            range(len(rows))
        ):
            raise MidtermReducedEvaluationError(
                "reduced step trace is not contiguous"
            )
        initial = episode.get("initial_coverage_rate")
        final = episode.get("final_coverage")
        if (
            isinstance(initial, bool)
            or not isinstance(initial, (int, float))
            or not math.isfinite(float(initial))
            or isinstance(final, bool)
            or not isinstance(final, (int, float))
            or not math.isfinite(float(final))
            or not 0.0 <= float(initial) <= float(final) <= 1.0
        ):
            raise MidtermReducedEvaluationError(
                "reduced episode coverage is invalid"
            )
        previous_coverage = float(initial)
        previous_path = 0.0
        for row in rows:
            coverage = row.get("coverage_rate")
            path_length = row.get("cumulative_path_length_m")
            if (
                isinstance(coverage, bool)
                or not isinstance(coverage, (int, float))
                or not math.isfinite(float(coverage))
                or isinstance(path_length, bool)
                or not isinstance(path_length, (int, float))
                or not math.isfinite(float(path_length))
                or not previous_coverage <= float(coverage) <= 1.0
                or float(path_length) < previous_path
            ):
                raise MidtermReducedEvaluationError(
                    "reduced step coverage or path trace drifted"
                )
            previous_coverage = float(coverage)
            previous_path = float(path_length)
        if rows and previous_coverage != float(final):
            raise MidtermReducedEvaluationError(
                "reduced final coverage differs from its step trace"
            )
        if not rows and float(initial) != float(final):
            raise MidtermReducedEvaluationError(
                "zero-step reduced episode coverage drifted"
            )
        frozen_scenario_id = expected_frozen_ids[index]
        proof = frozen_manifest.denominator_proofs[frozen_scenario_id]
        if (
            proof["scenario_hash"]
            != job_plan.scenario_hash_by_record_id[expected_ids[index]]
        ):
            raise MidtermReducedEvaluationError(
                "reduced trace scenario hash binding drifted"
            )
        steps80, path80 = _threshold_crossing(
            initial_coverage=float(initial),
            rows=rows,
            threshold=0.80,
        )
        steps99, path99 = _threshold_crossing(
            initial_coverage=float(initial),
            rows=rows,
            threshold=0.99,
        )
        final_coverages.append(float(final))
        episode_summaries.append(
            {
                "episode_id": episode["episode_id"],
                "episode_index": index,
                "scenario_id": expected_ids[index],
                "frozen_scenario_id": frozen_scenario_id,
                "lane_id": episode["lane_id"],
                "denominator_cell_count": proof["coverable_cell_count"],
                "denominator_sha256": proof["coverable_mask_sha256"],
                "final_coverage": float(final),
                "steps_to_80": steps80,
                "path_length_to_80_m": path80,
                "steps_to_99": steps99,
                "path_length_to_99_m": path99,
            }
        )

    proof_rows = [
        frozen_manifest.denominator_proofs[scenario_id]
        for scenario_id in expected_frozen_ids
    ]
    denominator_audit = {
        "source": DENOMINATOR_SOURCE,
        "algorithm": DENOMINATOR_ALGORITHM,
        "proof_count": len(proof_rows),
        "nonempty_count": sum(
            int(row["coverable_cell_count"]) > 0 for row in proof_rows
        ),
        "exact_count": sum(row["exact"] is True for row in proof_rows),
        "semantic_alias_proven_count": sum(
            row["semantic_alias_proven"] is True for row in proof_rows
        ),
        "passed": True,
    }
    return {
        "schema_version": _TRACE_SUMMARY_SCHEMA,
        "scale_profile": SCALE_PROFILE,
        "cohort": cohort,
        "split": _G1_COHORT_SPLITS[cohort],
        "episode_count": 24,
        "lane_sizes": dict(sorted(lane_sizes.items())),
        "denominator_audit": denominator_audit,
        "episodes": episode_summaries,
        "final_coverage": {
            "sample_count": 24,
            "mean": mean(final_coverages),
            "median": median(final_coverages),
            "sample_stddev": stdev(final_coverages),
            "min": min(final_coverages),
            "max": max(final_coverages),
            "coverage_80_count": sum(
                value >= 0.80 for value in final_coverages
            ),
            "coverage_99_count": sum(
                value >= 0.99 for value in final_coverages
            ),
        },
    }


def run_midterm_reduced_evaluation(
    *,
    catalog: StandardScenarioCatalog,
    frozen_bundle_root: str | Path,
    frozen_manifest_sha256: str,
    cohort: Literal["test_q24", "test_c24", "unseen24"],
    evaluation_seed_start: int,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    safety_contract: SafetyContract,
    config_sha256: str,
    resource_guard: Callable[[str], None] | None = None,
) -> MidtermReducedEvaluation:
    """加载固定 update80 并运行显式 24-job 内存评价，不写任何 artifact。"""

    frozen = load_frozen_scenario_manifest(
        bundle_root=frozen_bundle_root,
        expected_manifest_sha256=frozen_manifest_sha256,
    )
    job_plan = build_midterm_reduced_job_plan(
        catalog=catalog,
        frozen_manifest=frozen,
        cohort=cohort,
        evaluation_seed_start=evaluation_seed_start,
    )
    jobs = job_plan.jobs
    policy = load_midterm_update80_policy()
    execution = run_standard_evaluation_jobs(
        catalog=catalog,
        jobs=jobs,
        method="ppo_policy",
        policy=policy,
        policy_device="cuda",
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
        safety_contract=safety_contract,
        config_sha256=config_sha256,
        resource_guard=resource_guard,
    )
    if not isinstance(execution, StandardEvaluationExecution):
        raise MidtermReducedEvaluationError(
            "explicit Standard evaluation did not return in-memory evidence"
        )
    summary = derive_midterm_reduced_trace_summary(
        execution=execution,
        frozen_manifest=frozen,
        cohort=cohort,
        job_plan=job_plan,
    )
    return MidtermReducedEvaluation(
        frozen_manifest=frozen,
        jobs=jobs,
        execution=execution,
        trace_summary=MappingProxyType(summary),
    )


__all__ = [
    "DENOMINATOR_ALGORITHM",
    "DENOMINATOR_SOURCE",
    "FROZEN_MANIFEST_SCHEMA",
    "FrozenScenarioManifest",
    "MidtermReducedEvaluation",
    "MidtermReducedEvaluationError",
    "MidtermReducedJobPlan",
    "SCALE_PROFILE",
    "UPDATE80_CHECKPOINT_PATH",
    "UPDATE80_CHECKPOINT_SHA256",
    "UPDATE80_POLICY_STATE_SHA256",
    "build_midterm_reduced_job_plan",
    "build_midterm_reduced_jobs",
    "derive_midterm_reduced_trace_summary",
    "load_frozen_scenario_manifest",
    "load_midterm_update80_policy",
    "run_midterm_reduced_evaluation",
]
