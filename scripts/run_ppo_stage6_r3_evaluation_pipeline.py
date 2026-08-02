"""Run the Stage 6 R3 U80/U100 screen and the selected full final evaluation."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import torch

from lunar_exploration_ppo.configs.stage6 import SafetyContract
from lunar_exploration_ppo.env.standard_training import build_standard_catalog
from lunar_exploration_ppo.eval.standard import run_standard_evaluation
from lunar_exploration_ppo.ppo.standard_training import (
    ValidationRecord,
    select_seed_best,
)
from lunar_exploration_ppo.ppo.trainer import policy_state_sha256
from lunar_exploration_ppo.utils.artifact_io import ArtifactStore
from lunar_exploration_ppo.utils.durable_jsonl import DurableJsonl
from lunar_exploration_ppo.utils.resources import (
    capture_resource_snapshot,
    evaluate_resource_gates,
)
from lunar_exploration_ppo.workflows.stage6 import (
    FINAL_EVALUATION_METHODS,
    load_stage4_policy_for_standard,
)
from lunar_exploration_ppo.workflows.stage6_planning_warm_start import (
    parse_planning_effective_config_bytes,
)


SCHEMA = "stage6_r3_evaluation_pipeline/v1"
SEED = 20260716
SCREEN_UPDATES = (80, 100)
SCREEN_EPISODES = 16
FINAL_EPISODES = 64
FINAL_SPLITS = ("test", "unseen")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo_root), *args],
        text=True,
        encoding="utf-8",
    ).strip()


def _runtime_source_identity(repo_root: Path) -> dict[str, Any]:
    relative_paths = [Path("configs/ppo_highres_frontier_stage6_v1.json")]
    relative_paths.extend(
        sorted(
            path.relative_to(repo_root)
            for path in (repo_root / "src/lunar_exploration_ppo").rglob("*.py")
        )
    )
    relative_paths.append(Path("scripts/run_ppo_stage6_r3_evaluation_pipeline.py"))
    rows: list[dict[str, Any]] = []
    for relative in relative_paths:
        path = repo_root / relative
        payload = path.read_bytes()
        rows.append(
            {
                "path": relative.as_posix(),
                "sha256": _sha256_bytes(payload),
                "size_bytes": len(payload),
            }
        )
    canonical = ArtifactStore.canonical_json_bytes(rows)
    return {
        "schema_version": "stage6_r3_runtime_source_identity/v1",
        "source_set_sha256": _sha256_bytes(canonical),
        "paths": rows,
    }


def _checkpoint_identity(stage_root: Path, update: int) -> dict[str, Any]:
    root = stage_root / f"checkpoints/seed-{SEED}/update-{update:08d}"
    checkpoint = root / "checkpoint.pt"
    manifest_path = root / "manifest.json"
    complete_path = root / "complete.json"
    if {path.name for path in root.iterdir()} != {
        "checkpoint.pt",
        "manifest.json",
        "complete.json",
    }:
        raise RuntimeError(f"checkpoint member set drifted: U{update}")
    manifest = _read_json(manifest_path)
    checkpoint_sha256 = _sha256_file(checkpoint)
    if (
        manifest.get("update_step") != update
        or manifest.get("checkpoint", {}).get("sha256") != checkpoint_sha256
        or not isinstance(manifest.get("policy_state_sha256"), str)
    ):
        raise RuntimeError(f"checkpoint manifest drifted: U{update}")
    return {
        "update": update,
        "root": root.as_posix(),
        "checkpoint_path": checkpoint.as_posix(),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "manifest_sha256": _sha256_file(manifest_path),
        "complete_sha256": _sha256_file(complete_path),
        "policy_state_sha256": manifest["policy_state_sha256"],
        "config_sha256": manifest["config_sha256"],
        "lineage_sha256": manifest["lineage_sha256"],
    }


def _append_state(output_root: Path, event: str, **fields: Any) -> None:
    DurableJsonl(output_root / "state.jsonl").append(
        {
            "schema_version": "stage6_r3_evaluation_state/v1",
            "timestamp_utc": _utc_now(),
            "event": event,
            **fields,
        }
    )


def _resource_guard_factory(rows: list[dict[str, Any]]) -> Callable[[str], None]:
    last_capture = 0.0

    def guard(boundary: str) -> None:
        nonlocal last_capture
        now = time.monotonic()
        if rows and now - last_capture < 5.0:
            return
        last_capture = now
        peak_vram = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        snapshot = capture_resource_snapshot(peak_vram_bytes=peak_vram)
        decision = evaluate_resource_gates(snapshot, preflight=False)
        rows.append(
            {
                "boundary": boundary,
                "snapshot": dataclasses.asdict(snapshot),
                "decision": dataclasses.asdict(decision),
            }
        )
        if decision.passed is not True:
            raise RuntimeError(f"resource hard stop at {boundary}")

    return guard


def _summary_payload(summary: Any) -> dict[str, Any]:
    return {
        "metrics": dict(summary.metrics),
        "bootstrap_audit": dict(summary.bootstrap_audit),
        "fairness_audit": dict(summary.fairness_audit),
    }


def _verify_completed_result(
    *,
    summary_path: Path,
    trace_path: Path,
    source_sha256: str,
    checkpoint: Mapping[str, Any],
    episode_count: int,
) -> dict[str, Any]:
    value = _read_json(summary_path)
    trace_bytes = trace_path.read_bytes()
    if (
        value.get("schema_version") != "stage6_r3_evaluation_result/v1"
        or value.get("source_set_sha256") != source_sha256
        or value.get("checkpoint_sha256") != checkpoint["checkpoint_sha256"]
        or value.get("policy_state_sha256") != checkpoint["policy_state_sha256"]
        or value.get("episode_count") != episode_count
        or value.get("trace_sha256") != _sha256_bytes(trace_bytes)
        or value.get("trace_line_count") != len(trace_bytes.splitlines())
        or value.get("trace_line_count") != episode_count
        or not isinstance(value.get("result"), dict)
    ):
        raise RuntimeError(f"completed evaluation binding drifted: {summary_path}")
    return value


def _run_one_evaluation(
    *,
    output_root: Path,
    relative_stem: str,
    catalog: Any,
    config: Any,
    config_sha256: str,
    safety_contract: SafetyContract,
    source_identity: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    loaded_policy: torch.nn.Module,
    split: str,
    method: str,
    episode_count: int,
) -> dict[str, Any]:
    summary_path = output_root / f"{relative_stem}.summary.json"
    trace_path = output_root / f"{relative_stem}.jsonl"
    if summary_path.is_file() and trace_path.is_file():
        return _verify_completed_result(
            summary_path=summary_path,
            trace_path=trace_path,
            source_sha256=str(source_identity["source_set_sha256"]),
            checkpoint=checkpoint,
            episode_count=episode_count,
        )
    pending_trace = output_root / f"{relative_stem}.pending.jsonl"
    if any(path.exists() for path in (summary_path, trace_path, pending_trace)):
        raise RuntimeError(f"partial evaluation artifacts require diagnosis: {relative_stem}")

    current_source = _runtime_source_identity(Path.cwd())
    if current_source["source_set_sha256"] != source_identity["source_set_sha256"]:
        raise RuntimeError("runtime source changed before evaluation")
    if _sha256_file(Path(str(checkpoint["checkpoint_path"]))) != checkpoint["checkpoint_sha256"]:
        raise RuntimeError("checkpoint changed before evaluation")

    resources: list[dict[str, Any]] = []
    resource_guard = _resource_guard_factory(resources)
    policy_before = policy_state_sha256(loaded_policy)
    started = time.perf_counter()
    _append_state(
        output_root,
        "evaluation_started",
        stem=relative_stem,
        split=split,
        method=method,
        episode_count=episode_count,
        update=checkpoint["update"],
    )
    summary = run_standard_evaluation(
        catalog=catalog,
        split=split,
        method=method,
        episode_count=episode_count,
        evaluation_seed_start=config.evaluation.bootstrap_seed,
        policy=loaded_policy if method == "ppo_policy" else None,
        policy_device=config.device,
        bootstrap_resamples=config.evaluation.bootstrap_resamples,
        bootstrap_seed=config.evaluation.bootstrap_seed,
        trace_path=pending_trace,
        safety_contract=safety_contract,
        config_sha256=config_sha256,
        resource_guard=resource_guard,
    )
    wall_seconds = time.perf_counter() - started
    resource_guard("evaluation:post")
    policy_after = policy_state_sha256(loaded_policy)
    if policy_before != checkpoint["policy_state_sha256"] or policy_after != policy_before:
        raise RuntimeError("evaluation mutated or misbound policy")
    if _runtime_source_identity(Path.cwd())["source_set_sha256"] != source_identity["source_set_sha256"]:
        raise RuntimeError("runtime source changed during evaluation")
    if _sha256_file(Path(str(checkpoint["checkpoint_path"]))) != checkpoint["checkpoint_sha256"]:
        raise RuntimeError("checkpoint changed during evaluation")

    trace_bytes = pending_trace.read_bytes()
    if len(trace_bytes.splitlines()) != episode_count:
        raise RuntimeError("evaluation trace episode count drifted")
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(pending_trace, trace_path)
    payload = {
        "schema_version": "stage6_r3_evaluation_result/v1",
        "completed_at_utc": _utc_now(),
        "split": split,
        "method": method,
        "episode_count": episode_count,
        "update": checkpoint["update"],
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "policy_state_sha256": checkpoint["policy_state_sha256"],
        "config_sha256": config_sha256,
        "source_set_sha256": source_identity["source_set_sha256"],
        "trace_path": trace_path.relative_to(output_root).as_posix(),
        "trace_sha256": _sha256_bytes(trace_bytes),
        "trace_line_count": len(trace_bytes.splitlines()),
        "wall_seconds": wall_seconds,
        "result": _summary_payload(summary),
        "resources": resources,
    }
    ArtifactStore(output_root).write_json_exclusive(
        summary_path.relative_to(output_root), payload
    )
    _append_state(
        output_root,
        "evaluation_completed",
        stem=relative_stem,
        split=split,
        method=method,
        episode_count=episode_count,
        update=checkpoint["update"],
        wall_seconds=wall_seconds,
        metrics=payload["result"]["metrics"],
    )
    print(json.dumps({"completed": relative_stem, "wall_seconds": wall_seconds, "metrics": payload["result"]["metrics"]}, sort_keys=True), flush=True)
    return payload


def _load_policy(checkpoint: Mapping[str, Any], device: str) -> torch.nn.Module:
    policy = load_stage4_policy_for_standard(
        checkpoint_path=checkpoint["checkpoint_path"],
        checkpoint_sha256=checkpoint["checkpoint_sha256"],
        policy_state_sha256=checkpoint["policy_state_sha256"],
        device=device,
    )
    if not isinstance(policy, torch.nn.Module):
        raise RuntimeError("checkpoint did not load a policy")
    return policy


def _screen(
    *,
    output_root: Path,
    catalog: Any,
    config: Any,
    config_sha256: str,
    safety_contract: SafetyContract,
    source_identity: Mapping[str, Any],
    checkpoints: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    records: list[ValidationRecord] = []
    summaries: dict[str, Any] = {}
    for update in SCREEN_UPDATES:
        policy = _load_policy(checkpoints[update], config.device)
        result = _run_one_evaluation(
            output_root=output_root,
            relative_stem=f"screen/u{update:03d}-validation-16",
            catalog=catalog,
            config=config,
            config_sha256=config_sha256,
            safety_contract=safety_contract,
            source_identity=source_identity,
            checkpoint=checkpoints[update],
            loaded_policy=policy,
            split="validation",
            method="ppo_policy",
            episode_count=SCREEN_EPISODES,
        )
        metrics = result["result"]["metrics"]
        records.append(
            ValidationRecord(
                seed=SEED,
                update=update,
                success_rate_under_fixed_step_budget=float(
                    metrics["success_rate_under_fixed_step_budget"]
                ),
                mean_final_coverage=float(metrics["mean_final_coverage"]),
                checkpoint_ref=f"update-{update:08d}",
            )
        )
        summaries[str(update)] = result
        del policy
        torch.cuda.empty_cache()

    selected = select_seed_best(records)
    checkpoint = checkpoints[selected.update]
    stable_value = {
        "schema_version": "stage6_r3_global_best/v1",
        "selection_policy": "success_rate_then_mean_final_coverage_then_earlier_update/v1",
        "record": selected.to_dict(),
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "policy_state_sha256": checkpoint["policy_state_sha256"],
        "source_set_sha256": source_identity["source_set_sha256"],
        "screen_summary_sha256": _sha256_bytes(
            ArtifactStore.canonical_json_bytes(summaries)
        ),
    }
    path = output_root / "global-best.json"
    if path.exists():
        value = _read_json(path)
        if not isinstance(value.get("selected_at_utc"), str):
            raise RuntimeError("existing global best has no selection timestamp")
        if {key: value.get(key) for key in stable_value} != stable_value:
            raise RuntimeError("existing global best drifted")
    else:
        value = {**stable_value, "selected_at_utc": _utc_now()}
        ArtifactStore(output_root).write_json_exclusive("global-best.json", value)
        _append_state(output_root, "global_best_selected", **value)
    print(json.dumps({"global_best": value}, sort_keys=True), flush=True)
    return value


def _final(
    *,
    output_root: Path,
    catalog: Any,
    config: Any,
    config_sha256: str,
    safety_contract: SafetyContract,
    source_identity: Mapping[str, Any],
    checkpoints: Mapping[int, Mapping[str, Any]],
    global_best: Mapping[str, Any],
) -> dict[str, Any]:
    selected_update = int(global_best["record"]["update"])
    checkpoint = checkpoints[selected_update]
    policy = _load_policy(checkpoint, config.device)
    results: dict[str, Any] = {}
    for split in FINAL_SPLITS:
        for method in FINAL_EVALUATION_METHODS:
            key = f"{split}:{method}"
            results[key] = _run_one_evaluation(
                output_root=output_root,
                relative_stem=f"final/{split}-{method}-64",
                catalog=catalog,
                config=config,
                config_sha256=config_sha256,
                safety_contract=safety_contract,
                source_identity=source_identity,
                checkpoint=checkpoint,
                loaded_policy=policy,
                split=split,
                method=method,
                episode_count=FINAL_EPISODES,
            )
    stable_value = {
        "schema_version": "stage6_r3_final_summary/v1",
        "global_best": dict(global_best),
        "evaluation_count": len(results),
        "episode_count": sum(int(row["episode_count"]) for row in results.values()),
        "source_set_sha256": source_identity["source_set_sha256"],
        "config_sha256": config_sha256,
        "results": results,
    }
    if stable_value["evaluation_count"] != 10 or stable_value["episode_count"] != 640:
        raise RuntimeError("final evaluation schedule drifted")
    path = output_root / "final-summary.json"
    if path.exists():
        value = _read_json(path)
        if not isinstance(value.get("completed_at_utc"), str):
            raise RuntimeError("existing final summary has no completion timestamp")
        if {key: value.get(key) for key in stable_value} != stable_value:
            raise RuntimeError("existing final summary drifted")
    else:
        value = {**stable_value, "completed_at_utc": _utc_now()}
        ArtifactStore(output_root).write_json_exclusive("final-summary.json", value)
        _append_state(output_root, "pipeline_completed", evaluation_count=10, episode_count=640)
    print(json.dumps({"final_summary": "final-summary.json", "evaluation_count": 10, "episode_count": 640}, sort_keys=True), flush=True)
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--stage-root",
        type=Path,
        default=Path(
            "D:/xunce/out/ppo_frontier/"
            "s6-standard-single-r1-20260724T000124Z/s6"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.expanduser().resolve()
    stage_root = args.stage_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if output_root.drive.upper() != "D:":
        raise RuntimeError("evaluation output root must be on D drive")
    os.chdir(repo_root)
    source_identity = _runtime_source_identity(repo_root)
    effective_bytes = (stage_root / "config.json").read_bytes()
    effective = parse_planning_effective_config_bytes(effective_bytes)
    config = effective.base_config
    config_sha256 = _sha256_bytes(effective_bytes)
    if config_sha256 != effective.effective_config_sha256:
        raise RuntimeError("effective config SHA drifted")
    checkpoints = {
        update: _checkpoint_identity(stage_root, update)
        for update in SCREEN_UPDATES
    }
    if any(value["config_sha256"] != config_sha256 for value in checkpoints.values()):
        raise RuntimeError("checkpoint config binding drifted")
    if not torch.cuda.is_available() or config.device != "cuda":
        raise RuntimeError("Stage 6 R3 evaluation requires CUDA")
    safety_contract = SafetyContract.from_stage6_config(config)
    catalog = build_standard_catalog(verify_hashes=True)
    identity = {
        "schema_version": SCHEMA,
        "created_at_utc": _utc_now(),
        "repo_root": repo_root.as_posix(),
        "git_head": _git(repo_root, "rev-parse", "HEAD"),
        "git_branch": _git(repo_root, "branch", "--show-current"),
        "stage_root": stage_root.as_posix(),
        "output_root": output_root.as_posix(),
        "config_sha256": config_sha256,
        "catalog_sha256": catalog.sha256,
        "source_identity": source_identity,
        "checkpoints": checkpoints,
        "screen": {"updates": list(SCREEN_UPDATES), "episodes_each": SCREEN_EPISODES},
        "final": {
            "splits": list(FINAL_SPLITS),
            "methods": list(FINAL_EVALUATION_METHODS),
            "episodes_each": FINAL_EPISODES,
            "evaluation_count": 10,
            "episode_count": 640,
        },
    }
    if args.dry_run:
        for checkpoint in checkpoints.values():
            policy = _load_policy(checkpoint, config.device)
            if policy_state_sha256(policy) != checkpoint["policy_state_sha256"]:
                raise RuntimeError("dry-run policy hash drifted")
            del policy
            torch.cuda.empty_cache()
        print(json.dumps({"dry_run": "passed", "identity": identity}, sort_keys=True))
        return 0

    if output_root.exists() and not args.resume:
        raise RuntimeError("output root already exists; use --resume only after diagnosis")
    output_root.mkdir(parents=True, exist_ok=True)
    identity_path = output_root / "identity.json"
    if identity_path.exists():
        existing = _read_json(identity_path)
        stable_fields = {
            key: value for key, value in identity.items() if key != "created_at_utc"
        }
        existing_stable = {
            key: value for key, value in existing.items() if key != "created_at_utc"
        }
        if existing_stable != stable_fields:
            raise RuntimeError("resume identity drifted")
    else:
        ArtifactStore(output_root).write_json_exclusive("identity.json", identity)
    _append_state(output_root, "pipeline_started", source_set_sha256=source_identity["source_set_sha256"])
    global_best = _screen(
        output_root=output_root,
        catalog=catalog,
        config=config,
        config_sha256=config_sha256,
        safety_contract=safety_contract,
        source_identity=source_identity,
        checkpoints=checkpoints,
    )
    _final(
        output_root=output_root,
        catalog=catalog,
        config=config,
        config_sha256=config_sha256,
        safety_contract=safety_contract,
        source_identity=source_identity,
        checkpoints=checkpoints,
        global_best=global_best,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
