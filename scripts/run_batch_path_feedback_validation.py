from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MATRIX_SCHEMA_VERSION = "path-feedback-batch-matrix/v1"
RUN_INDEX_SCHEMA_VERSION = "path-feedback-batch-run-index/v1"
EVALUATION_SUMMARY_SCHEMA_VERSION = "path-feedback-batch-evaluation-summary/v1"
SUMMARY_SCHEMA_VERSION = "path-feedback-summary/v1"
SCENARIO_SETS = {"smoke", "stress", "all"}
DIAGNOSTIC_PROFILES = {"baseline", "execution", "iris", "all"}
SUBMODULES = ("dev-platform-constraints", "model-explorer", "path-planner")
RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")


class MatrixError(ValueError):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a batch of semi-real path-feedback validation commands."
    )
    parser.add_argument("--matrix", required=True, help="Path to path-feedback batch matrix JSON.")
    parser.add_argument(
        "--output-root",
        help="Override the batch output root. Each run writes to <output-root>/<run_id>/.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and print planned commands without writing outputs.")
    parser.add_argument("--validate-only", action="store_true", help="Validate the matrix and exit without running commands.")
    parser.add_argument(
        "--single-run-script",
        help="Single-run validation script to orchestrate. Defaults to scripts/run_path_feedback_validation.sh.",
    )
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    matrix_path = _resolve_path(args.matrix, repo_root)
    single_run_script = (
        _resolve_path(args.single_run_script, repo_root)
        if args.single_run_script
        else repo_root / "scripts" / "run_path_feedback_validation.sh"
    )

    try:
        matrix_payload = _load_matrix_json(matrix_path)
        batch_plan = _build_batch_plan(
            matrix_payload,
            matrix_path=matrix_path,
            repo_root=repo_root,
            cli_output_root=args.output_root,
            single_run_script=single_run_script,
        )
    except MatrixError as exc:
        print(f"matrix error: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "status": "matrix validated",
                "matrix": _display_path(matrix_path, repo_root),
                "run_count": len(batch_plan["runs"]),
                "output_root": _display_path(batch_plan["output_root"], repo_root),
            },
            ensure_ascii=False,
        )
    )

    if args.validate_only:
        return 0

    if args.dry_run:
        for run in batch_plan["runs"]:
            print(f"[DRY RUN] {run['run_id']}: (cd {repo_root} && {shlex.join(run['argv'])})")
        return 0

    git_snapshot = _git_snapshot(repo_root)
    batch_plan["output_root"].mkdir(parents=True, exist_ok=True)

    run_records: list[dict[str, Any]] = []
    for run in batch_plan["runs"]:
        run_records.append(_execute_run(run, repo_root=repo_root, git_snapshot=git_snapshot))

    run_index = _build_run_index(batch_plan, run_records, repo_root=repo_root, git_snapshot=git_snapshot)
    evaluation_summary = _build_evaluation_summary(batch_plan, run_records, repo_root=repo_root)

    index_path = batch_plan["output_root"] / "batch-run-index.json"
    summary_path = batch_plan["output_root"] / "batch-evaluation-summary.json"
    _write_json(index_path, run_index)
    _write_json(summary_path, evaluation_summary)

    print(
        json.dumps(
            {
                "status": "batch complete" if evaluation_summary["failed_count"] == 0 else "batch failed",
                "run_count": evaluation_summary["run_count"],
                "passed_count": evaluation_summary["passed_count"],
                "failed_count": evaluation_summary["failed_count"],
                "batch_run_index": _display_path(index_path, repo_root),
                "batch_evaluation_summary": _display_path(summary_path, repo_root),
            },
            ensure_ascii=False,
        )
    )
    return 1 if evaluation_summary["failed_count"] else 0


def _load_matrix_json(matrix_path: Path) -> dict[str, Any]:
    if not matrix_path.is_file():
        raise MatrixError(f"matrix file does not exist: {matrix_path}")
    try:
        payload = json.loads(matrix_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MatrixError(f"matrix JSON is invalid: {exc}") from exc
    if not isinstance(payload, dict):
        raise MatrixError("matrix root must be an object")
    schema_version = payload.get("schema_version")
    if schema_version != MATRIX_SCHEMA_VERSION:
        raise MatrixError(f"schema_version must be {MATRIX_SCHEMA_VERSION!r}")
    return payload


def _build_batch_plan(
    payload: dict[str, Any],
    *,
    matrix_path: Path,
    repo_root: Path,
    cli_output_root: str | None,
    single_run_script: Path,
) -> dict[str, Any]:
    defaults = payload.get("defaults", {})
    if defaults is None:
        defaults = {}
    if not isinstance(defaults, dict):
        raise MatrixError("defaults must be an object when present")

    raw_runs = payload.get("runs")
    if not isinstance(raw_runs, list) or not raw_runs:
        raise MatrixError("runs must be a non-empty array")

    output_root_value = (
        cli_output_root
        if cli_output_root is not None
        else payload.get("output_root", defaults.get("output_root", "outputs/path_feedback_batch"))
    )
    if not isinstance(output_root_value, str) or not output_root_value:
        raise MatrixError("output_root must be a non-empty string")
    batch_output_root = _resolve_path(output_root_value, repo_root)

    if not single_run_script.is_file():
        raise MatrixError(f"single-run script does not exist: {single_run_script}")

    seen_run_ids: set[str] = set()
    runs: list[dict[str, Any]] = []
    for index, raw_run in enumerate(raw_runs):
        if not isinstance(raw_run, dict):
            raise MatrixError(f"runs[{index}] must be an object")
        merged = {**defaults, **raw_run}
        run_id = _require_run_id(merged.get("run_id"), index)
        if run_id in seen_run_ids:
            raise MatrixError(f"run_id must be unique: {run_id}")
        seen_run_ids.add(run_id)

        scenario_set = _require_choice(merged.get("scenario_set"), SCENARIO_SETS, f"runs[{index}].scenario_set")
        diagnostic_profile = _require_choice(
            merged.get("diagnostic_profile"),
            DIAGNOSTIC_PROFILES,
            f"runs[{index}].diagnostic_profile",
        )
        top_k = _require_positive_int(merged.get("top_k"), f"runs[{index}].top_k")
        sample_quality_profile = merged.get("sample_quality_profile")
        if sample_quality_profile is not None and not isinstance(sample_quality_profile, (str, dict)):
            raise MatrixError(f"runs[{index}].sample_quality_profile must be a string or object when present")
        planner_extra_args = _optional_string_list(
            merged.get("planner_extra_args", ()),
            f"runs[{index}].planner_extra_args",
        )

        if cli_output_root is None and isinstance(raw_run.get("output_root"), str):
            run_output_root = _resolve_path(raw_run["output_root"], repo_root)
        else:
            run_output_root = batch_output_root / run_id

        argv = [
            "bash",
            str(single_run_script),
            "--scenario-set",
            scenario_set,
            "--diagnostic-profile",
            diagnostic_profile,
            "--top-k",
            str(top_k),
            "--output-root",
            str(run_output_root),
        ]
        argv.extend(planner_extra_args)
        runs.append(
            {
                "run_id": run_id,
                "scenario_set": scenario_set,
                "diagnostic_profile": diagnostic_profile,
                "top_k": top_k,
                "sample_quality_profile": sample_quality_profile,
                "planner_extra_args": planner_extra_args,
                "output_root": run_output_root,
                "argv": argv,
            }
        )

    return {
        "schema_version": MATRIX_SCHEMA_VERSION,
        "matrix_path": matrix_path,
        "output_root": batch_output_root,
        "single_run_script": single_run_script,
        "runs": runs,
    }


def _require_run_id(value: Any, index: int) -> str:
    if not isinstance(value, str) or not value:
        raise MatrixError(f"runs[{index}].run_id must be a non-empty string")
    if not RUN_ID_RE.match(value):
        raise MatrixError(f"runs[{index}].run_id may contain only letters, numbers, dot, underscore, and dash")
    return value


def _require_choice(value: Any, choices: set[str], label: str) -> str:
    if not isinstance(value, str) or value not in choices:
        raise MatrixError(f"{label} must be one of: {', '.join(sorted(choices))}")
    return value


def _require_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise MatrixError(f"{label} must be a positive integer")
    return value


def _optional_string_list(value: Any, label: str) -> list[str]:
    if value in (None, ()):
        return []
    if not isinstance(value, list):
        raise MatrixError(f"{label} must be an array of strings when present")
    if any(not isinstance(item, str) or item == "" for item in value):
        raise MatrixError(f"{label} must contain only non-empty strings")
    return list(value)


def _execute_run(run: dict[str, Any], *, repo_root: Path, git_snapshot: dict[str, Any]) -> dict[str, Any]:
    run_root = run["output_root"]
    run_root.mkdir(parents=True, exist_ok=True)
    stdout_log = run_root / "batch-run.stdout.log"
    stderr_log = run_root / "batch-run.stderr.log"

    print(f"==> batch run {run['run_id']}: {shlex.join(run['argv'])}")
    with stdout_log.open("w", encoding="utf-8") as stdout_file, stderr_log.open("w", encoding="utf-8") as stderr_file:
        completed = subprocess.run(
            run["argv"],
            cwd=repo_root,
            text=True,
            stdout=stdout_file,
            stderr=stderr_file,
        )

    record = _record_run_result(
        run,
        repo_root=repo_root,
        git_snapshot=git_snapshot,
        return_code=completed.returncode,
        stdout_log=stdout_log,
        stderr_log=stderr_log,
    )
    print(
        json.dumps(
            {
                "run_id": record["run_id"],
                "status": record["status"],
                "return_code": record["return_code"],
                "reason_codes": record["reason_codes"],
            },
            ensure_ascii=False,
        )
    )
    return record


def _record_run_result(
    run: dict[str, Any],
    *,
    repo_root: Path,
    git_snapshot: dict[str, Any],
    return_code: int,
    stdout_log: Path,
    stderr_log: Path,
) -> dict[str, Any]:
    run_root = run["output_root"]
    manifest_path = run_root / "path-feedback-manifest.json"
    summary_path = run_root / "path-feedback-summary.json"
    report_path = run_root / "path-feedback-summary.md"
    reason_codes: list[str] = []
    summary: dict[str, Any] | None = None

    if return_code != 0:
        _append_reason(reason_codes, "single_run_exit_nonzero")

    if not summary_path.is_file():
        _append_reason(reason_codes, "summary_missing")
    else:
        try:
            loaded_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            _append_reason(reason_codes, "summary_invalid_json")
        else:
            if isinstance(loaded_summary, dict):
                summary = loaded_summary
                _inspect_summary_for_reasons(summary, reason_codes)
            else:
                _append_reason(reason_codes, "summary_not_object")

    if not manifest_path.is_file():
        _append_reason(reason_codes, "manifest_missing")
    if not report_path.is_file():
        _append_reason(reason_codes, "report_missing")

    status = "failed" if reason_codes else "passed"
    acceptance_metadata = _acceptance_metadata_from_summary(summary)
    return {
        "run_id": run["run_id"],
        "status": status,
        "return_code": return_code,
        "reason_codes": reason_codes,
        "command_argv": list(run["argv"]),
        "command_args": {
            "scenario_set": run["scenario_set"],
            "diagnostic_profile": run["diagnostic_profile"],
            "top_k": run["top_k"],
            "output_root": _display_path(run_root, repo_root),
            "planner_extra_args": list(run["planner_extra_args"]),
        },
        "sample_quality_profile": run["sample_quality_profile"],
        "source_paths": {
            "output_root": _display_path(run_root, repo_root),
            "manifest": _display_path(manifest_path, repo_root),
            "summary": _display_path(summary_path, repo_root),
            "report": _display_path(report_path, repo_root),
            "stdout_log": _display_path(stdout_log, repo_root),
            "stderr_log": _display_path(stderr_log, repo_root),
        },
        "summary_path": _display_path(summary_path, repo_root),
        "report_path": _display_path(report_path, repo_root),
        "acceptance_metadata": acceptance_metadata,
        "open_grid_fallback_used": bool(summary.get("open_grid_fallback_used")) if summary else None,
        "summary": summary,
        "git": git_snapshot,
    }


def _inspect_summary_for_reasons(summary: dict[str, Any], reason_codes: list[str]) -> None:
    if summary.get("schema_version") != SUMMARY_SCHEMA_VERSION:
        _append_reason(reason_codes, "summary_schema_mismatch")

    open_grid_used = summary.get("open_grid_fallback_used")
    if open_grid_used is not False:
        _append_reason(reason_codes, "open_grid_fallback_used" if open_grid_used is True else "open_grid_fallback_unknown")

    acceptance_metadata = _acceptance_metadata_from_summary(summary)
    gate = acceptance_metadata.get("open_grid_fallback_used_gate")
    if not isinstance(gate, dict):
        gate = summary.get("open_grid_fallback_used_gate")
    if not isinstance(gate, dict):
        _append_reason(reason_codes, "open_grid_fallback_gate_missing")
        return
    if gate.get("status") != "passed":
        _append_reason(reason_codes, "open_grid_fallback_gate_failed")
    for code in gate.get("reason_codes", []):
        if isinstance(code, str) and code != "open_grid_fallback_not_used":
            _append_reason(reason_codes, code)


def _acceptance_metadata_from_summary(summary: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(summary, dict):
        return {}
    metadata = summary.get("acceptance_metadata")
    return dict(metadata) if isinstance(metadata, dict) else {}


def _build_run_index(
    batch_plan: dict[str, Any],
    run_records: list[dict[str, Any]],
    *,
    repo_root: Path,
    git_snapshot: dict[str, Any],
) -> dict[str, Any]:
    public_runs = [_public_run_record(record) for record in run_records]
    failed_run_ids = [record["run_id"] for record in public_runs if record["status"] == "failed"]
    return {
        "schema_version": RUN_INDEX_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "matrix_path": _display_path(batch_plan["matrix_path"], repo_root),
        "output_root": _display_path(batch_plan["output_root"], repo_root),
        "single_run_script": _display_path(batch_plan["single_run_script"], repo_root),
        "run_count": len(public_runs),
        "passed_count": len(public_runs) - len(failed_run_ids),
        "failed_count": len(failed_run_ids),
        "failed_run_ids": failed_run_ids,
        "git": git_snapshot,
        "runs": public_runs,
    }


def _public_run_record(record: dict[str, Any]) -> dict[str, Any]:
    public = dict(record)
    public.pop("summary", None)
    return public


def _build_evaluation_summary(
    batch_plan: dict[str, Any],
    run_records: list[dict[str, Any]],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    public_runs = [_summary_run_record(record) for record in run_records]
    failed_run_ids = [record["run_id"] for record in run_records if record["status"] == "failed"]
    parsed_summaries = [record["summary"] for record in run_records if isinstance(record.get("summary"), dict)]
    reason_code_counts = Counter(
        code
        for record in run_records
        for code in record["reason_codes"]
    )
    open_grid_count = sum(1 for summary in parsed_summaries if summary.get("open_grid_fallback_used") is True)

    return {
        "schema_version": EVALUATION_SUMMARY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "matrix_path": _display_path(batch_plan["matrix_path"], repo_root),
        "output_root": _display_path(batch_plan["output_root"], repo_root),
        "run_count": len(run_records),
        "passed_count": len(run_records) - len(failed_run_ids),
        "failed_count": len(failed_run_ids),
        "failed_run_ids": failed_run_ids,
        "failure_reason_code_counts": dict(sorted(reason_code_counts.items())),
        "open_grid_fallback_used_count": open_grid_count,
        "open_grid_fallback_gate": _aggregate_open_grid_gates(run_records),
        "path_planning_failure_count": _sum_summary_int(parsed_summaries, "path_planning_failure_count"),
        "replan_count": _sum_summary_int(parsed_summaries, "replan_count"),
        "iris_fallback_count": _sum_summary_int(parsed_summaries, "iris_fallback_count"),
        "region_graph_fallback_count": _sum_summary_int(parsed_summaries, "region_graph_fallback_count"),
        "region_graph_disconnected_count": _sum_region_graph_disconnected(parsed_summaries),
        "scenario_group_summary": _aggregate_scenario_groups(parsed_summaries),
        "source_summary_paths": [
            record["source_paths"]["summary"]
            for record in run_records
            if isinstance(record.get("summary"), dict)
        ],
        "runs": public_runs,
    }


def _summary_run_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": record["run_id"],
        "status": record["status"],
        "reason_codes": list(record["reason_codes"]),
        "command_args": dict(record["command_args"]),
        "sample_quality_profile": record["sample_quality_profile"],
        "summary_path": record["summary_path"],
        "report_path": record["report_path"],
        "acceptance_metadata": dict(record["acceptance_metadata"]),
        "open_grid_fallback_used": record["open_grid_fallback_used"],
    }


def _aggregate_open_grid_gates(run_records: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts: Counter[str] = Counter()
    by_run: dict[str, Any] = {}
    failed_run_ids: list[str] = []
    for record in run_records:
        metadata = record.get("acceptance_metadata", {})
        gate = metadata.get("open_grid_fallback_used_gate") if isinstance(metadata, dict) else None
        if not isinstance(gate, dict) and isinstance(record.get("summary"), dict):
            gate = record["summary"].get("open_grid_fallback_used_gate")
        if isinstance(gate, dict):
            status = str(gate.get("status", "unknown"))
            gate_payload = dict(gate)
        else:
            status = "missing"
            gate_payload = {
                "status": "missing",
                "expected": False,
                "actual": record.get("open_grid_fallback_used"),
                "reason_codes": ["open_grid_fallback_gate_missing"],
            }
        status_counts[status] += 1
        by_run[record["run_id"]] = gate_payload
        if status != "passed":
            failed_run_ids.append(record["run_id"])
    return {
        "status_counts": dict(sorted(status_counts.items())),
        "failed_run_ids": failed_run_ids,
        "by_run": by_run,
    }


def _aggregate_scenario_groups(summaries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    aggregate: dict[str, dict[str, Any]] = {}
    for summary in summaries:
        groups = summary.get("scenario_group_summary", {})
        if not isinstance(groups, dict):
            continue
        for group_name, group_payload in groups.items():
            if not isinstance(group_payload, dict):
                continue
            bucket = aggregate.setdefault(str(group_name), {})
            for key, value in group_payload.items():
                if isinstance(value, bool):
                    continue
                if isinstance(value, int):
                    bucket[key] = int(bucket.get(key, 0)) + value
                elif isinstance(value, float):
                    bucket[key] = float(bucket.get(key, 0.0)) + value
            if "failure_count" in bucket:
                bucket["path_planning_failure_count"] = bucket["failure_count"]
    return dict(sorted(aggregate.items()))


def _sum_summary_int(summaries: list[dict[str, Any]], key: str) -> int:
    total = 0
    for summary in summaries:
        value = summary.get(key, 0)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            total += value
    return total


def _sum_region_graph_disconnected(summaries: list[dict[str, Any]]) -> int:
    total = _sum_summary_int(summaries, "region_graph_disconnected_count")
    if total:
        return total
    return _sum_summary_int(summaries, "region_graph_start_goal_disconnected_count")


def _git_snapshot(repo_root: Path) -> dict[str, Any]:
    return {
        "parent": _git_repo_state(repo_root, repo_root=repo_root),
        "submodules": {
            name: _git_repo_state(repo_root / name, repo_root=repo_root)
            for name in SUBMODULES
        },
    }


def _git_repo_state(path: Path, *, repo_root: Path) -> dict[str, Any]:
    sha = _run_git(path, "rev-parse", "HEAD")
    branch = _run_git(path, "branch", "--show-current")
    return {
        "path": _display_path(path, repo_root),
        "sha": sha or "unknown",
        "branch": branch or None,
    }


def _run_git(path: Path, *args: str) -> str | None:
    if not path.exists():
        return None
    completed = subprocess.run(
        ["git", "-C", str(path), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def _append_reason(reason_codes: list[str], code: str) -> None:
    if code not in reason_codes:
        reason_codes.append(code)


def _resolve_path(value: str, repo_root: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return repo_root / path


def _display_path(path: Path, repo_root: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
