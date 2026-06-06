from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


TRIAGE_SCHEMA_VERSION = "gcs-control-point-candidate-triage-summary/v1"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export GCS control-point candidate triage from a path-feedback summary."
    )
    parser.add_argument("--summary", required=True, help="Input path-feedback-summary.json.")
    parser.add_argument("--output-json", help="Write the triage JSON to this path.")
    parser.add_argument("--output-markdown", help="Write a compact Markdown report to this path.")
    args = parser.parse_args(argv)

    summary_path = Path(args.summary)
    try:
        summary = _load_json(summary_path)
        triage = _extract_triage(summary)
    except ValueError as exc:
        print(f"triage export error: {exc}", file=sys.stderr)
        return 2

    if args.output_json:
        _write_text(Path(args.output_json), json.dumps(triage, indent=2, ensure_ascii=False) + "\n")
    else:
        print(json.dumps(triage, indent=2, ensure_ascii=False))

    if args.output_markdown:
        _write_text(Path(args.output_markdown), _render_markdown(triage, source_summary=summary_path))

    print(
        json.dumps(
            {
                "status": "triage exported",
                "schema_version": triage["schema_version"],
                "candidate_count": triage.get("candidate_count", 0),
                "attempted_count": triage.get("attempted_count", 0),
                "success_count": triage.get("success_count", 0),
                "selected_count": triage.get("selected_count", 0),
                "route_artifact_count": triage.get("route_artifact_count", 0),
                "output_json": args.output_json,
                "output_markdown": args.output_markdown,
            },
            ensure_ascii=False,
        )
    )
    return 0


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ValueError(f"summary not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("summary must be a JSON object")
    return payload


def _extract_triage(summary: dict[str, Any]) -> dict[str, Any]:
    triage = summary.get("gcs_control_point_candidate_triage")
    if not isinstance(triage, dict):
        raise ValueError("summary does not contain gcs_control_point_candidate_triage")
    schema = triage.get("schema_version")
    if schema != TRIAGE_SCHEMA_VERSION:
        raise ValueError(f"unsupported triage schema: {schema!r}")
    return triage


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _render_markdown(triage: dict[str, Any], *, source_summary: Path) -> str:
    lines = [
        "# GCS Control-Point Candidate Triage",
        "",
        f"- source_summary: `{source_summary}`",
        f"- schema_version: `{triage.get('schema_version')}`",
        f"- candidate_count: {triage.get('candidate_count', 0)}",
        f"- attempted_count: {triage.get('attempted_count', 0)}",
        f"- success_count: {triage.get('success_count', 0)}",
        f"- selected_count: {triage.get('selected_count', 0)}",
        f"- route_artifact_count: {triage.get('route_artifact_count', 0)}",
        "",
        "## Fallback Reasons",
        "",
    ]
    fallback_counts = triage.get("fallback_reason_counts")
    if isinstance(fallback_counts, dict) and fallback_counts:
        for reason, count in sorted(fallback_counts.items()):
            lines.append(f"- `{reason}`: {count}")
    else:
        lines.append("- none")

    sweep = triage.get("calibration_sweep")
    if isinstance(sweep, dict):
        lines.extend(
            [
                "",
                "## Calibration Sweep",
                "",
                f"- schema_version: `{sweep.get('schema_version')}`",
                f"- mode: `{sweep.get('mode')}`",
                f"- solver_rerun_required: {str(bool(sweep.get('solver_rerun_required'))).lower()}",
                f"- default_change_recommended: {str(bool(sweep.get('default_change_recommended'))).lower()}",
                f"- default_change_reason: `{sweep.get('default_change_reason')}`",
            ]
        )

    candidates = triage.get("candidates")
    if isinstance(candidates, list) and candidates:
        lines.extend(
            [
                "",
                "## Candidate Rows",
                "",
                "| scenario | action | fallback | selected | route_artifact |",
                "| --- | ---: | --- | --- | --- |",
            ]
        )
        for row in candidates:
            if not isinstance(row, dict):
                continue
            lines.append(
                "| {scenario} | {action} | {fallback} | {selected} | {artifact} |".format(
                    scenario=row.get("scenario_id", ""),
                    action=row.get("action_index", ""),
                    fallback=row.get("candidate_fallback_reason") or "",
                    selected=str(bool(row.get("candidate_selected"))).lower(),
                    artifact=row.get("route_artifact") or "",
                )
            )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
