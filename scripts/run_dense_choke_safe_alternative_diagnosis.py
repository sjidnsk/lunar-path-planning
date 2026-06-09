from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SUMMARY_SCHEMA_VERSION = "dense-choke-safe-alternative-diagnosis-summary/v1"
DENSE_CHOKE_FAMILY = "dense_choke_safe_bypass"
NEXT_DENSE_CHOKE_OPPORTUNITY_GAP = "dense_choke_opportunity_generation_gap"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnose dense-choke canary safe alternative gaps.")
    parser.add_argument("--batch-root", required=True)
    parser.add_argument(
        "--opportunity-summary",
        default="policy-gated-canary-opportunity-summary.json",
    )
    parser.add_argument(
        "--visual-output",
        default="outputs/dense_choke_safe_alternative_visual_diagnostics_v1/index.html",
    )
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    batch_root = _resolve_path(args.batch_root, repo_root)
    opportunity_path = batch_root / args.opportunity_summary
    visual_path = _resolve_path(args.visual_output, repo_root)

    opportunity_summary = _load_json(opportunity_path)
    summary, markdown, html = build_diagnosis(
        opportunity_summary=opportunity_summary,
        batch_root=batch_root,
        visual_path=visual_path,
        repo_root=repo_root,
    )

    summary_path = batch_root / "dense-choke-safe-alternative-diagnosis-summary.json"
    markdown_path = batch_root / "dense-choke-safe-alternative-diagnosis.md"
    _write_text(summary_path, json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    _write_text(markdown_path, markdown)
    _write_text(visual_path, html)

    print(
        json.dumps(
            {
                "status": summary["status"],
                "reason_codes": summary["reason_codes"],
                "dense_choke_opportunity_context_count": summary[
                    "dense_choke_opportunity_context_count"
                ],
                "dense_choke_acceptable_alternative_count": summary[
                    "dense_choke_acceptable_alternative_count"
                ],
                "next_required_change": summary["next_required_change"],
                "summary": _display_path(summary_path, repo_root),
            },
            ensure_ascii=False,
        )
    )
    return 1 if summary["status"] == "failed" else 0


def build_diagnosis(
    *,
    opportunity_summary: dict[str, Any],
    batch_root: Path,
    visual_path: Path,
    repo_root: Path,
) -> tuple[dict[str, Any], str, str]:
    opportunities = [
        item
        for item in opportunity_summary.get("opportunities", [])
        if item.get("scenario_group") == DENSE_CHOKE_FAMILY
    ]
    reason_counts: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    acceptable_count = 0
    for opportunity in opportunities:
        for alternative in opportunity.get("alternatives", []):
            reasons = list(alternative.get("canary_gate_rejection_reason_codes", []))
            reason_counts.update(reasons)
            if alternative.get("canary_gate_acceptable") is True:
                acceptable_count += 1
            rows.append(
                {
                    "run_id": opportunity.get("run_id"),
                    "scenario_id": opportunity.get("scenario_id"),
                    "source_context_id": opportunity.get("source_context_id"),
                    "action_index": alternative.get("action_index"),
                    "source_action_index": alternative.get("source_action_index"),
                    "policy_target_cell": alternative.get("policy_target_cell"),
                    "path_cost_delta": alternative.get("path_cost_delta"),
                    "risk_delta": alternative.get("risk_delta"),
                    "canary_gate_acceptable": bool(alternative.get("canary_gate_acceptable")),
                    "canary_gate_rejection_reason_codes": reasons,
                    "action_mask_valid": "invalid_action_mask" not in reasons,
                    "source_bound": (
                        alternative.get("source_action_index") is not None
                        and alternative.get("action_index") == alternative.get("source_action_index")
                    ),
                }
            )

    reason_codes: list[str] = []
    if not opportunities:
        reason_codes.append("dense_choke_opportunity_missing")
    next_required_change = None
    if acceptable_count <= 0:
        next_required_change = NEXT_DENSE_CHOKE_OPPORTUNITY_GAP

    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "failed" if reason_codes else "passed",
        "reason_codes": reason_codes,
        "batch_root": _display_path(batch_root, repo_root),
        "visual_output": _display_path(visual_path, repo_root),
        "dense_choke_opportunity_context_count": len(opportunities),
        "dense_choke_alternative_count": len(rows),
        "dense_choke_acceptable_alternative_count": acceptable_count,
        "dense_choke_rejection_reason_counts": dict(sorted(reason_counts.items())),
        "dense_choke_candidate_rows": rows,
        "next_required_change": next_required_change,
        "hard_positive_added_count": 0,
    }
    return summary, _markdown(summary), _html(summary)


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Dense-Choke Safe Alternative Diagnosis",
        "",
        f"- Status: `{summary['status']}`",
        f"- Dense-choke opportunity contexts: `{summary['dense_choke_opportunity_context_count']}`",
        f"- Alternatives: `{summary['dense_choke_alternative_count']}`",
        f"- Acceptable alternatives: `{summary['dense_choke_acceptable_alternative_count']}`",
        f"- Rejection reasons: `{json.dumps(summary['dense_choke_rejection_reason_counts'], sort_keys=True)}`",
        f"- Next required change: `{summary['next_required_change']}`",
        "",
        "| run | scenario | action | source action | cell | path delta | risk delta | acceptable | reasons |",
        "|---|---|---:|---:|---|---:|---:|---|---|",
    ]
    for row in summary["dense_choke_candidate_rows"]:
        lines.append(
            "| {run_id} | {scenario_id} | {action_index} | {source_action_index} | {cell} | {path} | {risk} | {acceptable} | {reasons} |".format(
                run_id=row.get("run_id"),
                scenario_id=row.get("scenario_id"),
                action_index=row.get("action_index"),
                source_action_index=row.get("source_action_index"),
                cell=row.get("policy_target_cell"),
                path=row.get("path_cost_delta"),
                risk=row.get("risk_delta"),
                acceptable=row.get("canary_gate_acceptable"),
                reasons=",".join(row.get("canary_gate_rejection_reason_codes", [])),
            )
        )
    return "\n".join(lines) + "\n"


def _html(summary: dict[str, Any]) -> str:
    rows = "\n".join(
        "<tr><td>{scenario}</td><td>{action}</td><td>{source}</td><td>{cell}</td><td>{path}</td><td>{risk}</td><td>{acceptable}</td><td>{reasons}</td></tr>".format(
            scenario=row.get("scenario_id"),
            action=row.get("action_index"),
            source=row.get("source_action_index"),
            cell=row.get("policy_target_cell"),
            path=row.get("path_cost_delta"),
            risk=row.get("risk_delta"),
            acceptable=row.get("canary_gate_acceptable"),
            reasons=", ".join(row.get("canary_gate_rejection_reason_codes", [])),
        )
        for row in summary["dense_choke_candidate_rows"]
    )
    return f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>Dense-Choke Safe Alternative Diagnosis</title>
<style>body{{font-family:sans-serif;margin:24px}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:6px}}</style></head>
<body>
<h1>Dense-Choke Safe Alternative Diagnosis</h1>
<p>Acceptable alternatives: {summary['dense_choke_acceptable_alternative_count']}</p>
<p>Rejection reasons: {json.dumps(summary['dense_choke_rejection_reason_counts'], sort_keys=True)}</p>
<table><thead><tr><th>Scenario</th><th>Action</th><th>Source Action</th><th>Cell</th><th>Path Delta</th><th>Risk Delta</th><th>Acceptable</th><th>Reasons</th></tr></thead>
<tbody>{rows}</tbody></table>
</body></html>
"""


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _resolve_path(value: str | Path, repo_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
