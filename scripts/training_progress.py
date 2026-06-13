from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, TextIO


EVENT_SCHEMA_VERSION = "training-progress-event/v1"
SUMMARY_SCHEMA_VERSION = "training-progress-summary/v1"
EVENTS_FILENAME = "training-progress-events.jsonl"
SUMMARY_FILENAME = "training-progress-summary.json"
PROGRESS_MODES = {"auto", "plain", "jsonl", "off"}


class ProgressReporter:
    def __init__(
        self,
        *,
        output_root: Path,
        mode: str = "auto",
        run_id: str | None = None,
        stream: TextIO | None = None,
        clock: Any | None = None,
    ) -> None:
        if mode not in PROGRESS_MODES:
            raise ValueError(f"progress mode must be one of {sorted(PROGRESS_MODES)}")
        self.output_root = Path(output_root)
        self.mode = mode
        self.run_id = run_id or os.environ.get("TRAINING_PROGRESS_RUN_ID") or str(uuid.uuid4())
        self.stream = stream if stream is not None else sys.stderr
        self._clock = clock or time.monotonic
        self._started_at = float(self._clock())
        self._events: list[dict[str, Any]] = []

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    @property
    def events_path(self) -> Path:
        return self.output_root / EVENTS_FILENAME

    @property
    def summary_path(self) -> Path:
        return self.output_root / SUMMARY_FILENAME

    def emit(
        self,
        *,
        stage: str,
        status: str,
        current: int | None = None,
        total: int | None = None,
        round_index: int | None = None,
        step_index: int | None = None,
        message: str | None = None,
        output_root: str | Path | None = None,
        summary_path: str | Path | None = None,
        reason_codes: list[str] | tuple[str, ...] | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.enabled:
            return {}
        event = {
            "schema_version": EVENT_SCHEMA_VERSION,
            "run_id": self.run_id,
            "stage": stage,
            "status": status,
            "current": current,
            "total": total,
            "round_index": round_index,
            "step_index": step_index,
            "message": message,
            "elapsed_seconds": round(float(self._clock()) - self._started_at, 6),
            "output_root": str(output_root if output_root is not None else self.output_root),
            "summary_path": None if summary_path is None else str(summary_path),
            "reason_codes": [str(reason) for reason in reason_codes or []],
            "metrics": _jsonable(metrics or {}),
        }
        self.output_root.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        self._events.append(event)
        if self._renders_plain():
            print(_plain_line(event), file=self.stream, flush=True)
        return event

    def finalize(
        self,
        *,
        status: str,
        readiness_status: str | None = None,
        recommended_debug_artifact: str | Path | None = None,
    ) -> dict[str, Any]:
        if not self.enabled:
            return {}
        events = self._read_events()
        last_event = events[-1] if events else {}
        failed_events = [event for event in events if event.get("status") == "failed"]
        debug_artifact = recommended_debug_artifact
        if debug_artifact is None and last_event.get("status") == "failed":
            debug_artifact = last_event.get("summary_path")
        summary = {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "run_id": self.run_id,
            "status": status,
            "event_count": len(events),
            "total_stage_count": len({event.get("stage") for event in events if event.get("stage")}),
            "failed_stage_count": len(failed_events),
            "last_stage": last_event.get("stage"),
            "last_status": last_event.get("status"),
            "last_reason_codes": list(last_event.get("reason_codes") or []),
            "readiness_status": readiness_status,
            "recommended_debug_artifact": None if debug_artifact is None else str(debug_artifact),
            "events_path": str(self.events_path),
            "elapsed_seconds": round(float(self._clock()) - self._started_at, 6),
        }
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return summary

    def _renders_plain(self) -> bool:
        if self.mode == "plain":
            return True
        if self.mode != "auto":
            return False
        return bool(getattr(self.stream, "isatty", lambda: False)()) and not os.environ.get("CI")

    def _read_events(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        if self.events_path.is_file():
            for line in self.events_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                events.append(json.loads(line))
        elif self._events:
            events.extend(self._events)
        return events


def add_progress_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--progress",
        choices=sorted(PROGRESS_MODES),
        default=None,
        help="training progress output mode; default comes from config/env or auto",
    )


def progress_mode(value: str | None = None, *, config: dict[str, Any] | None = None) -> str:
    if value:
        return value
    env_value = os.environ.get("TRAINING_PROGRESS_MODE")
    if env_value:
        return env_value if env_value in PROGRESS_MODES else "auto"
    if isinstance(config, dict):
        configured = (config.get("progress") or {}).get("mode") if isinstance(config.get("progress"), dict) else None
        if configured in PROGRESS_MODES:
            return str(configured)
    return "auto"


def progress_output_root(output_root: Path) -> Path:
    env_root = os.environ.get("TRAINING_PROGRESS_ROOT")
    return Path(env_root) if env_root else output_root


def make_progress_reporter(
    *,
    output_root: Path,
    mode: str | None = None,
    config: dict[str, Any] | None = None,
    run_id: str | None = None,
) -> ProgressReporter:
    return ProgressReporter(
        output_root=progress_output_root(output_root),
        mode=progress_mode(mode, config=config),
        run_id=run_id,
    )


def progress_child_env(
    env: dict[str, str],
    *,
    output_root: Path,
    mode: str,
    run_id: str,
) -> dict[str, str]:
    child = dict(env)
    child["TRAINING_PROGRESS_ROOT"] = str(output_root)
    child["TRAINING_PROGRESS_MODE"] = mode
    child["TRAINING_PROGRESS_RUN_ID"] = run_id
    return child


def ppo_update_progress_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    training_result = summary.get("training_result") if isinstance(summary.get("training_result"), dict) else {}
    non_finite_fields = (
        "loss_non_finite_count",
        "non_finite_gradient_count",
        "non_finite_reward_count",
        "non_finite_return_count",
        "non_finite_advantage_count",
    )
    return {
        "optimizer_train_transition_count": _int_value(summary.get("optimizer_train_transition_count")),
        "epochs": _int_value(training_result.get("epochs"), default=1),
        "loss": _number_or_none(training_result.get("final_total_loss")),
        "approx_kl": _number_or_none(summary.get("approx_kl")),
        "max_grad_norm_after_clip": _number_or_none(summary.get("max_grad_norm_after_clip")),
        "parameter_l2_delta": _number_or_none(summary.get("parameter_l2_delta")),
        "non_finite_count": sum(_int_value(summary.get(field)) for field in non_finite_fields),
    }


def sequential_progress_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "episode_count": _int_value(summary.get("episode_count")),
        "step_count": _int_value(summary.get("step_count")),
        "raw_rejected_policy_choice_count": _int_value(summary.get("canary_rejected_policy_choice_count")),
        "controlled_regression_count": sum(
            _int_value(summary.get(field))
            for field in (
                "cumulative_path_cost_regression_count",
                "cumulative_risk_regression_count",
                "cumulative_source_selection_regression_count",
                "controlled_path_cost_regression_count",
                "controlled_risk_regression_count",
            )
        ),
    }


def collector_progress_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "episode_count": _int_value(summary.get("episode_count")),
        "step_count": _int_value(summary.get("step_count")),
        "trainable_transition_count": _int_value(summary.get("ppo_trainable_transition_count")),
        "diagnostic_transition_count": _int_value(summary.get("diagnostic_transition_count")),
        "raw_rejected_policy_choice_count": _int_value(summary.get("canary_rejected_policy_choice_count")),
        "controlled_regression_count": sum(
            _int_value(summary.get(field))
            for field in (
                "path_cost_regression_count",
                "risk_regression_count",
                "source_selection_regression_count",
            )
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Emit or summarize training progress telemetry.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    emit_parser = subparsers.add_parser("emit")
    emit_parser.add_argument("--output-root", required=True)
    emit_parser.add_argument("--mode", choices=sorted(PROGRESS_MODES), default="auto")
    emit_parser.add_argument("--run-id", default=None)
    emit_parser.add_argument("--stage", required=True)
    emit_parser.add_argument("--status", required=True)
    emit_parser.add_argument("--current", type=int, default=None)
    emit_parser.add_argument("--total", type=int, default=None)
    emit_parser.add_argument("--round-index", type=int, default=None)
    emit_parser.add_argument("--step-index", type=int, default=None)
    emit_parser.add_argument("--message", default=None)
    emit_parser.add_argument("--summary-path", default=None)
    emit_parser.add_argument("--reason-code", action="append", default=[])
    emit_parser.add_argument("--metric", action="append", default=[])

    final_parser = subparsers.add_parser("finalize")
    final_parser.add_argument("--output-root", required=True)
    final_parser.add_argument("--mode", choices=sorted(PROGRESS_MODES), default="auto")
    final_parser.add_argument("--run-id", default=None)
    final_parser.add_argument("--status", required=True)
    final_parser.add_argument("--readiness-status", default=None)
    final_parser.add_argument("--recommended-debug-artifact", default=None)

    args = parser.parse_args(argv)
    reporter = ProgressReporter(output_root=Path(args.output_root), mode=args.mode, run_id=args.run_id)
    if args.command == "emit":
        reporter.emit(
            stage=args.stage,
            status=args.status,
            current=args.current,
            total=args.total,
            round_index=args.round_index,
            step_index=args.step_index,
            message=args.message,
            summary_path=args.summary_path,
            reason_codes=args.reason_code,
            metrics=_parse_metrics(args.metric),
        )
    else:
        reporter.finalize(
            status=args.status,
            readiness_status=args.readiness_status,
            recommended_debug_artifact=args.recommended_debug_artifact,
        )
    return 0


def _plain_line(event: dict[str, Any]) -> str:
    total = event.get("total")
    current = event.get("current")
    progress = f"{current}/{total}" if current is not None and total is not None else "-"
    message = event.get("message") or event.get("stage")
    return f"[progress] {progress} {event.get('stage')} {event.get('status')}: {message}"


def _parse_metrics(values: list[str]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for item in values:
        if "=" not in item:
            continue
        key, raw_value = item.split("=", 1)
        metrics[key] = _coerce_scalar(raw_value)
    return metrics


def _coerce_scalar(value: str) -> Any:
    if value in {"true", "false"}:
        return value == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(key): _jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [_jsonable(item) for item in value]
        return str(value)


def _int_value(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _number_or_none(value: Any) -> float | int | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result.is_integer():
        return int(result)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
