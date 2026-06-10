from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from git_provenance import git_snapshot as _git_snapshot


SCHEMA_VERSION = "quasi-real-lola-data-prepare-summary/v1"


def run_quasi_real_lola_data_prepare(
    *,
    manifest_path: str | Path,
    output_root: str | Path,
    repo_root: str | Path,
    download_missing: bool = True,
) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    _ensure_model_explorer_path(repo)
    from model_explorer.data.manifest import load_data_manifest, validate_data_manifest

    manifest = Path(manifest_path).resolve()
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    payload = load_data_manifest(manifest)
    initial = validate_data_manifest(manifest)
    downloaded = 0
    download_errors: list[dict[str, Any]] = []

    if initial.status != "passed" and download_missing:
        raw_dir = Path(initial.raw_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)
        for file_info in _manifest_files(payload):
            name = str(file_info.get("name", ""))
            url = str(file_info.get("url", ""))
            if not name or not url:
                continue
            target = raw_dir / name
            if target.exists() and _file_matches_basic(target, file_info):
                continue
            try:
                _download_file(url, target)
                downloaded += 1
            except Exception as exc:  # pragma: no cover - exercised through summary path.
                download_errors.append(
                    {
                        "name": name,
                        "url": url,
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )

    final = validate_data_manifest(manifest)
    issue_codes = [issue.code for issue in final.issues]
    reason_codes: list[str] = []
    if download_errors or final.status != "passed":
        if any(code in {"sha256_mismatch", "bytes_mismatch"} for code in issue_codes):
            reason_codes.append("real_map_raw_data_hash_mismatch")
        if any(code == "file_missing" for code in issue_codes) or download_errors:
            reason_codes.append("real_map_raw_data_provision_required")
        if not reason_codes:
            reason_codes.append("real_map_raw_data_validation_failed")

    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _now_iso(),
        "status": "passed" if not reason_codes else "failed",
        "reason_codes": reason_codes,
        "dataset_manifest": str(manifest),
        "dataset_id": final.dataset_id,
        "data_class": final.data_class,
        "raw_dir": final.raw_dir,
        "checked_file_count": final.checked_file_count,
        "total_bytes": final.total_bytes,
        "downloaded_file_count": downloaded,
        "missing_file_count": issue_codes.count("file_missing"),
        "file_missing_count": issue_codes.count("file_missing"),
        "sha256_mismatch_count": issue_codes.count("sha256_mismatch"),
        "bytes_mismatch_count": issue_codes.count("bytes_mismatch"),
        "download_errors": download_errors,
        "data_validation": final.to_dict(),
        "git_provenance": {"current": _git_snapshot(repo), "current_matches_sources": True},
    }
    output_path = output / "quasi-real-lola-data-prepare-summary.json"
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    summary["summary_output"] = str(output_path)
    return summary


def _manifest_files(payload: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    files: list[dict[str, Any]] = []
    for product in payload.get("products", []):
        if not isinstance(product, dict):
            continue
        for file_info in product.get("files", []):
            if isinstance(file_info, dict):
                files.append(file_info)
    return tuple(files)


def _file_matches_basic(path: Path, file_info: dict[str, Any]) -> bool:
    expected_bytes = file_info.get("bytes")
    return expected_bytes is None or path.stat().st_size == int(expected_bytes)


def _download_file(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    urls = [url]
    if url.startswith("https://"):
        urls.append("http://" + url.removeprefix("https://"))
    last_error: Exception | None = None
    for candidate in urls:
        for attempt in range(4):
            try:
                _download_file_once(candidate, target)
                return
            except Exception as exc:  # pragma: no cover - network dependent.
                last_error = exc
                if attempt < 3:
                    time.sleep(1.0 + attempt)
    if last_error is not None:
        raise last_error


def _download_file_once(url: str, target: Path) -> None:
    if url.startswith("file://"):
        shutil.copyfile(Path(urllib.request.url2pathname(url.removeprefix("file://"))), target)
        return
    source = Path(url)
    if source.exists():
        shutil.copyfile(source, target)
        return
    with urllib.request.urlopen(url, timeout=120) as response, target.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def _ensure_model_explorer_path(repo_root: Path) -> None:
    src = repo_root / "model-explorer" / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare and validate local LOLA quasi-real raw data.")
    parser.add_argument(
        "--manifest",
        default="model-explorer/data/manifests/lunar_south_pole_lro_lola_gdr_875s_20m.json",
    )
    parser.add_argument("--output-root", default="outputs/path_feedback_batch_quasi_real_map_domain_gap_v1")
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    summary = run_quasi_real_lola_data_prepare(
        manifest_path=args.manifest,
        output_root=args.output_root,
        repo_root=repo_root,
        download_missing=not args.no_download,
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
