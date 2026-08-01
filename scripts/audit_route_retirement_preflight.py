from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence

Classification = Literal["protected", "manual_review", "retire_candidate", "unclassified"]


@dataclass(frozen=True)
class CandidateRecord:
    path: str
    classification: Classification
    matched_rules: tuple[str, ...]
    tracked: bool
    ignored: bool


@dataclass(frozen=True)
class GraphSummary:
    git_commit_hash: str
    analyzed_files: int
    nodes: int
    edges: int
    layers: int


_POLICY_SCHEMA = "route_retirement_policy/v1"
_PRECEDENCE = ("protected", "manual_review", "retire_candidate", "unclassified")
_GRAPH_COMMIT = "1682d7f9f755eb59c85e6d0ceaaa6f6d5d203dfd"
_GRAPH_COUNTS = (252, 1480, 2710, 7)
_BASELINE_COMMIT = "6a4c2dd0352fd6c1918a5eef39c9783b9d3c5c65"
_IGNORED_OUTPUT_SAMPLE_LIMIT = 16
_GRAPH_PATH = Path("D:/codex/project/lunar-path-planning/.ua/knowledge-graph.json")
_BACKUP_ROOT = Path("D:/CodexDownloads/lunar-path-planning-backups/2026-08-01-6a4c2dd")
_POLICY_RELPATH = Path("configs/route_retirement_policy_v1.json")
_EXPECTED_GITLINKS = {
    "dev-platform-constraints": "61e9fa8afd09db83632456bdcf181c222ee13513",
    "model-explorer": "b547a997d94ad199c822136d1ae345e180b87ca7",
    "path-planner": "2f6378d3c47da027c0d4146d94cab881b8f2a594",
    "visual-workbench": "9acc4ce83fc2884221a9759fa00824692eab3315",
}
_EXPECTED_BUNDLE_COMMITS = {
    "dev-platform-constraints.bundle": _EXPECTED_GITLINKS["dev-platform-constraints"],
    "lunar-path-planning.bundle": _BASELINE_COMMIT,
    "model-explorer.bundle": _EXPECTED_GITLINKS["model-explorer"],
    "path-planner.bundle": _EXPECTED_GITLINKS["path-planner"],
    "visual-workbench.bundle": _EXPECTED_GITLINKS["visual-workbench"],
}
_EXPECTED_BUNDLE_HASHES = {
    "dev-platform-constraints.bundle": "e5b156378377d05986c5f1f8dc1bc3ea4d8d0419868dd1788803d214cba111f5",
    "lunar-path-planning.bundle": "a23f5d663868692c2cbf52dc6713ecbfcec4c5cdb6711aeb1546157e711c7dea",
    "model-explorer.bundle": "891860eb54c5ec5cebdd44b84bc4af1a9eede90c659de112f531cd9296185a55",
    "path-planner.bundle": "37216255c231cf490e91d4f6d74c242b15e43570a8ee0f3ef035614eb97a4bc1",
    "visual-workbench.bundle": "11053c6199c0f79a52668f3c7bfd42515d86620c005bd0937e5be016e483b977",
}


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _normalize_repo_path(path: str) -> str:
    normalized = path.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    posix_path = PurePosixPath(normalized)
    if (
        not normalized
        or posix_path.is_absolute()
        or re.match(r"^[A-Za-z]:/", normalized)
        or ".." in posix_path.parts
    ):
        raise ValueError(f"repository path must be relative: {path!r}")
    return posix_path.as_posix()


def _path_key(path: Path) -> str:
    return os.path.normcase(str(path.resolve()))


def _validate_fixed_paths(
    repo_root: Path,
    policy_path: Path,
    graph_path: Path,
    backup_root: Path,
) -> None:
    expected_policy = repo_root.resolve() / _POLICY_RELPATH
    if _path_key(policy_path) != _path_key(expected_policy):
        raise ValueError(f"policy path must be {expected_policy.as_posix()}")
    if _path_key(graph_path) != _path_key(_GRAPH_PATH):
        raise ValueError(f"knowledge graph path must be {_GRAPH_PATH.as_posix()}")
    if _path_key(backup_root) != _path_key(_BACKUP_ROOT):
        raise ValueError(f"backup root must be {_BACKUP_ROOT.as_posix()}")


def _validate_fixed_cli_inputs(
    repo_root: Path,
    policy_path: Path,
    graph_path: Path,
    backup_root: Path,
    baseline_commit: str,
) -> None:
    _validate_fixed_paths(repo_root, policy_path, graph_path, backup_root)
    if baseline_commit != _BASELINE_COMMIT:
        raise ValueError(f"baseline commit must be {_BASELINE_COMMIT}")


def load_policy(path: Path) -> Mapping[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != _POLICY_SCHEMA:
        raise ValueError(f"expected policy schema {_POLICY_SCHEMA}")
    if tuple(data.get("classification_precedence", ())) != _PRECEDENCE:
        raise ValueError(f"classification_precedence must be {_PRECEDENCE!r}")
    for classification in _PRECEDENCE[:-1]:
        key = f"{classification}_globs"
        rules = data.get(key)
        if not isinstance(rules, list) or not all(isinstance(rule, str) for rule in rules):
            raise ValueError(f"{key} must be a list of strings")
    forbidden_roots = data.get("forbidden_delete_roots")
    if not isinstance(forbidden_roots, list) or not all(isinstance(item, str) for item in forbidden_roots):
        raise ValueError("forbidden_delete_roots must be a list of strings")
    return _freeze(data)


def classify_path(path: str, policy: Mapping[str, Any]) -> CandidateRecord:
    normalized = _normalize_repo_path(path)
    matches_by_classification: dict[str, list[str]] = {}
    all_matches: list[str] = []
    for classification in policy["classification_precedence"]:
        rules = policy.get(f"{classification}_globs", ())
        matches = [rule for rule in rules if fnmatchcase(normalized, rule)]
        matches_by_classification[classification] = matches
        all_matches.extend(matches)

    selected: Classification = "unclassified"
    for classification in policy["classification_precedence"]:
        if classification == "unclassified" or matches_by_classification[classification]:
            selected = classification
            break
    return CandidateRecord(
        path=normalized,
        classification=selected,
        matched_rules=tuple(all_matches),
        tracked=False,
        ignored=False,
    )


def _read_graph(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("knowledge graph must be a JSON object")
    return data


def validate_graph(path: Path) -> GraphSummary:
    data = _read_graph(path)
    project = data.get("project")
    nodes = data.get("nodes")
    edges = data.get("edges")
    layers = data.get("layers")
    if not isinstance(project, dict) or not all(isinstance(items, list) for items in (nodes, edges, layers)):
        raise ValueError("knowledge graph is missing project, nodes, edges or layers")
    file_paths = {
        node.get("filePath")
        for node in nodes
        if isinstance(node, dict) and isinstance(node.get("filePath"), str)
    }
    summary = GraphSummary(
        git_commit_hash=str(project.get("gitCommitHash", "")),
        analyzed_files=len(file_paths),
        nodes=len(nodes),
        edges=len(edges),
        layers=len(layers),
    )
    actual = (summary.analyzed_files, summary.nodes, summary.edges, summary.layers)
    if summary.git_commit_hash != _GRAPH_COMMIT:
        raise ValueError(f"unexpected graph commit: {summary.git_commit_hash}")
    if actual != _GRAPH_COUNTS:
        raise ValueError(f"unexpected graph counts: {actual}")
    return summary


def _run(args: Sequence[str], cwd: Path, *, input_bytes: bytes | None = None) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        list(args),
        cwd=cwd,
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def list_tracked_paths(repo_root: Path) -> tuple[str, ...]:
    result = _run(("git", "ls-files", "-z"), repo_root)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace").strip())
    paths = (_normalize_repo_path(item.decode("utf-8")) for item in result.stdout.split(b"\0") if item)
    return tuple(sorted(paths))


def _is_permission_failure(result: subprocess.CompletedProcess[bytes]) -> bool:
    stderr = result.stderr.decode("utf-8", errors="replace").lower()
    return any(marker in stderr for marker in ("permission denied", "access is denied", "operation not permitted"))


def _audit_ignored_outputs(repo_root: Path) -> dict[str, Any]:
    try:
        result = _run(("git", "ls-files", "--others", "--ignored", "--exclude-standard", "-z"), repo_root)
    except PermissionError:
        return {"status": "permission_denied", "paths": []}
    if result.returncode != 0:
        if _is_permission_failure(result):
            return {"status": "permission_denied", "paths": []}
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace").strip())
    enumerated = tuple(sorted({
        _normalize_repo_path(item.decode("utf-8"))
        for item in result.stdout.split(b"\0")
        if item
    }))
    if not enumerated:
        return {"status": "ok", "paths": []}
    payload = b"\0".join(path.encode("utf-8") for path in enumerated) + b"\0"
    try:
        checked = _run(("git", "check-ignore", "-z", "--stdin"), repo_root, input_bytes=payload)
    except PermissionError:
        return {"status": "permission_denied", "paths": []}
    if checked.returncode not in (0, 1):
        if _is_permission_failure(checked):
            return {"status": "permission_denied", "paths": []}
        raise RuntimeError(checked.stderr.decode("utf-8", errors="replace").strip())
    confirmed = sorted({
        _normalize_repo_path(item.decode("utf-8"))
        for item in checked.stdout.split(b"\0")
        if item
    })
    return {"status": "ok", "paths": confirmed}


def _compact_ignored_outputs(ignored_outputs: Mapping[str, Any]) -> dict[str, Any]:
    paths = tuple(sorted(str(path) for path in ignored_outputs["paths"]))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
    return {
        "status": ignored_outputs["status"],
        "count": len(paths),
        "sha256": digest.hexdigest(),
        "sample": list(paths[:_IGNORED_OUTPUT_SAMPLE_LIMIT]),
    }


def _json_text(value: Mapping[str, Any]) -> str:
    if "text" in value:
        return str(value["text"])
    return base64.b64decode(str(value["bytes"])).decode("utf-8", errors="replace")


def _rg_references(repo_root: Path, candidate_paths: Sequence[str]) -> dict[str, list[dict[str, Any]]]:
    evidence: dict[str, list[dict[str, Any]]] = {path: [] for path in candidate_paths}
    tracked_sources = set(list_tracked_paths(repo_root))
    for offset in range(0, len(candidate_paths), 24):
        batch = candidate_paths[offset : offset + 24]
        args = [
            "rg",
            "--json",
            "--fixed-strings",
            "--line-number",
            "--hidden",
            "--glob",
            "!.git/**",
        ]
        for candidate_path in batch:
            args.extend(("-e", candidate_path))
        args.append(".")
        result = _run(args, repo_root)
        if result.returncode not in (0, 1):
            raise RuntimeError(result.stderr.decode("utf-8", errors="replace").strip())
        for raw_line in result.stdout.splitlines():
            event = json.loads(raw_line.decode("utf-8"))
            if event.get("type") != "match":
                continue
            data = event["data"]
            source_path = _normalize_repo_path(_json_text(data["path"]))
            if source_path not in tracked_sources:
                continue
            line_text = _json_text(data["lines"]).rstrip("\r\n")
            for candidate_path in batch:
                if candidate_path in line_text and source_path != candidate_path:
                    evidence[candidate_path].append(
                        {
                            "path": source_path,
                            "line": int(data["line_number"]),
                            "text": line_text,
                        }
                    )
    for candidate_path in evidence:
        evidence[candidate_path].sort(key=lambda item: (item["path"], item["line"], item["text"]))
    return evidence


def audit_references(
    repo_root: Path,
    records: Sequence[CandidateRecord],
    graph_path: Path,
) -> dict[str, Any]:
    candidate_paths = tuple(sorted({_normalize_repo_path(record.path) for record in records}))
    rg_evidence = _rg_references(repo_root, candidate_paths)
    graph = _read_graph(graph_path)
    node_paths = {
        node["id"]: _normalize_repo_path(node["filePath"])
        for node in graph["nodes"]
        if isinstance(node, dict)
        and isinstance(node.get("id"), str)
        and isinstance(node.get("filePath"), str)
    }
    graph_evidence: dict[str, list[dict[str, Any]]] = {path: [] for path in candidate_paths}
    candidate_set = set(candidate_paths)
    for edge in graph["edges"]:
        if not isinstance(edge, dict):
            continue
        source_path = node_paths.get(edge.get("source"))
        target_path = node_paths.get(edge.get("target"))
        if source_path not in candidate_set and target_path not in candidate_set:
            continue
        item = {
            "source_path": source_path,
            "target_path": target_path,
            "type": str(edge.get("type", "")),
            "direction": str(edge.get("direction", "")),
        }
        if source_path in candidate_set:
            graph_evidence[source_path].append(item)
        if target_path in candidate_set and target_path != source_path:
            graph_evidence[target_path].append(item)
    result: dict[str, Any] = {}
    for candidate_path in candidate_paths:
        edges = sorted(
            graph_evidence[candidate_path],
            key=lambda item: (
                item["source_path"] or "",
                item["target_path"] or "",
                item["type"],
                item["direction"],
            ),
        )
        result[candidate_path] = {"rg": rg_evidence[candidate_path], "graph_edges": edges}
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_bundle_hashes(hash_path: Path) -> dict[str, str]:
    expected: dict[str, str] = {}
    for line in hash_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, raw_path = line.split(maxsplit=1)
        expected[PureWindowsPath(raw_path.strip()).name] = digest.lower()
    return expected


def _bundle_heads(result: subprocess.CompletedProcess[bytes]) -> tuple[str, ...]:
    if result.returncode != 0:
        return ()
    heads = {
        line.split(maxsplit=1)[0].decode("ascii")
        for line in result.stdout.splitlines()
        if re.match(rb"^[0-9a-f]{40}(?:\s|$)", line)
    }
    return tuple(sorted(heads))


def _audit_backups(
    repo_root: Path,
    backup_root: Path,
    *,
    expected_hashes: Mapping[str, str] = _EXPECTED_BUNDLE_HASHES,
    expected_commits: Mapping[str, str] = _EXPECTED_BUNDLE_COMMITS,
) -> dict[str, Any]:
    bundle_names = tuple(sorted(expected_commits))
    if set(expected_hashes) != set(expected_commits):
        raise ValueError("expected bundle hash and commit inventories must match")
    hash_path = backup_root / "bundle-hashes.txt"
    declared_hashes = _expected_bundle_hashes(hash_path) if hash_path.is_file() else {}
    records: list[dict[str, Any]] = []
    for bundle_name in bundle_names:
        bundle_path = backup_root / bundle_name
        expected_hash = expected_hashes[bundle_name]
        expected_commit = expected_commits[bundle_name]
        declared_hash = declared_hashes.get(bundle_name)
        if not bundle_path.is_file():
            records.append(
                {
                    "name": bundle_name,
                    "expected_sha256": expected_hash,
                    "declared_sha256": declared_hash,
                    "actual_sha256": None,
                    "hash_matches": False,
                    "bundle_verified": False,
                    "expected_commit": expected_commit,
                    "contains_expected_commit": False,
                    "heads": [],
                    "status": "missing",
                }
            )
            continue
        actual = _sha256(bundle_path)
        verify = _run(("git", "bundle", "verify", str(bundle_path)), repo_root)
        listed = _run(("git", "bundle", "list-heads", str(bundle_path)), repo_root)
        heads = _bundle_heads(listed)
        hash_matches = expected_hash == actual
        bundle_verified = verify.returncode == 0
        contains_expected_commit = expected_commit in heads
        if not hash_matches:
            status = "hash_mismatch"
        elif not bundle_verified or listed.returncode != 0:
            status = "verification_failed"
        elif not contains_expected_commit:
            status = "missing_expected_commit"
        else:
            status = "verified"
        records.append(
            {
                "name": bundle_name,
                "expected_sha256": expected_hash,
                "declared_sha256": declared_hash,
                "actual_sha256": actual,
                "hash_matches": hash_matches,
                "bundle_verified": bundle_verified,
                "expected_commit": expected_commit,
                "contains_expected_commit": contains_expected_commit,
                "heads": list(heads),
                "status": status,
            }
        )
    hash_manifest_matches_expected = declared_hashes == dict(expected_hashes)
    return {
        "root": backup_root.as_posix(),
        "hash_manifest": hash_path.name,
        "hash_manifest_sha256": _sha256(hash_path) if hash_path.is_file() else None,
        "hash_manifest_matches_expected": hash_manifest_matches_expected,
        "records": records,
        "all_verified": hash_manifest_matches_expected and all(record["status"] == "verified" for record in records),
    }


def _audit_baseline_gitlinks(
    repo_root: Path,
    baseline_commit: str,
    expected_gitlinks: Mapping[str, str] = _EXPECTED_GITLINKS,
) -> dict[str, Any]:
    result = _run(("git", "ls-tree", baseline_commit, "--", *sorted(expected_gitlinks)), repo_root)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace").strip())
    actual: dict[str, str] = {}
    for line in result.stdout.decode("utf-8").splitlines():
        match = re.match(r"^160000 commit ([0-9a-f]{40})\t(.+)$", line)
        if match:
            actual[_normalize_repo_path(match.group(2))] = match.group(1)
    records = []
    for path in sorted(expected_gitlinks):
        expected = expected_gitlinks[path]
        observed = actual.get(path)
        status = "matched" if observed == expected else "missing" if observed is None else "mismatch"
        records.append(
            {
                "path": path,
                "expected_commit": expected,
                "actual_commit": observed,
                "status": status,
            }
        )
    return {"records": records, "all_match": all(record["status"] == "matched" for record in records)}


def _resolve_commit(repo_root: Path, commit: str) -> str:
    result = _run(("git", "rev-parse", "--verify", f"{commit}^{{commit}}"), repo_root)
    if result.returncode != 0:
        raise ValueError(f"invalid baseline commit: {commit}")
    return result.stdout.decode("ascii").strip()


def _is_ancestor(repo_root: Path, ancestor: str, descendant: str) -> bool:
    result = _run(("git", "merge-base", "--is-ancestor", ancestor, descendant), repo_root)
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace").strip())
    return result.returncode == 0


def _blocking_references(
    candidate_path: str,
    audit: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    for evidence in audit["rg"]:
        source_path = evidence["path"]
        source_classification = classify_path(source_path, policy).classification
        if source_classification != "retire_candidate":
            blockers.append(
                {
                    "evidence": "rg",
                    "source_path": source_path,
                    "source_classification": source_classification,
                    "line": evidence["line"],
                }
            )
    for edge in audit["graph_edges"]:
        if edge["target_path"] != candidate_path or not edge["source_path"]:
            continue
        source_path = edge["source_path"]
        source_classification = classify_path(source_path, policy).classification
        if source_classification != "retire_candidate":
            blockers.append(
                {
                    "evidence": "graph",
                    "source_path": source_path,
                    "source_classification": source_classification,
                    "edge_type": edge["type"],
                }
            )
    unique = {json.dumps(item, sort_keys=True, ensure_ascii=False): item for item in blockers}
    return [unique[key] for key in sorted(unique)]


def build_manifest(
    repo_root: Path,
    policy_path: Path,
    graph_path: Path,
    backup_root: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    _validate_fixed_paths(repo_root, policy_path, graph_path, backup_root)
    policy = load_policy(policy_path)
    graph_summary = validate_graph(graph_path)
    baseline_commit = _resolve_commit(repo_root, _BASELINE_COMMIT)
    baseline_tree = _audit_baseline_gitlinks(repo_root, baseline_commit)
    tracked_paths = list_tracked_paths(repo_root)
    ignored_outputs = _audit_ignored_outputs(repo_root)
    ignored_paths = set(ignored_outputs["paths"])
    classified: list[CandidateRecord] = []
    for path in tracked_paths:
        record = classify_path(path, policy)
        classified.append(
            CandidateRecord(
                path=record.path,
                classification=record.classification,
                matched_rules=record.matched_rules,
                tracked=True,
                ignored=record.path in ignored_paths,
            )
        )
    retirement_records = [record for record in classified if record.classification == "retire_candidate"]
    reference_audit = audit_references(repo_root, retirement_records, graph_path)
    records: list[dict[str, Any]] = []
    blocking_reasons: list[dict[str, Any]] = []
    for record in classified:
        item = asdict(record)
        item["matched_rules"] = list(record.matched_rules)
        audit = reference_audit.get(record.path, {"rg": [], "graph_edges": []})
        blockers = _blocking_references(record.path, audit, policy) if record.classification == "retire_candidate" else []
        item["blocking_references"] = blockers
        records.append(item)
        if blockers:
            blocking_reasons.append(
                {
                    "path": record.path,
                    "reason": "referenced_by_retained_path",
                    "reference_count": len(blockers),
                }
            )
    backups = _audit_backups(repo_root, backup_root)
    backups["bundles_all_verified"] = backups["all_verified"]
    backups["all_verified"] = backups["bundles_all_verified"] and baseline_tree["all_match"]
    return {
        "schema_version": "route_retirement_preflight/v1",
        "baseline_commit": baseline_commit,
        "graph": {
            **asdict(graph_summary),
            "is_ancestor_of_baseline": _is_ancestor(repo_root, graph_summary.git_commit_hash, baseline_commit),
        },
        "ignored_outputs": ignored_outputs,
        "ignored_state_status": ignored_outputs["status"],
        "records": records,
        "reference_audit": reference_audit,
        "backups": backups,
        "baseline_tree": baseline_tree,
        "blocking_reasons": blocking_reasons,
    }


def _local_tags(repo_root: Path, baseline_commit: str) -> list[str]:
    result = _run(
        (
            "git",
            "for-each-ref",
            "--format=%(refname:short)%00%(objectname)%00%(*objectname)",
            "refs/tags",
        ),
        repo_root,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace").strip())
    tags: list[str] = []
    for line in result.stdout.decode("utf-8").splitlines():
        fields = line.split("\0")
        if len(fields) != 3:
            continue
        name, direct, peeled = fields
        if (peeled or direct) == baseline_commit:
            tags.append(name)
    return sorted(tags)


def _classify_remote_tag_status(tags: Sequence[str], baseline_commit: str, stdout: bytes) -> str:
    refs: dict[str, dict[str, str]] = {}
    for line in stdout.decode("utf-8").splitlines():
        fields = line.split("\t", 1)
        if len(fields) != 2:
            continue
        commit, ref = fields
        if not ref.startswith("refs/tags/"):
            continue
        name = ref[len("refs/tags/") :]
        peeled = name.endswith("^{}")
        if peeled:
            name = name[:-3]
        refs.setdefault(name, {})["peeled" if peeled else "direct"] = commit
    matched = 0
    for tag in tags:
        remote = refs.get(tag)
        if remote is None:
            continue
        target = remote.get("peeled") or remote.get("direct")
        if target != baseline_commit:
            return "mismatch"
        matched += 1
    if matched == len(tags):
        return "pushed"
    if matched:
        return "partial"
    return "not_pushed"


def _remote_tag_status(repo_root: Path, tags: Sequence[str], baseline_commit: str) -> str:
    if not tags:
        return "missing_local_tag"
    patterns: list[str] = []
    for tag in tags:
        patterns.extend((f"refs/tags/{tag}", f"refs/tags/{tag}^{{}}"))
    result = _run(("git", "ls-remote", "--tags", "origin", *patterns), repo_root)
    if result.returncode != 0:
        return "remote_unavailable"
    return _classify_remote_tag_status(tags, baseline_commit, result.stdout)


def _hash_baseline_tests(root: Path | None) -> dict[str, Any]:
    if root is None:
        return {"status": "not_provided", "records": []}
    if not root.is_dir():
        return {"status": "missing", "records": []}
    try:
        paths = sorted(path for path in root.rglob("*.xml") if path.is_file())
        records = [
            {"path": path.relative_to(root).as_posix(), "sha256": _sha256(path)}
            for path in paths
        ]
    except PermissionError:
        return {"status": "permission_denied", "records": []}
    return {"status": "hashed", "records": records}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _classification_counts(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        classification: sum(record["classification"] == classification for record in records)
        for classification in _PRECEDENCE
    }


def _render_report(summary: Mapping[str, Any], blocking_reasons: Sequence[Mapping[str, Any]]) -> str:
    counts = summary["classification_counts"]
    lines = [
        "# 路线退役只读预检报告",
        "",
        f"- 基线提交：`{summary['baseline_commit']}`",
        f"- 图谱提交：`{summary['graph']['git_commit_hash']}`",
        f"- 图谱是基线祖先：`{str(summary['graph']['is_ancestor_of_baseline']).lower()}`",
        f"- 远端标签状态：`{summary['remote_tag_status']}`",
        f"- 备份全部验证：`{str(summary['backups_all_verified']).lower()}`",
        f"- 受保护：{counts['protected']}",
        f"- 人工复核：{counts['manual_review']}",
        f"- 退役候选：{counts['retire_candidate']}",
        f"- 未分类：{counts['unclassified']}",
        f"- 阻塞候选：{summary['blocking_candidate_count']}",
        "- owner approval=0；本 artifact 不授权任何物理删除。",
        "",
        "本报告仅提供审计证据，不执行文件系统或 Git 变更。",
    ]
    if blocking_reasons:
        lines.extend(("", "## 阻塞引用", ""))
        for item in blocking_reasons:
            lines.append(f"- `{item['path']}`：{item['reference_count']} 条保留侧引用")
    return "\n".join(lines) + "\n"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成确定性的路线退役只读预检审计")
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--knowledge-graph", required=True, type=Path)
    parser.add_argument("--backup-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--baseline-commit", required=True)
    parser.add_argument("--baseline-tests-root", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    repo_root = args.repo_root.resolve()
    _validate_fixed_cli_inputs(
        repo_root,
        args.policy,
        args.knowledge_graph,
        args.backup_root,
        args.baseline_commit,
    )
    baseline_commit = _resolve_commit(repo_root, args.baseline_commit)
    manifest = build_manifest(repo_root, args.policy, args.knowledge_graph, args.backup_root)
    manifest["baseline_commit"] = baseline_commit
    manifest["graph"]["is_ancestor_of_baseline"] = _is_ancestor(
        repo_root,
        manifest["graph"]["git_commit_hash"],
        baseline_commit,
    )
    local_tags = _local_tags(repo_root, baseline_commit)
    remote_tag_status = _remote_tag_status(repo_root, local_tags, baseline_commit)
    baseline_tests = _hash_baseline_tests(args.baseline_tests_root)
    records = manifest["records"]
    ignored_outputs = _compact_ignored_outputs(manifest["ignored_outputs"])
    summary = {
        "schema_version": "route_retirement_preflight_summary/v1",
        "baseline_commit": baseline_commit,
        "graph": manifest["graph"],
        "classification_counts": _classification_counts(records),
        "blocking_candidate_count": len(manifest["blocking_reasons"]),
        "backups_all_verified": manifest["backups"]["all_verified"],
        "baseline_tree_all_match": manifest["baseline_tree"]["all_match"],
        "ignored_outputs": ignored_outputs,
        "ignored_state_status": ignored_outputs["status"],
        "local_tags": local_tags,
        "remote_tag_status": remote_tag_status,
        "baseline_tests": baseline_tests,
    }
    candidate_manifest = {
        "schema_version": "route_retirement_candidate_manifest/v1",
        "baseline_commit": baseline_commit,
        "records": records,
        "blocking_reasons": manifest["blocking_reasons"],
    }
    reference_manifest = {
        "schema_version": "route_retirement_reference_audit/v1",
        "baseline_commit": baseline_commit,
        "records": manifest["reference_audit"],
    }
    backup_manifest = {
        "schema_version": "route_retirement_backup_manifest/v1",
        "baseline_commit": baseline_commit,
        "local_tags": local_tags,
        "remote_tag_status": remote_tag_status,
        "baseline_tests": baseline_tests,
        "baseline_tree": manifest["baseline_tree"],
        **manifest["backups"],
    }
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "summary.json", summary)
    _write_json(output_root / "candidate-manifest.json", candidate_manifest)
    _write_json(output_root / "reference-audit.json", reference_manifest)
    _write_json(output_root / "backup-manifest.json", backup_manifest)
    (output_root / "report.md").write_text(
        _render_report(summary, manifest["blocking_reasons"]),
        encoding="utf-8",
        newline="\n",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
