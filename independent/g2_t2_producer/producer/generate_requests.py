from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from .canonical import (
    canonical_json_bytes,
    canonical_npz_bytes,
    derive_seed,
    domain_hash,
    sha256_bytes,
)
from .finite_graph import reachability_certificate
from .models import validate_truth_row


_PLATFORMS = ("wheel", "legged", "hopper")
_REJECTION_REASONS = {
    "wheel": "G2I_W_CLOSED_OBSTACLE_CONTACT",
    "legged": "G2I_L_FOOTHOLD_OBSTACLE",
    "hopper": "G2I_H_LANDING_FOOTPRINT",
}


def _grid_graph(platform: str, kind: str) -> tuple[dict[str, Any], str, str]:
    nodes = [f"n:{x}:{y}" for x in range(5) for y in range(5)]
    start = "n:0:2"
    goal = "n:4:2"
    if kind == "reachable":
        blocked: set[tuple[int, int]] = set()
        slack = 40
    elif kind == "hard_reachable":
        blocked = {(1, 2), (2, 2), (3, 2)}
        slack = 10
    elif kind == "unreachable":
        blocked = {(2, y) for y in range(5)}
        slack = 40
    else:
        raise ValueError(f"unknown request kind: {kind}")
    edges: list[dict[str, Any]] = []
    for x in range(5):
        for y in range(5):
            source = f"n:{x}:{y}"
            for action_index, (dx, dy) in enumerate(
                ((1, 0), (-1, 0), (0, 1), (0, -1))
            ):
                tx, ty = x + dx, y + dy
                if not (0 <= tx < 5 and 0 <= ty < 5):
                    continue
                target = f"n:{tx}:{ty}"
                accepted = (x, y) not in blocked and (tx, ty) not in blocked
                edges.append(
                    {
                        "accepted": accepted,
                        "cost_milli": 1000,
                        "edge_id": f"{platform}-{x}-{y}-{action_index}",
                        "from_node": source,
                        "reject_reason": None
                        if accepted
                        else _REJECTION_REASONS[platform],
                        "safety_slack_mm": slack if accepted else -1,
                        "to_node": target,
                    }
                )
    return {
        "schema_version": "g2-request-finite-graph/v1",
        "platform_kind": platform,
        "nodes": nodes,
        "candidate_edges": edges,
        "expected_node_count": 25,
        "expected_candidate_edge_count": len(edges),
    }, start, goal


def _difficulty(
    certificate: dict[str, Any],
    graph: dict[str, Any],
    start: str,
    goal: str,
) -> tuple[str, dict[str, int]]:
    if not bool(certificate["oracle_reachable"]):
        return "unreachable", {
            "detour_ratio_milli": 0,
            "minimum_safety_slack_mm": -1,
            "path_primitive_count": 0,
        }
    coordinates = {
        node: tuple(int(value) for value in node.split(":")[1:]) for node in graph["nodes"]
    }
    direct_steps = abs(coordinates[start][0] - coordinates[goal][0]) + abs(
        coordinates[start][1] - coordinates[goal][1]
    )
    path_steps = len(certificate["path_edge_ids"])
    detour_ratio = path_steps * 1000 // max(1, direct_steps)
    edge_by_id = {edge["edge_id"]: edge for edge in graph["candidate_edges"]}
    minimum_slack = min(
        int(edge_by_id[edge_id]["safety_slack_mm"])
        for edge_id in certificate["path_edge_ids"]
    )
    hard = detour_ratio >= 1500 and minimum_slack <= 20
    return ("hard_reachable" if hard else "reachable"), {
        "detour_ratio_milli": detour_ratio,
        "minimum_safety_slack_mm": minimum_slack,
        "path_primitive_count": path_steps,
    }


def _terrain_provenance(
    *,
    scale: str,
    base_index: int,
    root_seed: int,
    lola_provenance: dict[str, Any],
) -> dict[str, Any]:
    proxy_seed = derive_seed(root_seed, "shared", "requests", scale, base_index)
    if scale == "standard":
        return {
            "closed_obstacle_family": base_index % 7,
            "corridor_width_mm": 800 + (base_index % 11) * 50,
            "height_step_mm": (base_index % 9) * 10,
            "physical_obstacle_cells_written": False,
            "slope_cdeg": (base_index % 17) * 100,
            "source_kind": "procedural_simulation_proxy/v1",
            "source_seed": proxy_seed,
            "unknown_family": base_index % 5,
        }
    provenance = dict(lola_provenance)
    provenance.update(
        {
            "interpolation_spec_sha256": domain_hash(
                "g2-lola-interpolation/v1", b"integer-bilinear-macro-only"
            ),
            "proxy_generator_sha256": domain_hash(
                "g2-kilometer-micro-proxy/v1", b"closed-integer-polygons"
            ),
            "proxy_seed": proxy_seed,
            "roi_index": base_index,
        }
    )
    return provenance


def _pool_kind(base_index: int) -> str:
    bucket = base_index % 10
    if bucket < 6:
        return "reachable"
    if bucket < 8:
        return "hard_reachable"
    return "unreachable"


def terrain_npz_bytes(provenance: dict[str, Any]) -> bytes:
    seed = int(provenance.get("source_seed", provenance.get("proxy_seed", 0)))
    base_height = seed % 10000
    height_mm = np.fromfunction(
        lambda y, x: base_height + x * ((seed % 13) + 1) + y * ((seed % 17) + 1),
        (5, 5),
        dtype=int,
    ).astype("<i4")
    cell_class = np.zeros((5, 5), dtype="u1")
    cell_class[(seed // 7) % 5, (seed // 11) % 5] = 2
    known = np.ones((5, 5), dtype="u1")
    known[(seed // 13) % 5, (seed // 17) % 5] = 0
    confidence_ppm = np.full((5, 5), 1_000_000, dtype="<u4")
    confidence_ppm[known == 0] = 0
    return canonical_npz_bytes(
        {
            "height_mm": height_mm,
            "cell_class": cell_class,
            "known": known,
            "confidence_ppm": confidence_ppm,
        },
        {
            "cell_size_mm": 500 if "source_seed" in provenance else 20000,
            "geometry_schema_version": "g2-neutral-terrain/v1",
            "provenance": provenance,
        },
    )

def generate_request_pool(
    specification: dict[str, Any],
    *,
    profile_record_sha256: dict[str, str],
    lola_provenance: dict[str, Any],
) -> list[dict[str, Any]]:
    root_seed = int(specification["root_seed"])
    if set(profile_record_sha256) != set(_PLATFORMS):
        raise ValueError("profile hash map must cover exactly three platforms")
    if any(
        len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
        for value in profile_record_sha256.values()
    ):
        raise ValueError("invalid profile record SHA-256")
    if lola_provenance.get("physical_obstacle_cells_written") is not False:
        raise ValueError("kilometer synthetic obstacles cannot be physical")
    if (
        lola_provenance.get("micro_source_kind")
        != "synthetic_terrain_obstacle_proxy/v1"
    ):
        raise ValueError("kilometer microterrain source kind mismatch")

    pool: list[dict[str, Any]] = []
    for scale, count in (
        ("standard", int(specification["request_pool"]["standard_base_terrains"])),
        ("kilometer", int(specification["request_pool"]["kilometer_base_terrains"])),
    ):
        for base_index in range(count):
            desired_kind = _pool_kind(base_index)
            provenance = _terrain_provenance(
                scale=scale,
                base_index=base_index,
                root_seed=root_seed,
                lola_provenance=lola_provenance,
            )
            terrain_sha = sha256_bytes(terrain_npz_bytes(provenance))
            for platform in _PLATFORMS:
                graph, start, goal = _grid_graph(platform, desired_kind)
                certificate = reachability_certificate(graph, start, goal)
                difficulty, difficulty_witness = _difficulty(
                    certificate, graph, start, goal
                )
                if difficulty != desired_kind:
                    raise ValueError(
                        f"request construction mismatch: {desired_kind} != {difficulty}"
                    )
                certificate_payload = {
                    **certificate,
                    "difficulty_witness": difficulty_witness,
                    "graph": graph,
                    "platform_kind": platform,
                    "scale": scale,
                }
                certificate_sha = sha256_bytes(
                    canonical_json_bytes(certificate_payload)
                )
                objective = {
                    "kind": "minimum_resource_complete_l2/v1",
                    "resource_weight_milli": 1000,
                }
                request_sha = domain_hash(
                    "g2-truth-request/v1",
                    platform.encode("ascii"),
                    scale.encode("ascii"),
                    terrain_sha.encode("ascii"),
                    start.encode("ascii"),
                    goal.encode("ascii"),
                    canonical_json_bytes(objective),
                    certificate_sha.encode("ascii"),
                    profile_record_sha256[platform].encode("ascii"),
                )
                row = {
                    "determinism_seed": derive_seed(
                        root_seed,
                        platform,
                        "requests",
                        scale,
                        base_index,
                    ),
                    "difficulty_class": difficulty,
                    "goal": {"node_id": goal},
                    "objective": objective,
                    "oracle_reachable": bool(certificate["oracle_reachable"]),
                    "platform_kind": platform,
                    "producer_implementation_sha256": domain_hash(
                        "g2-request-producer/v1", b"integer-grid-ucs-hash-selection"
                    ),
                    "profile_or_parameter_record_sha256": profile_record_sha256[
                        platform
                    ],
                    "request_id": f"g2i-req-{platform}-{scale}-{request_sha[:20]}",
                    "resource_budget": {
                        "max_path_primitives": 128 if scale == "standard" else 512
                    },
                    "scale": scale,
                    "schema_version": "g2-truth-request/v1",
                    "selection_seed": derive_seed(
                        root_seed,
                        platform,
                        "request-selection",
                        scale,
                        0,
                    ),
                    "source_ids": [f"{scale}-terrain-{base_index:03d}"],
                    "start": {"node_id": start},
                    "terrain_provenance": provenance,
                    "terrain_sha256": terrain_sha,
                    "truth_certificate": certificate_payload,
                    "truth_certificate_sha256": certificate_sha,
                    "truth_request_sha256": request_sha,
                }
                validate_truth_row(row)
                pool.append(row)
    return pool


def select_requests(
    pool: list[dict[str, Any]],
    specification: dict[str, Any],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in pool:
        grouped[
            (
                str(row["platform_kind"]),
                str(row["scale"]),
                str(row["difficulty_class"]),
            )
        ].append(row)
    selected: list[dict[str, Any]] = []
    for platform in _PLATFORMS:
        for scale in ("standard", "kilometer"):
            for difficulty in ("reachable", "hard_reachable", "unreachable"):
                quota = int(specification["request_mix"][scale][difficulty])
                candidates = grouped[(platform, scale, difficulty)]
                ranked = sorted(
                    candidates,
                    key=lambda row: domain_hash(
                        "g2-selection/v1",
                        str(row["selection_seed"]).encode("ascii"),
                        platform.encode("ascii"),
                        scale.encode("ascii"),
                        str(row["truth_request_sha256"]).encode("ascii"),
                    ),
                )
                if len(ranked) < quota:
                    raise ValueError(
                        f"request stratum shortfall: {platform}/{scale}/{difficulty}"
                    )
                for rank, row in enumerate(ranked[:quota]):
                    output = dict(row)
                    output["selection_rank"] = rank
                    output["selection_hash"] = domain_hash(
                        "g2-selection/v1",
                        str(row["selection_seed"]).encode("ascii"),
                        platform.encode("ascii"),
                        scale.encode("ascii"),
                        str(row["truth_request_sha256"]).encode("ascii"),
                    )
                    selected.append(output)
    return selected
