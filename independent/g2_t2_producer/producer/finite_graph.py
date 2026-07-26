from __future__ import annotations

import heapq
from collections import defaultdict
from decimal import Decimal
from typing import Any

from .canonical import canonical_json_bytes, domain_hash, sha256_bytes
from .oracle_hopper import evaluate_hopper, validate_hopper_parameter_record


def _validate_graph(graph: dict[str, Any]) -> None:
    nodes = list(graph["nodes"])
    edges = list(graph["candidate_edges"])
    if len(nodes) != len(set(nodes)):
        raise ValueError("duplicate graph node")
    if len(nodes) != int(graph["expected_node_count"]):
        raise ValueError("node envelope incomplete")
    if len(edges) != int(graph["expected_candidate_edge_count"]):
        raise ValueError("candidate edge envelope incomplete")
    node_set = set(nodes)
    edge_ids: set[str] = set()
    for edge in edges:
        edge_id = str(edge["edge_id"])
        if edge_id in edge_ids:
            raise ValueError(f"duplicate edge_id: {edge_id}")
        edge_ids.add(edge_id)
        if edge["from_node"] not in node_set or edge["to_node"] not in node_set:
            raise ValueError(f"dangling edge: {edge_id}")
        if int(edge["cost_milli"]) < 0:
            raise ValueError(f"negative edge cost: {edge_id}")
        if bool(edge["accepted"]) and edge.get("reject_reason") is not None:
            raise ValueError(f"accepted edge has reject reason: {edge_id}")
        if not bool(edge["accepted"]) and not edge.get("reject_reason"):
            raise ValueError(f"rejected edge lacks reason: {edge_id}")


def _accepted_adjacency(graph: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    adjacency: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in graph["candidate_edges"]:
        if bool(edge["accepted"]):
            adjacency[str(edge["from_node"])].append(edge)
    for edges in adjacency.values():
        edges.sort(key=lambda edge: (str(edge["to_node"]), str(edge["edge_id"])))
    return adjacency


def _reconstruct(
    predecessors: dict[str, tuple[str, str]],
    start_node: str,
    goal_node: str,
) -> tuple[list[str], list[str]]:
    if goal_node == start_node:
        return [start_node], []
    if goal_node not in predecessors:
        return [], []
    nodes = [goal_node]
    edges: list[str] = []
    cursor = goal_node
    while cursor != start_node:
        parent, edge_id = predecessors[cursor]
        nodes.append(parent)
        edges.append(edge_id)
        cursor = parent
    nodes.reverse()
    edges.reverse()
    return nodes, edges


def _result(
    *,
    nodes: list[str],
    distances: dict[str, int],
    predecessors: dict[str, tuple[str, str]],
    settled_order: list[str],
    start_node: str,
    goal_node: str,
) -> dict[str, Any]:
    path_nodes, path_edges = _reconstruct(predecessors, start_node, goal_node)
    return {
        "cost_milli": distances.get(goal_node),
        "distance_labels_milli": {
            node: distances.get(node) for node in sorted(nodes)
        },
        "path_edge_ids": path_edges,
        "path_nodes": path_nodes,
        "predecessors": {
            node: {"edge_id": value[1], "parent": value[0]}
            for node, value in sorted(predecessors.items())
        },
        "settled_order": settled_order,
    }

def heap_dijkstra(
    graph: dict[str, Any],
    start_node: str,
    goal_node: str,
) -> dict[str, Any]:
    _validate_graph(graph)
    nodes = list(graph["nodes"])
    if start_node not in nodes or goal_node not in nodes:
        raise ValueError("start or goal outside graph")
    adjacency = _accepted_adjacency(graph)
    distances = {start_node: 0}
    predecessors: dict[str, tuple[str, str]] = {}
    settled: set[str] = set()
    settled_order: list[str] = []
    queue: list[tuple[int, str]] = [(0, start_node)]
    while queue:
        distance, node = heapq.heappop(queue)
        if node in settled or distance != distances.get(node):
            continue
        settled.add(node)
        settled_order.append(node)
        for edge in adjacency.get(node, []):
            target = str(edge["to_node"])
            candidate = distance + int(edge["cost_milli"])
            current = distances.get(target)
            if current is None or candidate < current:
                distances[target] = candidate
                predecessors[target] = (node, str(edge["edge_id"]))
                heapq.heappush(queue, (candidate, target))
    return _result(
        nodes=nodes,
        distances=distances,
        predecessors=predecessors,
        settled_order=settled_order,
        start_node=start_node,
        goal_node=goal_node,
    )


def quadratic_dijkstra(
    graph: dict[str, Any],
    start_node: str,
    goal_node: str,
) -> dict[str, Any]:
    _validate_graph(graph)
    nodes = sorted(graph["nodes"])
    if start_node not in nodes or goal_node not in nodes:
        raise ValueError("start or goal outside graph")
    adjacency = _accepted_adjacency(graph)
    infinity = 10**30
    distances = {node: infinity for node in nodes}
    distances[start_node] = 0
    predecessors: dict[str, tuple[str, str]] = {}
    unvisited = set(nodes)
    settled_order: list[str] = []
    while unvisited:
        node = min(unvisited, key=lambda item: (distances[item], item))
        if distances[node] == infinity:
            break
        unvisited.remove(node)
        settled_order.append(node)
        for edge in adjacency.get(node, []):
            target = str(edge["to_node"])
            if target not in unvisited:
                continue
            candidate = distances[node] + int(edge["cost_milli"])
            if candidate < distances[target]:
                distances[target] = candidate
                predecessors[target] = (node, str(edge["edge_id"]))
    finite_distances = {
        node: distance for node, distance in distances.items() if distance != infinity
    }
    return _result(
        nodes=nodes,
        distances=finite_distances,
        predecessors=predecessors,
        settled_order=settled_order,
        start_node=start_node,
        goal_node=goal_node,
    )


def reachability_certificate(
    graph: dict[str, Any],
    start_node: str,
    goal_node: str,
) -> dict[str, Any]:
    solved = heap_dijkstra(graph, start_node, goal_node)
    reachable_nodes = sorted(
        node
        for node, distance in solved["distance_labels_milli"].items()
        if distance is not None
    )
    reachable_set = set(reachable_nodes)
    reachable = solved["cost_milli"] is not None
    frontier_cut = sorted(
        (
            {
                "edge_id": str(edge["edge_id"]),
                "from_node": str(edge["from_node"]),
                "reject_reason": str(edge["reject_reason"]),
                "to_node": str(edge["to_node"]),
            }
            for edge in graph["candidate_edges"]
            if not bool(edge["accepted"])
            and edge["from_node"] in reachable_set
            and edge["to_node"] not in reachable_set
        ),
        key=lambda edge: edge["edge_id"],
    )
    certificate = {
        "cost_milli": solved["cost_milli"],
        "distance_labels_sha256": sha256_bytes(
            canonical_json_bytes(solved["distance_labels_milli"])
        ),
        "frontier_cut": frontier_cut,
        "open_set_exhausted": not reachable,
        "oracle_reachable": reachable,
        "path_edge_ids": solved["path_edge_ids"],
        "path_nodes": solved["path_nodes"],
        "reachable_nodes": reachable_nodes,
        "reachable_set_sha256": domain_hash(
            "g2-reachable-set/v1", canonical_json_bytes(reachable_nodes)
        ),
        "settled_order": solved["settled_order"],
    }
    return certificate


def _edge(
    edge_id: str,
    source: str,
    target: str,
    *,
    accepted: bool,
    cost_milli: int,
    reason: str | None,
    slack: int,
) -> dict[str, Any]:
    return {
        "accepted": accepted,
        "cost_milli": cost_milli,
        "edge_id": edge_id,
        "from_node": source,
        "reject_reason": reason,
        "safety_slack_mm": slack,
        "to_node": target,
    }


def _wheel_small_map() -> tuple[dict[str, Any], str, str]:
    nodes = [f"w:{x}:{y}:{heading}" for x in range(6) for y in range(6) for heading in range(8)]
    actions = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1), (0, 0, 0))
    blocked = {(2, 2), (3, 2)}
    edges: list[dict[str, Any]] = []
    for x in range(6):
        for y in range(6):
            for heading in range(8):
                source = f"w:{x}:{y}:{heading}"
                for action_index, (dx, dy, dh) in enumerate(actions):
                    tx, ty, th = x + dx, y + dy, (heading + dh) % 8
                    inside = 0 <= tx < 6 and 0 <= ty < 6
                    obstacle = inside and (tx, ty) in blocked
                    accepted = inside and not obstacle
                    target = f"w:{tx}:{ty}:{th}" if inside else source
                    cost = 100 if action_index == 6 else (250 if action_index in {4, 5} else 1000)
                    reason = None if accepted else (
                        "G2I_W_CLOSED_OBSTACLE_CONTACT" if obstacle else "G2I_W_OUTSIDE_MAP"
                    )
                    edges.append(
                        _edge(
                            f"w-e-{x}-{y}-{heading}-{action_index}",
                            source,
                            target,
                            accepted=accepted,
                            cost_milli=cost,
                            reason=reason,
                            slack=20 if accepted else -1,
                        )
                    )
    return {
        "schema_version": "g2-finite-graph/v1",
        "platform_kind": "wheel",
        "nodes": nodes,
        "candidate_edges": edges,
        "expected_node_count": 288,
        "expected_candidate_edge_count": 2016,
        "node_envelope": {"cell_x": 6, "cell_y": 6, "heading": 8},
        "action_envelope": {"per_node": 7},
    }, "w:0:0:0", "w:5:5:0"


def _legged_small_map() -> tuple[dict[str, Any], str, str]:
    nodes = [
        f"l:{x}:{y}:{phase}:{com}"
        for x in range(5)
        for y in range(5)
        for phase in range(4)
        for com in range(2)
    ]
    actions = ((1, 0), (-1, 0), (0, 1), (0, -1), (0, 0))
    blocked = {(2, 2)}
    edges: list[dict[str, Any]] = []
    for x in range(5):
        for y in range(5):
            for phase in range(4):
                for com in range(2):
                    source = f"l:{x}:{y}:{phase}:{com}"
                    for action_index, (dx, dy) in enumerate(actions):
                        tx, ty = x + dx, y + dy
                        next_phase = (phase + 1) % 4
                        next_com = 1 - com
                        inside = 0 <= tx < 5 and 0 <= ty < 5
                        obstacle = inside and (tx, ty) in blocked
                        accepted = inside and not obstacle
                        target = (
                            f"l:{tx}:{ty}:{next_phase}:{next_com}" if inside else source
                        )
                        reason = None if accepted else (
                            "G2I_L_FOOTHOLD_OBSTACLE"
                            if obstacle
                            else "G2I_L_GRID_DOMAIN"
                        )
                        edges.append(
                            _edge(
                                f"l-e-{x}-{y}-{phase}-{com}-{action_index}",
                                source,
                                target,
                                accepted=accepted,
                                cost_milli=300 if action_index == 4 else 1200,
                                reason=reason,
                                slack=15 if accepted else -1,
                            )
                        )
    return {
        "schema_version": "g2-finite-graph/v1",
        "platform_kind": "legged",
        "nodes": nodes,
        "candidate_edges": edges,
        "expected_node_count": 200,
        "expected_candidate_edge_count": 1000,
        "node_envelope": {
            "body_cell_x": 5,
            "body_cell_y": 5,
            "com_offset": 2,
            "moving_phase": 4,
        },
        "action_envelope": {"per_node": 5},
    }, "l:0:0:0:0", "l:4:4:0:0"


def _hopper_action(action_index: int) -> dict[str, int]:
    speed_index = action_index // 48
    remainder = action_index % 48
    elevation_index = remainder // 16
    azimuth_index = remainder % 16
    return {
        "azimuth_index": azimuth_index,
        "azimuth_mdeg": azimuth_index * 22500,
        "elevation_index": elevation_index,
        "elevation_mdeg": (30000, 45000, 60000)[elevation_index],
        "speed_index": speed_index,
        "speed_mm_s": (1000, 1500, 2000, 2500)[speed_index],
    }


def _hopper_small_map(
    parameter_record: dict[str, Any],
) -> tuple[dict[str, Any], str, str]:
    validate_hopper_parameter_record(parameter_record)
    nodes = [f"h:pad:{pad}" for pad in range(9)]
    edges: list[dict[str, Any]] = []
    for pad in range(9):
        source = f"h:pad:{pad}"
        for action_index in range(192):
            advance = 1 + action_index % 3
            destination = pad + advance
            inside = destination < 9
            target = f"h:pad:{destination}" if inside else source
            action = _hopper_action(action_index)
            terrain = {
                "arc_clearance_slack_mm": 20 if inside else -1,
                "arc_inside_map": inside,
                "arc_known": True,
                "landing_footprint_clearance_mm": 20,
                "landing_height_error_mm": 0,
                "landing_height_tolerance_mm": 50,
                "landing_halfwidth_mm": 5000,
                "landing_sigma_mm": 289,
                "landing_slope_cdeg": 0,
                "launch_clearance_mm": 20,
            }
            decision = evaluate_hopper(
                {
                    "action": action,
                    "numeric_state": "decided",
                    "terrain": terrain,
                    "touchdown_speed_mm_s": int(action["speed_mm_s"]),
                },
                parameter_record,
            )
            accepted = inside and bool(decision["oracle_safe"])
            edges.append(
                _edge(
                    f"h-e-{pad}-{action_index}",
                    source,
                    target,
                    accepted=accepted,
                    cost_milli=500 + int(action["speed_index"]) * 125,
                    reason=None if accepted else str(decision["oracle_reason_code"]),
                    slack=int(decision["safety_slacks"].get("arc_clearance_mm", -1)),
                )
            )
    return {
        "schema_version": "g2-finite-graph/v1",
        "platform_kind": "hopper",
        "nodes": nodes,
        "candidate_edges": edges,
        "expected_node_count": 9,
        "expected_candidate_edge_count": 1728,
        "node_envelope": {"landing_pad": 9},
        "action_envelope": {"per_node": 192},
    }, "h:pad:0", "h:pad:8"


def _bounded_exhaustive_crosscheck(
    graph: dict[str, Any],
    start_node: str,
    goal_node: str,
    optimum: int,
    *,
    cap: int = 10000,
) -> dict[str, Any]:
    adjacency = _accepted_adjacency(graph)
    reverse_graph = {
        **graph,
        "candidate_edges": [
            {**edge, "from_node": edge["to_node"], "to_node": edge["from_node"]}
            for edge in graph["candidate_edges"]
        ],
    }
    reverse = quadratic_dijkstra(reverse_graph, goal_node, start_node)
    lower = reverse["distance_labels_milli"]
    path_count = 0
    matched = False
    truncated = False

    def walk(node: str, cost: int, visited: set[str]) -> None:
        nonlocal path_count, matched, truncated
        if truncated:
            return
        if node == goal_node:
            path_count += 1
            matched = matched or cost == optimum
            if path_count >= cap:
                truncated = True
            return
        for edge in adjacency.get(node, []):
            target = str(edge["to_node"])
            if target in visited:
                continue
            lower_bound = lower.get(target)
            if lower_bound is None:
                continue
            next_cost = cost + int(edge["cost_milli"])
            if next_cost + int(lower_bound) > optimum:
                continue
            walk(target, next_cost, visited | {target})

    walk(start_node, 0, {start_node})
    return {
        "bound_cost_decimal": format(Decimal(optimum) / Decimal(1000), "f"),
        "matched": matched,
        "path_count": path_count,
        "truncated_at": cap if truncated else None,
    }


def _build_optimum_artifact(
    platform_kind: str,
    graph: dict[str, Any],
    start_node: str,
    goal_node: str,
) -> dict[str, Any]:
    primary = heap_dijkstra(graph, start_node, goal_node)
    secondary = quadratic_dijkstra(graph, start_node, goal_node)
    if primary["cost_milli"] is None or primary["cost_milli"] != secondary["cost_milli"]:
        raise ValueError(f"solver disagreement for {platform_kind}")
    optimum = int(primary["cost_milli"])
    exhaustive = _bounded_exhaustive_crosscheck(
        graph, start_node, goal_node, optimum
    )
    if not exhaustive["matched"]:
        raise ValueError(f"bounded exhaustive mismatch for {platform_kind}")
    edge_bytes = canonical_json_bytes(graph["candidate_edges"])
    graph_bytes = canonical_json_bytes(graph)
    certificate = {
        "certificate_schema_version": "g2-small-map-certificate/v1",
        "graph": graph,
        "heap_result": primary,
        "quadratic_cost_decimal": format(
            Decimal(int(secondary["cost_milli"])) / Decimal(1000), "f"
        ),
        "quadratic_result": secondary,
        "start_node": start_node,
        "goal_node": goal_node,
    }
    certificate_sha = sha256_bytes(canonical_json_bytes(certificate))
    row = {
        "candidate_edge_count": len(graph["candidate_edges"]),
        "case_id": f"g2i-optimum-{platform_kind}-small-map-v1",
        "certificate_sha256": certificate_sha,
        "completeness": {
            "actual_candidate_edge_count": len(graph["candidate_edges"]),
            "actual_node_count": len(graph["nodes"]),
            "complete": len(graph["nodes"]) == graph["expected_node_count"]
            and len(graph["candidate_edges"])
            == graph["expected_candidate_edge_count"],
            "expected_candidate_edge_count": graph["expected_candidate_edge_count"],
            "expected_node_count": graph["expected_node_count"],
        },
        "cost_unit": "resource_milli",
        "distance_labels_sha256": sha256_bytes(
            canonical_json_bytes(primary["distance_labels_milli"])
        ),
        "edge_list_sha256": sha256_bytes(edge_bytes),
        "exhaustive_crosscheck": exhaustive,
        "goal_node": goal_node,
        "map_sha256": domain_hash("g2-small-map/v1", graph_bytes),
        "node_count": len(graph["nodes"]),
        "objective": {"distance_weight_milli": 1000, "resource_weight_milli": 1},
        "optimal_path_edge_ids": primary["path_edge_ids"],
        "optimum_cost_decimal": format(Decimal(optimum) / Decimal(1000), "f"),
        "platform_kind": platform_kind,
        "safe_edge_count": sum(
            bool(edge["accepted"]) for edge in graph["candidate_edges"]
        ),
        "schema_version": "g2-small-map-optimum/v1",
        "settled_order_sha256": sha256_bytes(
            canonical_json_bytes(primary["settled_order"])
        ),
        "solver_implementation_sha256": domain_hash(
            "g2-solver-implementation/v1", b"heap+quadratic+bounded-exhaustive"
        ),
        "start_node": start_node,
    }
    return {"certificate": certificate, "row": row}


def build_all_optima(
    specification: dict[str, Any],
    *,
    hopper_parameter_record: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    del specification
    wheel_graph, wheel_start, wheel_goal = _wheel_small_map()
    legged_graph, legged_start, legged_goal = _legged_small_map()
    hopper_graph, hopper_start, hopper_goal = _hopper_small_map(
        hopper_parameter_record
    )
    return {
        "wheel": _build_optimum_artifact(
            "wheel", wheel_graph, wheel_start, wheel_goal
        ),
        "legged": _build_optimum_artifact(
            "legged", legged_graph, legged_start, legged_goal
        ),
        "hopper": _build_optimum_artifact(
            "hopper", hopper_graph, hopper_start, hopper_goal
        ),
    }


def verify_optimum_record(artifact: dict[str, Any]) -> bool:
    row = artifact["row"]
    certificate = artifact["certificate"]
    graph = certificate["graph"]
    _validate_graph(graph)
    if sha256_bytes(canonical_json_bytes(certificate)) != row["certificate_sha256"]:
        return False
    if sha256_bytes(canonical_json_bytes(graph["candidate_edges"])) != row[
        "edge_list_sha256"
    ]:
        return False
    primary = heap_dijkstra(graph, row["start_node"], row["goal_node"])
    secondary = quadratic_dijkstra(graph, row["start_node"], row["goal_node"])
    if primary["cost_milli"] != secondary["cost_milli"]:
        return False
    expected_decimal = format(
        Decimal(int(primary["cost_milli"])) / Decimal(1000), "f"
    )
    if expected_decimal != row["optimum_cost_decimal"]:
        return False
    distances = primary["distance_labels_milli"]
    for edge in graph["candidate_edges"]:
        if not bool(edge["accepted"]):
            continue
        source_distance = distances.get(edge["from_node"])
        target_distance = distances.get(edge["to_node"])
        if source_distance is None or target_distance is None:
            continue
        if int(target_distance) > int(source_distance) + int(edge["cost_milli"]):
            return False
    edge_by_id = {edge["edge_id"]: edge for edge in graph["candidate_edges"]}
    path_cost = sum(
        int(edge_by_id[edge_id]["cost_milli"])
        for edge_id in row["optimal_path_edge_ids"]
    )
    return path_cost == int(primary["cost_milli"]) and bool(
        row["completeness"]["complete"]
    )
