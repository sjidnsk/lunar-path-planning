from __future__ import annotations

import math
from typing import Any, Iterable, Sequence


Point = tuple[int, int]
Polygon = list[list[int]]


def plane_height_mm(plane: dict[str, Any], x_mm: int, y_mm: int) -> int:
    return int(plane["origin_z_mm"]) + round(
        (
            (int(x_mm) - int(plane["origin_x_mm"]))
            * int(plane["gradient_x_ppm"])
            + (int(y_mm) - int(plane["origin_y_mm"]))
            * int(plane["gradient_y_ppm"])
        )
        / 1_000_000
    )


def plane_slope_cdeg(plane: dict[str, Any]) -> int:
    gradient = math.hypot(
        int(plane["gradient_x_ppm"]), int(plane["gradient_y_ppm"])
    ) / 1_000_000.0
    return round(math.degrees(math.atan(gradient)) * 100.0)


def gradient_ppm_for_slope_cdeg(slope_cdeg: int) -> int:
    return round(math.tan(math.radians(int(slope_cdeg) / 100.0)) * 1_000_000)


def point_in_polygon_closed(point: Point, polygon: Sequence[Sequence[int]]) -> bool:
    x, y = point
    inside = False
    if len(polygon) < 3:
        return False
    previous_x, previous_y = int(polygon[-1][0]), int(polygon[-1][1])
    for raw_current in polygon:
        current_x, current_y = int(raw_current[0]), int(raw_current[1])
        cross = (current_x - previous_x) * (y - previous_y) - (
            current_y - previous_y
        ) * (x - previous_x)
        if (
            cross == 0
            and min(previous_x, current_x) <= x <= max(previous_x, current_x)
            and min(previous_y, current_y) <= y <= max(previous_y, current_y)
        ):
            return True
        if (current_y > y) != (previous_y > y):
            intersection_x = previous_x + (
                (y - previous_y) * (current_x - previous_x)
                / (current_y - previous_y)
            )
            if x < intersection_x:
                inside = not inside
        previous_x, previous_y = current_x, current_y
    return inside


def point_to_segment_distance_mm(
    point: Point, start: Point, end: Point
) -> float:
    px, py = point
    ax, ay = start
    bx, by = end
    dx = bx - ax
    dy = by - ay
    denominator = dx * dx + dy * dy
    if denominator == 0:
        return math.hypot(px - ax, py - ay)
    projection = ((px - ax) * dx + (py - ay) * dy) / denominator
    projection = min(1.0, max(0.0, projection))
    closest_x = ax + projection * dx
    closest_y = ay + projection * dy
    return math.hypot(px - closest_x, py - closest_y)


def point_to_polygon_distance_mm(
    point: Point, polygon: Sequence[Sequence[int]]
) -> float:
    if point_in_polygon_closed(point, polygon):
        return 0.0
    vertices = [(int(vertex[0]), int(vertex[1])) for vertex in polygon]
    return min(
        point_to_segment_distance_mm(
            point, vertices[index - 1], vertices[index]
        )
        for index in range(len(vertices))
    )


def minimum_clearance_mm(
    points: Iterable[Point],
    polygons: Sequence[Sequence[Sequence[int]]],
    radius_mm: int,
) -> int:
    point_list = list(points)
    if not polygons:
        return 2**30
    distance = min(
        point_to_polygon_distance_mm(point, polygon)
        for point in point_list
        for polygon in polygons
    )
    return round(distance - int(radius_mm))


def map_clearance_mm(
    points: Iterable[Point], bounds_mm: Sequence[int], radius_mm: int
) -> int:
    minimum_x, minimum_y, maximum_x, maximum_y = [
        int(value) for value in bounds_mm
    ]
    return min(
        min(
            x - minimum_x,
            maximum_x - x,
            y - minimum_y,
            maximum_y - y,
        )
        - int(radius_mm)
        for x, y in points
    )


def sample_segment(
    start: Point, end: Point, *, sample_count: int = 65
) -> list[Point]:
    if sample_count < 2:
        raise ValueError("sample_count must be at least two")
    return [
        (
            round(start[0] + (end[0] - start[0]) * index / (sample_count - 1)),
            round(start[1] + (end[1] - start[1]) * index / (sample_count - 1)),
        )
        for index in range(sample_count)
    ]


def points_touch_polygons(
    points: Iterable[Point], polygons: Sequence[Sequence[Sequence[int]]]
) -> bool:
    return any(
        point_in_polygon_closed(point, polygon)
        for point in points
        for polygon in polygons
    )


def signed_polygon_margin_mm(
    point: Point, polygon: Sequence[Sequence[int]]
) -> int:
    vertices = [(int(vertex[0]), int(vertex[1])) for vertex in polygon]
    distance = round(
        min(
            point_to_segment_distance_mm(
                point, vertices[index - 1], vertices[index]
            )
            for index in range(len(vertices))
        )
    )
    return distance if point_in_polygon_closed(point, polygon) else -distance


def square_polygon(center_x: int, center_y: int, halfwidth_mm: int) -> Polygon:
    return [
        [center_x - halfwidth_mm, center_y - halfwidth_mm],
        [center_x + halfwidth_mm, center_y - halfwidth_mm],
        [center_x + halfwidth_mm, center_y + halfwidth_mm],
        [center_x - halfwidth_mm, center_y + halfwidth_mm],
    ]
