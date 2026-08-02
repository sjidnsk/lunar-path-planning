from __future__ import annotations

import math
from typing import Any, Iterable, Sequence


Point = tuple[int, int]
Polygon = list[list[int]]


def _point_segment_distance_squared(
    point_x: float,
    point_y: float,
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
) -> float:
    delta_x = end_x - start_x
    delta_y = end_y - start_y
    denominator = delta_x * delta_x + delta_y * delta_y
    if denominator == 0.0:
        return (point_x - start_x) ** 2 + (point_y - start_y) ** 2
    fraction = max(
        0.0,
        min(
            1.0,
            (
                (point_x - start_x) * delta_x
                + (point_y - start_y) * delta_y
            )
            / denominator,
        ),
    )
    closest_x = start_x + fraction * delta_x
    closest_y = start_y + fraction * delta_y
    return (point_x - closest_x) ** 2 + (point_y - closest_y) ** 2


def _orientation(
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    point_x: float,
    point_y: float,
) -> float:
    return (
        (end_x - start_x) * (point_y - start_y)
        - (end_y - start_y) * (point_x - start_x)
    )


def _on_segment_closed(
    point_x: float,
    point_y: float,
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
) -> bool:
    return (
        min(start_x, end_x) <= point_x <= max(start_x, end_x)
        and min(start_y, end_y) <= point_y <= max(start_y, end_y)
    )


def _segments_intersect_closed(
    first_start: tuple[float, float],
    first_end: tuple[float, float],
    second_start: tuple[float, float],
    second_end: tuple[float, float],
) -> bool:
    ax, ay = first_start
    bx, by = first_end
    cx, cy = second_start
    dx, dy = second_end
    first_c = _orientation(ax, ay, bx, by, cx, cy)
    first_d = _orientation(ax, ay, bx, by, dx, dy)
    second_a = _orientation(cx, cy, dx, dy, ax, ay)
    second_b = _orientation(cx, cy, dx, dy, bx, by)
    if (
        (first_c > 0.0 and first_d < 0.0)
        or (first_c < 0.0 and first_d > 0.0)
    ) and (
        (second_a > 0.0 and second_b < 0.0)
        or (second_a < 0.0 and second_b > 0.0)
    ):
        return True
    tolerance = 1e-12
    return (
        abs(first_c) <= tolerance
        and _on_segment_closed(cx, cy, ax, ay, bx, by)
    ) or (
        abs(first_d) <= tolerance
        and _on_segment_closed(dx, dy, ax, ay, bx, by)
    ) or (
        abs(second_a) <= tolerance
        and _on_segment_closed(ax, ay, cx, cy, dx, dy)
    ) or (
        abs(second_b) <= tolerance
        and _on_segment_closed(bx, by, cx, cy, dx, dy)
    )


def _point_aabb_distance_squared(
    point_x: float,
    point_y: float,
    minimum_x: float,
    minimum_y: float,
    maximum_x: float,
    maximum_y: float,
) -> float:
    delta_x = max(minimum_x - point_x, 0.0, point_x - maximum_x)
    delta_y = max(minimum_y - point_y, 0.0, point_y - maximum_y)
    return delta_x * delta_x + delta_y * delta_y


def _segment_aabb_distance_squared(
    *,
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    minimum_x: float,
    minimum_y: float,
    maximum_x: float,
    maximum_y: float,
) -> float:
    if (
        minimum_x <= start_x <= maximum_x
        and minimum_y <= start_y <= maximum_y
    ) or (
        minimum_x <= end_x <= maximum_x
        and minimum_y <= end_y <= maximum_y
    ):
        return 0.0
    corners = (
        (minimum_x, minimum_y),
        (maximum_x, minimum_y),
        (maximum_x, maximum_y),
        (minimum_x, maximum_y),
    )
    segment = ((start_x, start_y), (end_x, end_y))
    if any(
        _segments_intersect_closed(
            segment[0],
            segment[1],
            corners[index - 1],
            corners[index],
        )
        for index in range(len(corners))
    ):
        return 0.0
    return min(
        _point_aabb_distance_squared(
            start_x,
            start_y,
            minimum_x,
            minimum_y,
            maximum_x,
            maximum_y,
        ),
        _point_aabb_distance_squared(
            end_x,
            end_y,
            minimum_x,
            minimum_y,
            maximum_x,
            maximum_y,
        ),
        *(
            _point_segment_distance_squared(
                corner_x,
                corner_y,
                start_x,
                start_y,
                end_x,
                end_y,
            )
            for corner_x, corner_y in corners
        ),
    )


def _closed_interval_candidate_indices(
    *,
    minimum_coordinate: float,
    maximum_coordinate: float,
    resolution_mm: int,
    cell_count: int,
) -> range:
    """Return cells whose closed axis interval can meet the query interval."""
    first = max(
        0,
        math.ceil(float(minimum_coordinate) / resolution_mm) - 1,
    )
    last = min(
        cell_count - 1,
        math.floor(float(maximum_coordinate) / resolution_mm),
    )
    if first > last:
        return range(0)
    return range(first, last + 1)


def fine_cells_intersecting_disk(
    *,
    center_x_mm: float,
    center_y_mm: float,
    radius_mm: int,
    resolution_mm: int,
    shape_height_width: tuple[int, int],
) -> list[list[int]]:
    if radius_mm < 0 or resolution_mm <= 0:
        raise ValueError("fine-cell disk dimensions must be nonnegative")
    height, width = shape_height_width
    cells: list[list[int]] = []
    radius_squared = float(radius_mm) * float(radius_mm)
    candidate_rows = _closed_interval_candidate_indices(
        minimum_coordinate=float(center_y_mm) - radius_mm,
        maximum_coordinate=float(center_y_mm) + radius_mm,
        resolution_mm=resolution_mm,
        cell_count=height,
    )
    candidate_columns = _closed_interval_candidate_indices(
        minimum_coordinate=float(center_x_mm) - radius_mm,
        maximum_coordinate=float(center_x_mm) + radius_mm,
        resolution_mm=resolution_mm,
        cell_count=width,
    )
    for row in candidate_rows:
        minimum_y = row * resolution_mm
        maximum_y = minimum_y + resolution_mm
        closest_y = min(max(float(center_y_mm), minimum_y), maximum_y)
        for column in candidate_columns:
            minimum_x = column * resolution_mm
            maximum_x = minimum_x + resolution_mm
            closest_x = min(max(float(center_x_mm), minimum_x), maximum_x)
            if (
                (closest_x - float(center_x_mm)) ** 2
                + (closest_y - float(center_y_mm)) ** 2
                <= radius_squared
            ):
                cells.append([column, row])
    return cells


def fine_cells_intersecting_capsule(
    *,
    start_x_mm: float,
    start_y_mm: float,
    end_x_mm: float,
    end_y_mm: float,
    radius_mm: int,
    resolution_mm: int,
    shape_height_width: tuple[int, int],
) -> list[list[int]]:
    if radius_mm < 0 or resolution_mm <= 0:
        raise ValueError("fine-cell capsule dimensions must be nonnegative")
    height, width = shape_height_width
    cells: list[list[int]] = []
    radius_squared = float(radius_mm) ** 2
    candidate_rows = _closed_interval_candidate_indices(
        minimum_coordinate=min(float(start_y_mm), float(end_y_mm)) - radius_mm,
        maximum_coordinate=max(float(start_y_mm), float(end_y_mm)) + radius_mm,
        resolution_mm=resolution_mm,
        cell_count=height,
    )
    candidate_columns = _closed_interval_candidate_indices(
        minimum_coordinate=min(float(start_x_mm), float(end_x_mm)) - radius_mm,
        maximum_coordinate=max(float(start_x_mm), float(end_x_mm)) + radius_mm,
        resolution_mm=resolution_mm,
        cell_count=width,
    )
    for row in candidate_rows:
        minimum_y = float(row * resolution_mm)
        maximum_y = minimum_y + resolution_mm
        for column in candidate_columns:
            minimum_x = float(column * resolution_mm)
            maximum_x = minimum_x + resolution_mm
            if (
                _segment_aabb_distance_squared(
                    start_x=float(start_x_mm),
                    start_y=float(start_y_mm),
                    end_x=float(end_x_mm),
                    end_y=float(end_y_mm),
                    minimum_x=minimum_x,
                    minimum_y=minimum_y,
                    maximum_x=maximum_x,
                    maximum_y=maximum_y,
                )
                <= radius_squared
            ):
                cells.append([column, row])
    return cells


def fine_slope_cdeg(
    elevation_um: Sequence[Sequence[int]],
    *,
    resolution_mm: int,
) -> list[list[int]]:
    if resolution_mm <= 0 or not elevation_um or not elevation_um[0]:
        raise ValueError("fine slope requires a non-empty positive-resolution grid")
    height = len(elevation_um)
    width = len(elevation_um[0])
    if any(len(row) != width for row in elevation_um):
        raise ValueError("fine elevation rows must have one width")

    def derivative(row: int, column: int, *, axis: int) -> float:
        if axis == 0:
            left = max(0, column - 1)
            right = min(width - 1, column + 1)
            delta_um = int(elevation_um[row][right]) - int(
                elevation_um[row][left]
            )
            run_mm = max(1, right - left) * resolution_mm
        else:
            lower = max(0, row - 1)
            upper = min(height - 1, row + 1)
            delta_um = int(elevation_um[upper][column]) - int(
                elevation_um[lower][column]
            )
            run_mm = max(1, upper - lower) * resolution_mm
        return delta_um / (run_mm * 1000.0)

    return [
        [
            round(
                math.degrees(
                    math.atan(
                        math.hypot(
                            derivative(row, column, axis=0),
                            derivative(row, column, axis=1),
                        )
                    )
                )
                * 100.0
            )
            for column in range(width)
        ]
        for row in range(height)
    ]


def relative_relief_um(
    elevation_um: Sequence[Sequence[int]],
) -> list[list[int]]:
    if not elevation_um or not elevation_um[0]:
        raise ValueError("relative relief requires a non-empty grid")
    reference = int(elevation_um[0][0])
    return [
        [int(value) - reference for value in row]
        for row in elevation_um
    ]


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
