"""Pen-up candidate construction and geometric route parameterization."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Hashable

import numpy as np

from .models import Edge, Graph, Route


class RouteGeometryError(ValueError):
    """A route cannot be converted to a continuous geometric path."""


def _stable_key(value: Hashable):
    return (type(value).__qualname__, repr(value))


def add_pen_up_candidates(graph: Graph, *, lift_height: float = 1.0) -> Graph:
    """Connect required drawing components with optional straight pen-up edges.

    Every pair of vertices in different required-edge components receives a
    candidate edge. Its route cost is horizontal distance plus the cost of one
    lift and one landing (`2 * lift_height`). The edge is hidden from the final
    drawing and retains the XY segment needed by the animation layer.
    """
    try:
        lift_height = float(lift_height)
    except (TypeError, ValueError) as exc:
        raise RouteGeometryError("lift_height must be finite and non-negative") from exc
    if not math.isfinite(lift_height) or lift_height < 0:
        raise RouteGeometryError("lift_height must be finite and non-negative")

    required_edges = [edge for edge in graph.edges if edge.required]
    if not required_edges:
        return graph

    adjacency = {vertex: set() for vertex in graph.vertices}
    active_vertices = set()
    for edge in required_edges:
        adjacency[edge.u].add(edge.v)
        adjacency[edge.v].add(edge.u)
        active_vertices.update((edge.u, edge.v))

    components: list[set[Hashable]] = []
    component_by_vertex = {}
    unseen = set(active_vertices)
    while unseen:
        seed = min(unseen, key=_stable_key)
        component = {seed}
        stack = [seed]
        unseen.remove(seed)
        while stack:
            vertex = stack.pop()
            for neighbor in adjacency[vertex]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
        component_index = len(components)
        components.append(component)
        for vertex in component:
            component_by_vertex[vertex] = component_index

    if len(components) <= 1:
        return graph

    positions = {vertex: graph.vertex_position(vertex) for vertex in active_vertices}
    missing = [vertex for vertex, position in positions.items() if position is None]
    if missing:
        raise RouteGeometryError(
            "pen-up candidates need pixel geometry for every required-edge vertex"
        )

    existing_pairs = {
        frozenset((edge.u, edge.v)) for edge in graph.edges if not edge.required
    }
    next_id = 0
    existing_ids = {edge.id for edge in graph.edges}
    candidates = []
    ordered_vertices = sorted(active_vertices, key=_stable_key)
    for index, u in enumerate(ordered_vertices):
        for v in ordered_vertices[index + 1 :]:
            if component_by_vertex[u] == component_by_vertex[v]:
                continue
            pair = frozenset((u, v))
            if pair in existing_pairs:
                continue
            start = positions[u]
            end = positions[v]
            distance = math.hypot(end[0] - start[0], end[1] - start[1])
            while f"pen-up-{next_id:05d}" in existing_ids:
                next_id += 1
            edge_id = f"pen-up-{next_id:05d}"
            existing_ids.add(edge_id)
            next_id += 1
            candidates.append(
                Edge(
                    id=edge_id,
                    u=u,
                    v=v,
                    weight=distance + 2 * lift_height,
                    required=False,
                    kind="pen_up",
                    geometry=(start, end),
                )
            )
    return Graph(graph.vertices, graph.edges + tuple(candidates))


@dataclass(frozen=True, slots=True)
class RoutePolyline:
    """Closed piecewise-linear `(x, y, h)` route before uniform resampling."""

    points: np.ndarray
    draw_segments: np.ndarray

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=np.float64)
        draw_segments = np.asarray(self.draw_segments, dtype=bool)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
            raise RouteGeometryError("route polyline must contain at least two 3D points")
        if not np.allclose(points[0], points[-1]):
            raise RouteGeometryError("route polyline must be closed")
        if len(draw_segments) != len(points) - 1:
            raise RouteGeometryError("draw_segments must label every polyline segment")
        if not np.all(np.isfinite(points)):
            raise RouteGeometryError("route points must be finite")
        points = np.array(points, copy=True)
        draw_segments = np.array(draw_segments, copy=True)
        points.setflags(write=False)
        draw_segments.setflags(write=False)
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "draw_segments", draw_segments)

    @property
    def length(self) -> float:
        return float(np.linalg.norm(np.diff(self.points, axis=0), axis=1).sum())

    def resample(self, sample_count: int) -> "RouteSamples":
        if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 2:
            raise RouteGeometryError("sample_count must be an integer of at least 2")
        segment_lengths = np.linalg.norm(np.diff(self.points, axis=0), axis=1)
        total_length = float(segment_lengths.sum())
        if total_length == 0:
            samples = np.repeat(self.points[:1], sample_count, axis=0)
            visible = np.full(sample_count, bool(self.draw_segments.any()), dtype=bool)
            return RouteSamples(samples, visible)

        cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
        targets = np.arange(sample_count, dtype=np.float64) * (total_length / sample_count)
        segment_indices = np.searchsorted(cumulative, targets, side="right") - 1
        segment_indices = np.clip(segment_indices, 0, len(segment_lengths) - 1)
        local = (targets - cumulative[segment_indices]) / segment_lengths[segment_indices]
        samples = (
            self.points[segment_indices]
            + local[:, None] * (self.points[segment_indices + 1] - self.points[segment_indices])
        )
        visible = self.draw_segments[segment_indices]
        return RouteSamples(samples, visible)


@dataclass(frozen=True, slots=True)
class RouteSamples:
    """Uniform arc-length samples and source pen-down state."""

    points: np.ndarray
    draw: np.ndarray

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=np.float64)
        draw = np.asarray(self.draw, dtype=bool)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) != len(draw):
            raise RouteGeometryError("route samples require aligned (x, y, h) and draw arrays")
        if not np.all(np.isfinite(points)):
            raise RouteGeometryError("route samples must be finite")
        points = np.array(points, copy=True)
        draw = np.array(draw, copy=True)
        points.setflags(write=False)
        draw.setflags(write=False)
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "draw", draw)


def polyline_from_route(route: Route, *, lift_height: float = 1.0) -> RoutePolyline:
    """Expand oriented graph-edge geometry into one closed 3D path."""
    try:
        lift_height = float(lift_height)
    except (TypeError, ValueError) as exc:
        raise RouteGeometryError("lift_height must be finite and positive") from exc
    if not math.isfinite(lift_height) or lift_height <= 0:
        raise RouteGeometryError("lift_height must be finite and positive")
    if not route.traversals:
        raise RouteGeometryError("cannot build a polyline from an empty route")
    if not route.is_closed:
        raise RouteGeometryError("Fourier route must be a closed walk")

    points: list[tuple[float, float, float]] = []
    draw_segments: list[bool] = []

    def append_point(point, draw: bool):
        point = tuple(map(float, point))
        if not points:
            points.append(point)
        else:
            draw_segments.append(draw)
            points.append(point)

    for step in route.traversals:
        edge = route.graph.edge_by_id(step.edge_id)
        if not edge.geometry:
            raise RouteGeometryError(f"edge {edge.id!r} has no pixel geometry")
        geometry = edge.geometry if step.from_vertex == edge.u else tuple(reversed(edge.geometry))
        start_xy = geometry[0]
        end_xy = geometry[-1]
        if not points:
            points.append((start_xy[0], start_xy[1], lift_height))

        if edge.kind == "pen_up":
            if not np.allclose(points[-1], (start_xy[0], start_xy[1], lift_height)):
                append_point((start_xy[0], start_xy[1], lift_height), True)
            append_point((start_xy[0], start_xy[1], 0.0), False)
            for x, y in geometry[1:]:
                append_point((x, y, 0.0), False)
            append_point((end_xy[0], end_xy[1], lift_height), False)
            continue

        if points[-1][2] != lift_height:
            append_point((points[-1][0], points[-1][1], lift_height), False)
        if not np.allclose(points[-1], (start_xy[0], start_xy[1], lift_height)):
            append_point((start_xy[0], start_xy[1], lift_height), True)
        if len(geometry) == 1:
            append_point((start_xy[0], start_xy[1], lift_height), True)
        else:
            for x, y in geometry[1:]:
                append_point((x, y, lift_height), True)

    first = points[0]
    if not np.allclose(points[-1], first):
        append_point(first, points[-1][2] >= lift_height / 2)
    if len(points) < 2:
        raise RouteGeometryError("route geometry collapsed to fewer than two points")
    return RoutePolyline(np.asarray(points), np.asarray(draw_segments))

