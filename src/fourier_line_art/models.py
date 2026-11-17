"""Small, validated data structures shared by the solvers and renderers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Hashable


class GraphValidationError(ValueError):
    """Raised when a graph or route violates its structural invariants."""


@dataclass(frozen=True, slots=True)
class Edge:
    """An undirected weighted multigraph edge.

    Edge IDs, rather than endpoint pairs, identify edges so parallel edges and
    repeated traversals remain unambiguous.
    """

    id: str
    u: Hashable
    v: Hashable
    weight: float
    required: bool = True
    kind: str = "draw"
    geometry: tuple[tuple[float, float], ...] = ()


@dataclass(frozen=True, slots=True)
class Graph:
    """An immutable undirected multigraph with deterministic edge ordering."""

    vertices: tuple[Hashable, ...]
    edges: tuple[Edge, ...]

    def __post_init__(self) -> None:
        vertices = tuple(self.vertices)
        edges = tuple(self.edges)
        try:
            vertex_set = set(vertices)
        except TypeError as exc:
            raise GraphValidationError("vertices must be hashable") from exc

        if len(vertex_set) != len(vertices):
            raise GraphValidationError("vertex IDs must be unique")

        edge_ids: set[str] = set()
        for edge in edges:
            if not isinstance(edge, Edge):
                raise GraphValidationError("edges must contain Edge instances")
            if not isinstance(edge.id, str) or not edge.id:
                raise GraphValidationError("edge IDs must be non-empty strings")
            if edge.id in edge_ids:
                raise GraphValidationError(f"duplicate edge ID: {edge.id}")
            edge_ids.add(edge.id)
            if edge.u not in vertex_set or edge.v not in vertex_set:
                raise GraphValidationError(f"edge {edge.id!r} has a missing endpoint")
            try:
                weight = float(edge.weight)
            except (TypeError, ValueError) as exc:
                raise GraphValidationError(f"edge {edge.id!r} weight must be numeric") from exc
            if not math.isfinite(weight) or weight < 0:
                raise GraphValidationError(
                    f"edge {edge.id!r} weight must be finite and non-negative"
                )
            object.__setattr__(edge, "weight", weight)
            if not isinstance(edge.required, bool):
                raise GraphValidationError(f"edge {edge.id!r} required must be bool")
            if not isinstance(edge.kind, str) or not edge.kind:
                raise GraphValidationError(f"edge {edge.id!r} kind must be non-empty")
            geometry = tuple(tuple(point) for point in edge.geometry)
            if any(len(point) != 2 for point in geometry):
                raise GraphValidationError(f"edge {edge.id!r} geometry points must be 2D")
            try:
                finite_geometry = all(math.isfinite(float(value)) for point in geometry for value in point)
            except (TypeError, ValueError) as exc:
                raise GraphValidationError(f"edge {edge.id!r} geometry must be numeric") from exc
            if not finite_geometry:
                raise GraphValidationError(f"edge {edge.id!r} geometry must be finite")
            object.__setattr__(edge, "geometry", geometry)

        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "edges", edges)

    @property
    def edge_map(self) -> dict[str, Edge]:
        return {edge.id: edge for edge in self.edges}

    def edge_by_id(self, edge_id: str) -> Edge:
        for edge in self.edges:
            if edge.id == edge_id:
                return edge
        raise KeyError(edge_id)

    def incident_edges(self, vertex: Hashable) -> tuple[Edge, ...]:
        if vertex not in set(self.vertices):
            raise KeyError(vertex)
        return tuple(edge for edge in self.edges if edge.u == vertex or edge.v == vertex)

    def degree(self, vertex: Hashable) -> int:
        """Return undirected degree; a self-loop contributes two incidences."""
        if vertex not in set(self.vertices):
            raise KeyError(vertex)
        return sum(2 if edge.u == edge.v == vertex else 1 for edge in self.edges if vertex in (edge.u, edge.v))

    def vertex_position(self, vertex: Hashable) -> tuple[float, float] | None:
        """Estimate a drawing vertex's pixel position from incident geometry."""
        if vertex not in set(self.vertices):
            raise KeyError(vertex)
        points = []
        for edge in self.edges:
            if not edge.geometry:
                continue
            if edge.u == vertex:
                points.append(edge.geometry[0])
            if edge.v == vertex:
                points.append(edge.geometry[-1])
        if not points:
            return None
        return (
            sum(point[0] for point in points) / len(points),
            sum(point[1] for point in points) / len(points),
        )


@dataclass(frozen=True, slots=True)
class Traversal:
    """One oriented traversal of an undirected graph edge."""

    edge_id: str
    from_vertex: Hashable
    to_vertex: Hashable


@dataclass(frozen=True, slots=True)
class Route:
    """A continuous ordered walk over graph edges."""

    traversals: tuple[Traversal, ...]
    graph: Graph

    def __post_init__(self) -> None:
        traversals = tuple(self.traversals)
        edge_map = self.graph.edge_map
        previous_end: Hashable | None = None
        for index, step in enumerate(traversals):
            if step.edge_id not in edge_map:
                raise GraphValidationError(f"route refers to unknown edge {step.edge_id!r}")
            edge = edge_map[step.edge_id]
            if not (
                (step.from_vertex == edge.u and step.to_vertex == edge.v)
                or (step.from_vertex == edge.v and step.to_vertex == edge.u)
            ):
                raise GraphValidationError(f"route traversal does not match edge {edge.id!r}")
            if index and step.from_vertex != previous_end:
                raise GraphValidationError("route traversals are not contiguous")
            previous_end = step.to_vertex
        object.__setattr__(self, "traversals", traversals)

    @property
    def cost(self) -> float:
        edge_map = self.graph.edge_map
        return sum(edge_map[step.edge_id].weight for step in self.traversals)

    @property
    def is_closed(self) -> bool:
        return not self.traversals or self.traversals[0].from_vertex == self.traversals[-1].to_vertex

    @property
    def start_vertex(self) -> Hashable | None:
        return self.traversals[0].from_vertex if self.traversals else None

    @property
    def end_vertex(self) -> Hashable | None:
        return self.traversals[-1].to_vertex if self.traversals else None

