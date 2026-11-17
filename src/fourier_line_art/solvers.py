"""Chinese and Rural Postman solvers for undirected weighted multigraphs."""

from __future__ import annotations

from collections import defaultdict
import heapq
import itertools
import math
from typing import Hashable, Iterable

import networkx as nx

from .models import Edge, Graph, Route, Traversal


class NoFeasibleRoute(ValueError):
    """The required edges cannot be covered by one closed walk."""


class ProblemTooLarge(ValueError):
    """The exact solver's documented state-space limit was exceeded."""


def _sort_key(value: Hashable) -> tuple[str, str]:
    return (type(value).__qualname__, repr(value))


def _adjacency(graph: Graph, edges: Iterable[Edge] | None = None):
    adjacency: dict[Hashable, list[tuple[Hashable, Edge]]] = {
        vertex: [] for vertex in graph.vertices
    }
    for edge in graph.edges if edges is None else edges:
        adjacency[edge.u].append((edge.v, edge))
        adjacency[edge.v].append((edge.u, edge))
    for neighbors in adjacency.values():
        neighbors.sort(key=lambda item: (item[1].id, _sort_key(item[0])))
    return adjacency


def _connected_component(adjacency, start: Hashable) -> set[Hashable]:
    seen = {start}
    stack = [start]
    while stack:
        vertex = stack.pop()
        for neighbor, _ in adjacency[vertex]:
            if neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return seen


def _dijkstra(adjacency, source: Hashable):
    distances = {source: 0.0}
    previous: dict[Hashable, tuple[Hashable, str]] = {}
    sequence = itertools.count()
    queue = [(0.0, next(sequence), source)]
    while queue:
        distance, _, vertex = heapq.heappop(queue)
        if distance != distances.get(vertex):
            continue
        for neighbor, edge in adjacency[vertex]:
            candidate = distance + edge.weight
            if candidate < distances.get(neighbor, math.inf):
                distances[neighbor] = candidate
                previous[neighbor] = (vertex, edge.id)
                heapq.heappush(queue, (candidate, next(sequence), neighbor))
    return distances, previous


def _multi_source_dijkstra(adjacency, sources: Iterable[Hashable]):
    sources = sorted(set(sources), key=_sort_key)
    distances = {source: 0.0 for source in sources}
    previous: dict[Hashable, tuple[Hashable, str]] = {}
    sequence = itertools.count()
    queue = [(0.0, next(sequence), source) for source in sources]
    heapq.heapify(queue)
    while queue:
        distance, _, vertex = heapq.heappop(queue)
        if distance != distances.get(vertex):
            continue
        for neighbor, edge in adjacency[vertex]:
            candidate = distance + edge.weight
            if candidate < distances.get(neighbor, math.inf):
                distances[neighbor] = candidate
                previous[neighbor] = (vertex, edge.id)
                heapq.heappush(queue, (candidate, next(sequence), neighbor))
    return distances, previous


def _shortest_path(previous, source: Hashable, target: Hashable) -> list[tuple[str, Hashable, Hashable]]:
    if source == target:
        return []
    reversed_path = []
    current = target
    while current != source:
        parent, edge_id = previous[current]
        reversed_path.append((edge_id, parent, current))
        current = parent
    return list(reversed(reversed_path))


def _euler_tour(graph: Graph, edge_instances, start: Hashable) -> Route:
    """Hierholzer's algorithm over edge instances, retaining parallel edges."""
    adjacency: dict[Hashable, list[tuple[int, Hashable, str]]] = defaultdict(list)
    for instance_id, (edge_id, u, v) in enumerate(edge_instances):
        adjacency[u].append((instance_id, v, edge_id))
        adjacency[v].append((instance_id, u, edge_id))
    for incident in adjacency.values():
        incident.sort(key=lambda item: (item[2], _sort_key(item[1]), item[0]))

    cursors: dict[Hashable, int] = defaultdict(int)
    used: set[int] = set()
    vertex_stack: list[tuple[Hashable, Traversal | None]] = [(start, None)]
    reversed_steps: list[Traversal] = []

    while vertex_stack:
        vertex, entering_step = vertex_stack[-1]
        incident = adjacency[vertex]
        cursor = cursors[vertex]
        while cursor < len(incident) and incident[cursor][0] in used:
            cursor += 1
        cursors[vertex] = cursor
        if cursor < len(incident):
            instance_id, neighbor, edge_id = incident[cursor]
            used.add(instance_id)
            cursors[vertex] += 1
            vertex_stack.append(
                (neighbor, Traversal(edge_id, vertex, neighbor))
            )
        else:
            vertex_stack.pop()
            if entering_step is not None:
                reversed_steps.append(entering_step)

    if len(used) != len(edge_instances):
        raise NoFeasibleRoute("Eulerization left edges outside the selected component")
    return Route(tuple(reversed(reversed_steps)), graph)


def solve_cpp(graph: Graph) -> Route:
    """Solve the undirected Chinese Postman Problem exactly.

    The implementation uses shortest paths between odd-degree vertices,
    minimum-weight perfect matching (NetworkX's blossom implementation), then
    Hierholzer's Euler-tour algorithm on the augmented multigraph.
    """
    if not graph.edges:
        return Route((), graph)
    if any(not edge.required for edge in graph.edges):
        raise ValueError("CPP expects every graph edge to be required")

    adjacency = _adjacency(graph)
    nonzero_vertices = {vertex for edge in graph.edges for vertex in (edge.u, edge.v)}
    start = min(nonzero_vertices, key=_sort_key)
    component = _connected_component(adjacency, start)
    if any(vertex not in component for vertex in nonzero_vertices):
        raise NoFeasibleRoute("required graph is disconnected")

    odd_vertices = sorted(
        (vertex for vertex in nonzero_vertices if graph.degree(vertex) % 2),
        key=_sort_key,
    )
    shortest: dict[Hashable, tuple[dict, dict]] = {
        vertex: _dijkstra(adjacency, vertex) for vertex in odd_vertices
    }

    matching = []
    if odd_vertices:
        complete = nx.Graph()
        complete.add_nodes_from(odd_vertices)
        for index, u in enumerate(odd_vertices):
            distances, _ = shortest[u]
            for v in odd_vertices[index + 1 :]:
                complete.add_edge(u, v, weight=distances[v])
        matching = list(nx.algorithms.matching.min_weight_matching(complete, weight="weight"))
        if len(matching) * 2 != len(odd_vertices):
            raise NoFeasibleRoute("failed to find a perfect matching for odd vertices")

    edge_instances: list[tuple[str, Hashable, Hashable]] = [
        (edge.id, edge.u, edge.v) for edge in graph.edges
    ]
    for u, v in sorted(
        ((min((u, v), key=_sort_key), max((u, v), key=_sort_key)) for u, v in matching),
        key=lambda pair: (_sort_key(pair[0]), _sort_key(pair[1])),
    ):
        _, previous = shortest[u]
        edge_instances.extend(_shortest_path(previous, u, v))

    route = _euler_tour(graph, edge_instances, start)
    if not route.is_closed:
        raise NoFeasibleRoute("Euler tour is unexpectedly open")
    return route


def solve_rpp_exact(
    graph: Graph,
    *,
    max_required_edges: int = 15,
    max_states: int = 2_000_000,
) -> Route:
    """Find an exact closed walk covering required edges in an undirected graph.

    This runs Dijkstra on a product state graph `(vertex, required-edge mask)`.
    Every input edge may be traversed in either direction; traversing a required
    edge sets its bit. The cycle start is restricted to required-edge endpoints,
    which loses no optimum because any closed tour can be rotated to one of
    those endpoints. The method is exact but exponential in required-edge
    count, so callers must keep compressed RPP instances small.
    """
    if isinstance(max_required_edges, bool) or not isinstance(max_required_edges, int) or max_required_edges < 0:
        raise ValueError("max_required_edges must be a non-negative integer")
    if isinstance(max_states, bool) or not isinstance(max_states, int) or max_states < 1:
        raise ValueError("max_states must be a positive integer")
    required = tuple(sorted((edge for edge in graph.edges if edge.required), key=lambda e: e.id))
    if not required:
        return Route((), graph)
    if len(required) > max_required_edges:
        raise ProblemTooLarge(
            f"exact RPP supports at most {max_required_edges} required edges; got {len(required)}"
        )

    adjacency = _adjacency(graph)
    endpoints = {vertex for edge in required for vertex in (edge.u, edge.v)}
    seed = min(endpoints, key=_sort_key)
    reachable = _connected_component(adjacency, seed)
    if not endpoints <= reachable:
        raise NoFeasibleRoute("optional edges do not connect all required edges")

    vertex_count = len(reachable)
    mask_count = 1 << len(required)
    state_bound = vertex_count * mask_count
    starts = sorted(endpoints, key=_sort_key)
    work_bound = state_bound * len(starts)
    if work_bound > max_states:
        raise ProblemTooLarge(
            "exact RPP state-space limit exceeded "
            f"({vertex_count} vertices, {len(required)} required edges, "
            f"up to {work_bound:,} states; limit {max_states:,})"
        )

    required_bit = {edge.id: 1 << index for index, edge in enumerate(required)}
    full_mask = mask_count - 1
    best_cost = math.inf
    best_traversals: tuple[Traversal, ...] | None = None

    for start in starts:
        initial = (start, 0)
        distances = {initial: 0.0}
        previous: dict[tuple[Hashable, int], tuple[tuple[Hashable, int], Traversal]] = {}
        sequence = itertools.count()
        queue = [(0.0, next(sequence), initial)]
        goal = (start, full_mask)

        while queue:
            distance, _, state = heapq.heappop(queue)
            if distance != distances.get(state):
                continue
            if distance >= best_cost:
                continue
            if state == goal:
                best_cost = distance
                reversed_steps = []
                current = goal
                while current != initial:
                    parent, step = previous[current]
                    reversed_steps.append(step)
                    current = parent
                best_traversals = tuple(reversed(reversed_steps))
                break

            vertex, mask = state
            for neighbor, edge in adjacency[vertex]:
                next_mask = mask | required_bit.get(edge.id, 0)
                next_state = (neighbor, next_mask)
                candidate = distance + edge.weight
                if candidate < distances.get(next_state, math.inf):
                    distances[next_state] = candidate
                    previous[next_state] = (
                        state,
                        Traversal(edge.id, vertex, neighbor),
                    )
                    heapq.heappush(queue, (candidate, next(sequence), next_state))

    if best_traversals is None:
        raise NoFeasibleRoute("no closed route covers every required edge")
    return Route(best_traversals, graph)


def solve_rpp_frederickson(graph: Graph) -> Route:
    """Run a Frederickson-style MST-connect-then-CPP RPP heuristic.

    The required-edge components are connected using a minimum spanning tree
    over their shortest-path distances in the full graph. Those paths are
    added as required multiedge copies, and the resulting connected graph is
    solved as a CPP with the exact matching-based solver. This constructs a
    valid tour for larger instances, but it is a heuristic and does not claim
    exact optimality for the image-derived graph.
    """
    required_edges = tuple(edge for edge in graph.edges if edge.required)
    if not required_edges:
        return Route((), graph)

    required_adjacency = _adjacency(graph, required_edges)
    required_vertices = {vertex for edge in required_edges for vertex in (edge.u, edge.v)}
    components = []
    unseen = set(required_vertices)
    while unseen:
        seed = min(unseen, key=_sort_key)
        component = _connected_component(required_adjacency, seed) & required_vertices
        components.append(component)
        unseen -= component

    if len(components) == 1:
        component = components[0]
        component_edges = tuple(
            edge for edge in required_edges if edge.u in component and edge.v in component
        )
        subgraph = Graph(tuple(graph.vertices), component_edges)
        return solve_cpp(subgraph)

    adjacency = _adjacency(graph)
    component_graph = nx.Graph()
    component_graph.add_nodes_from(range(len(components)))
    paths: dict[tuple[int, int], list[tuple[str, Hashable, Hashable]]] = {}

    for first_index, first_component in enumerate(components):
        distances, previous = _multi_source_dijkstra(adjacency, first_component)
        for second_index in range(first_index + 1, len(components)):
            second_component = components[second_index]
            reachable = [
                vertex for vertex in second_component if vertex in distances
            ]
            if not reachable:
                continue
            target = min(reachable, key=lambda vertex: (distances[vertex], _sort_key(vertex)))
            component_graph.add_edge(first_index, second_index, weight=distances[target])
            paths[(first_index, second_index)] = _shortest_path_to_any(
                previous, first_component, target
            )

    tree = nx.minimum_spanning_tree(component_graph, weight="weight", algorithm="kruskal")
    if tree.number_of_edges() != len(components) - 1:
        raise NoFeasibleRoute("optional edges do not connect all required components")

    augmented_edges = list(required_edges)
    for link_index, (first_index, second_index) in enumerate(
        sorted(tree.edges(), key=lambda pair: (pair[0], pair[1]))
    ):
        path = paths[(min(first_index, second_index), max(first_index, second_index))]
        for path_index, (edge_id, u, v) in enumerate(path):
            original = graph.edge_by_id(edge_id)
            augmented_edges.append(
                Edge(
                    id=f"frederickson-{link_index:04d}-{path_index:05d}-{edge_id}",
                    u=u,
                    v=v,
                    weight=original.weight,
                    required=True,
                    kind=original.kind,
                    geometry=(
                        original.geometry
                        if u == original.u
                        else tuple(reversed(original.geometry))
                    ),
                )
            )

    augmented_graph = Graph(graph.vertices, tuple(augmented_edges))
    return solve_cpp(augmented_graph)


def _shortest_path_to_any(previous, sources: Iterable[Hashable], target: Hashable):
    source_set = set(sources)
    if target in source_set:
        return []
    reversed_path = []
    current = target
    while current not in source_set:
        parent, edge_id = previous[current]
        reversed_path.append((edge_id, parent, current))
        current = parent
    return list(reversed(reversed_path))

