import pytest

from fourier_line_art.models import Edge, Graph
from fourier_line_art.solvers import NoFeasibleRoute, solve_cpp


def assert_route_covers_all_edges(graph, route):
    traversed = {step.edge_id for step in route.traversals}
    assert {edge.id for edge in graph.edges} <= traversed
    assert route.is_closed
    assert route.cost == pytest.approx(
        sum(graph.edge_by_id(step.edge_id).weight for step in route.traversals)
    )


def test_cpp_eulerian_triangle_uses_each_edge_once():
    graph = Graph(
        vertices=("a", "b", "c"),
        edges=(
            Edge("ab", "a", "b", 1),
            Edge("bc", "b", "c", 1),
            Edge("ca", "c", "a", 1),
        ),
    )

    route = solve_cpp(graph)

    assert_route_covers_all_edges(graph, route)
    assert len(route.traversals) == 3
    assert route.cost == pytest.approx(3)


def test_cpp_three_edge_path_duplicates_the_path_to_close():
    graph = Graph(
        vertices=(0, 1, 2, 3),
        edges=(Edge("a", 0, 1, 1), Edge("b", 1, 2, 1), Edge("c", 2, 3, 1)),
    )

    route = solve_cpp(graph)

    assert_route_covers_all_edges(graph, route)
    assert len(route.traversals) == 6
    assert route.cost == pytest.approx(6)


def test_cpp_uses_minimum_matching_for_odd_vertices():
    graph = Graph(
        vertices=("a", "b", "c", "d"),
        edges=(
            Edge("ab", "a", "b", 1),
            Edge("bc", "b", "c", 2),
            Edge("cd", "c", "d", 1),
            Edge("da", "d", "a", 5),
            Edge("ac", "a", "c", 1),
        ),
    )

    route = solve_cpp(graph)

    assert_route_covers_all_edges(graph, route)
    assert route.cost == pytest.approx(11)


def test_cpp_rejects_disconnected_required_edges():
    graph = Graph(
        vertices=(0, 1, 2, 3),
        edges=(Edge("left", 0, 1, 1), Edge("right", 2, 3, 1)),
    )

    with pytest.raises(NoFeasibleRoute):
        solve_cpp(graph)


def test_cpp_is_deterministic_for_same_input():
    graph = Graph(
        vertices=("a", "b", "c"),
        edges=(Edge("ab", "a", "b", 1), Edge("bc", "b", "c", 1)),
    )

    first = solve_cpp(graph)
    second = solve_cpp(graph)

    assert first.traversals == second.traversals


def test_cpp_preserves_parallel_edge_ids_in_the_euler_tour():
    graph = Graph(
        vertices=("a", "b"),
        edges=(Edge("ab-1", "a", "b", 1), Edge("ab-2", "a", "b", 2)),
    )

    route = solve_cpp(graph)

    assert_route_covers_all_edges(graph, route)
    assert route.cost == pytest.approx(3)

