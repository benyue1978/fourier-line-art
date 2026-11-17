import pytest

from fourier_line_art.models import Edge, Graph
from fourier_line_art.solvers import (
    NoFeasibleRoute,
    ProblemTooLarge,
    solve_rpp_exact,
)


def test_rpp_finds_minimum_closed_walk_covering_two_required_components():
    graph = Graph(
        vertices=("a", "b", "c", "d"),
        edges=(
            Edge("ab", "a", "b", 1, required=True, kind="draw"),
            Edge("cd", "c", "d", 1, required=True, kind="draw"),
            Edge("bc", "b", "c", 1, required=False, kind="pen_up"),
            Edge("da", "d", "a", 10, required=False, kind="pen_up"),
        ),
    )

    route = solve_rpp_exact(graph)

    assert route.is_closed
    assert {"ab", "cd"} <= {step.edge_id for step in route.traversals}
    assert route.cost == pytest.approx(6)


def test_rpp_raises_when_optional_edges_cannot_connect_required_edges():
    graph = Graph(
        vertices=(0, 1, 2, 3),
        edges=(
            Edge("required-a", 0, 1, 1, required=True),
            Edge("required-b", 2, 3, 1, required=True),
        ),
    )

    with pytest.raises(NoFeasibleRoute):
        solve_rpp_exact(graph)


def test_rpp_exact_solver_has_an_explicit_required_edge_limit():
    graph = Graph(
        vertices=tuple(range(18)),
        edges=tuple(
            Edge(f"e{i}", i, i + 1, 1, required=True)
            for i in range(17)
        ),
    )

    with pytest.raises(ProblemTooLarge):
        solve_rpp_exact(graph, max_required_edges=12)


def test_rpp_no_required_edges_returns_empty_route():
    graph = Graph(vertices=(0, 1), edges=(Edge("optional", 0, 1, 1, required=False),))

    route = solve_rpp_exact(graph)

    assert route.traversals == ()
    assert route.cost == 0
    assert route.is_closed

