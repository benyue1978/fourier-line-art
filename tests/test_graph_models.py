import math

import pytest

from fourier_line_art.models import Edge, Graph, GraphValidationError


def test_graph_preserves_parallel_edges_and_counts_undirected_degree():
    graph = Graph(
        vertices=("a", "b"),
        edges=(
            Edge("ab-1", "a", "b", 1.0),
            Edge("ab-2", "a", "b", 2.0),
            Edge("loop", "a", "a", 3.0),
        ),
    )

    assert {edge.id for edge in graph.incident_edges("a")} == {"ab-1", "ab-2", "loop"}
    assert graph.degree("a") == 4
    assert graph.degree("b") == 2


@pytest.mark.parametrize(
    "vertices, edges",
    [
        (("a", "b"), (Edge("x", "a", "b", 1), Edge("x", "a", "b", 2))),
        (("a",), (Edge("x", "a", "missing", 1),)),
        (("a", "b"), (Edge("x", "a", "b", -1),)),
        (("a", "b"), (Edge("x", "a", "b", math.inf),)),
    ],
)
def test_graph_rejects_invalid_edges(vertices, edges):
    with pytest.raises(GraphValidationError):
        Graph(vertices=vertices, edges=edges)


def test_route_cost_and_closed_property_are_derived_from_traversals():
    from fourier_line_art.models import Route, Traversal

    graph = Graph(
        vertices=(0, 1),
        edges=(Edge("e", 0, 1, 2.5),),
    )
    route = Route(
        traversals=(Traversal("e", 0, 1), Traversal("e", 1, 0)),
        graph=graph,
    )

    assert route.cost == pytest.approx(5.0)
    assert route.is_closed

