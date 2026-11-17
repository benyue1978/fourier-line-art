import numpy as np
import pytest

from fourier_line_art.models import Edge, Graph, Route, Traversal
from fourier_line_art.routes import add_pen_up_candidates, polyline_from_route
from fourier_line_art.solvers import solve_rpp_exact


def two_strokes_graph():
    return Graph(
        vertices=("a", "b", "c", "d"),
        edges=(
            Edge("ab", "a", "b", 1, geometry=((0, 0), (1, 0))),
            Edge("cd", "c", "d", 1, geometry=((4, 0), (5, 0))),
        ),
    )


def test_pen_up_candidate_builder_connects_distinct_drawing_components():
    graph = add_pen_up_candidates(two_strokes_graph(), lift_height=0.5)
    optional = [edge for edge in graph.edges if not edge.required]

    assert len(optional) == 4
    assert all(edge.kind == "pen_up" for edge in optional)
    nearest = next(edge for edge in optional if {edge.u, edge.v} == {"b", "c"})
    assert nearest.weight == pytest.approx(4)
    assert nearest.geometry == ((1.0, 0.0), (4.0, 0.0))


def test_rpp_candidates_form_a_closed_3d_route_with_hidden_lift_segments():
    graph = add_pen_up_candidates(two_strokes_graph(), lift_height=0.5)
    route = solve_rpp_exact(graph)
    polyline = polyline_from_route(route, lift_height=0.5)
    samples = polyline.resample(128)

    assert route.is_closed
    assert route.cost == pytest.approx(12)
    assert np.allclose(polyline.points[0], polyline.points[-1])
    assert polyline.points[:, 2].min() == pytest.approx(0)
    assert polyline.points[:, 2].max() == pytest.approx(0.5)
    assert samples.draw.any()
    assert (~samples.draw).any()
    assert samples.points.shape == (128, 3)


def test_route_geometry_obeys_edge_orientation_and_closes():
    graph = Graph(
        vertices=("a", "b"),
        edges=(Edge("ab", "a", "b", 2, geometry=((0, 0), (2, 0))),),
    )
    route = Route(
        (Traversal("ab", "a", "b"), Traversal("ab", "b", "a")),
        graph,
    )

    polyline = polyline_from_route(route)

    assert polyline.points[0] == pytest.approx(polyline.points[-1])
    assert polyline.length == pytest.approx(4)
    assert polyline.draw_segments.all()


def test_pen_up_candidates_require_geometry_and_positive_lift_height():
    graph = Graph(
        vertices=(0, 1, 2, 3),
        edges=(Edge("a", 0, 1, 1), Edge("b", 2, 3, 1)),
    )

    with pytest.raises(ValueError, match="pixel geometry"):
        add_pen_up_candidates(graph)
    with pytest.raises(ValueError, match="lift_height"):
        polyline_from_route(Route((), Graph((), ())), lift_height=0)

