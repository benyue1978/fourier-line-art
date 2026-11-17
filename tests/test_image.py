import numpy as np
import pytest

from fourier_line_art.image import ImageInputError, analyze_image, skeleton_to_graph


def test_plus_skeleton_compresses_into_four_required_edges():
    skeleton = np.zeros((7, 7), dtype=bool)
    skeleton[3, 1:6] = True
    skeleton[1:6, 3] = True

    graph, labels, component_count = skeleton_to_graph(skeleton, connectivity=4)

    assert component_count == 1
    assert labels.max() == 1
    assert len(graph.vertices) == 5
    assert len(graph.edges) == 4
    assert all(edge.required and edge.kind == "draw" for edge in graph.edges)
    assert all(edge.geometry[0] != edge.geometry[-1] for edge in graph.edges)


def test_degree_two_closed_loop_becomes_one_self_loop_edge():
    skeleton = np.zeros((5, 7), dtype=bool)
    skeleton[1, 1:6] = True
    skeleton[3, 1:6] = True
    skeleton[1:4, 1] = True
    skeleton[1:4, 5] = True

    graph, _, component_count = skeleton_to_graph(skeleton, connectivity=4)

    assert component_count == 1
    assert len(graph.vertices) == 1
    assert len(graph.edges) == 1
    assert graph.edges[0].u == graph.edges[0].v
    assert graph.edges[0].weight == pytest.approx(12)


def test_disconnected_strokes_are_labeled_as_separate_components():
    skeleton = np.zeros((8, 8), dtype=bool)
    skeleton[1, 1:5] = True
    skeleton[6, 2:7] = True

    graph, labels, component_count = skeleton_to_graph(skeleton, connectivity=4)

    assert component_count == 2
    assert labels[1, 1] != labels[6, 2]
    assert len(graph.edges) == 2


def test_diagonal_contact_depends_on_selected_connectivity():
    skeleton = np.eye(3, dtype=bool)

    graph4, _, count4 = skeleton_to_graph(skeleton, connectivity=4)
    graph8, _, count8 = skeleton_to_graph(skeleton, connectivity=8)

    assert count4 == 3
    assert count8 == 1
    assert len(graph4.edges) == 3
    assert all(edge.u == edge.v for edge in graph4.edges)
    assert len(graph8.edges) == 1


def test_threshold_handles_rgb_and_transparent_background():
    image = np.full((5, 5, 4), 255, dtype=np.uint8)
    image[..., 3] = 0
    image[2, 1:4] = (0, 0, 0, 255)

    result = analyze_image(image, threshold=128, connectivity=4)

    assert result.component_count == 1
    assert result.foreground.sum() == 3
    assert result.skeleton.sum() == 3
    assert len(result.graph.edges) == 1


def test_empty_image_has_no_components_or_edges():
    result = analyze_image(np.full((4, 4), 255, dtype=np.uint8))

    assert result.component_count == 0
    assert result.graph.vertices == ()
    assert result.graph.edges == ()


def test_single_pixel_is_preserved_as_a_zero_length_required_loop():
    skeleton = np.zeros((3, 3), dtype=bool)
    skeleton[1, 1] = True

    graph, _, component_count = skeleton_to_graph(skeleton, connectivity=8)

    assert component_count == 1
    assert len(graph.edges) == 1
    assert graph.edges[0].u == graph.edges[0].v
    assert graph.edges[0].weight == 0
    assert graph.edges[0].geometry == ((1.0, 1.0),)


def test_image_input_and_connectivity_are_validated():
    with pytest.raises(ImageInputError):
        analyze_image(np.zeros((2, 2, 2), dtype=np.uint8))
    with pytest.raises(ImageInputError):
        analyze_image(np.zeros((2, 2), dtype=np.uint8), connectivity=6)
    with pytest.raises(ImageInputError):
        skeleton_to_graph(np.zeros((2, 2)), connectivity=6)

