"""Black-and-white image preparation and skeleton-to-multigraph conversion."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.measure import label as connected_component_labels
from skimage.morphology import skeletonize

from .models import Edge, Graph


class ImageInputError(ValueError):
    """An image or connectivity option is not supported."""


@dataclass(frozen=True, slots=True)
class ImageAnalysis:
    """Threshold, skeleton, component labels, and compressed drawing graph."""

    foreground: np.ndarray
    skeleton: np.ndarray
    component_labels: np.ndarray
    graph: Graph
    connectivity: int

    @property
    def component_count(self) -> int:
        return int(self.component_labels.max(initial=0))


def _as_grayscale(image) -> np.ndarray:
    if isinstance(image, (str, Path)):
        try:
            with Image.open(image) as opened:
                image = np.asarray(opened.convert("RGBA"))
        except (OSError, ValueError) as exc:
            raise ImageInputError(f"could not read image: {image}") from exc
    elif isinstance(image, Image.Image):
        image = np.asarray(image.convert("RGBA"))
    else:
        image = np.asarray(image)

    if image.ndim == 2:
        if not np.issubdtype(image.dtype, np.number):
            raise ImageInputError("image pixels must be numeric")
        return _scale_to_byte(image)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ImageInputError("image must be grayscale, RGB, or RGBA")
    if not np.issubdtype(image.dtype, np.number):
        raise ImageInputError("image pixels must be numeric")

    rgba = _scale_to_byte(image)
    rgb = rgba[..., :3].astype(np.float64)
    gray = np.rint(0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2])
    if rgba.shape[2] == 4:
        alpha = rgba[..., 3]
        gray = np.where(alpha == 0, 255, gray)
    return gray.astype(np.uint8)


def _scale_to_byte(values: np.ndarray) -> np.ndarray:
    if not np.all(np.isfinite(values)):
        raise ImageInputError("image pixels must be finite")
    if np.issubdtype(values.dtype, np.integer):
        if values.min(initial=0) < 0:
            raise ImageInputError("integer image pixels must be non-negative")
        data_maximum = int(values.max(initial=0))
        if data_maximum <= 255:
            return values.astype(np.uint8)
        maximum = float(np.iinfo(values.dtype).max)
        return np.rint(values.astype(np.float64) * (255.0 / maximum)).astype(np.uint8)
    floats = values.astype(np.float64)
    if floats.min(initial=0.0) < 0:
        raise ImageInputError("floating-point image pixels must be in [0, 1] or [0, 255]")
    maximum = float(floats.max(initial=0.0))
    if maximum <= 1.0:
        floats *= 255.0
    elif maximum > 255.0:
        raise ImageInputError("image pixels must be in [0, 1] or [0, 255]")
    return np.rint(floats).astype(np.uint8)


def _pixel_neighbors(mask: np.ndarray, connectivity: int):
    rows, columns = mask.shape
    points = {tuple(map(int, point)) for point in np.argwhere(mask)}
    neighbors = {point: [] for point in points}
    offsets = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    if connectivity == 8:
        offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    for row, column in sorted(points):
        for dr, dc in offsets:
            other = (row + dr, column + dc)
            if other not in points:
                continue
            if abs(dr) == abs(dc) == 1:
                # Do not add a diagonal shortcut when an orthogonal pixel
                # already connects the same corner. This avoids false cycles
                # around ordinary right-angle line bends.
                if (row + dr, column) in points or (row, column + dc) in points:
                    continue
            neighbors[(row, column)].append(other)
    for point in neighbors:
        neighbors[point].sort()
    return neighbors


def _link_key(a, b):
    return (a, b) if a <= b else (b, a)


def _polyline_weight(points) -> float:
    return sum(
        math.hypot(second[0] - first[0], second[1] - first[1])
        for first, second in zip(points, points[1:])
    )


def skeleton_to_graph(skeleton, *, connectivity: int = 8) -> tuple[Graph, np.ndarray, int]:
    """Compress a one-pixel skeleton into a required-edge multigraph.

    Coordinates in edge geometry are `(x, y)` pixel coordinates. Connected
    branch pixels are contracted to one graph vertex; degree-2 chains become
    weighted edges. Pure cycles are represented as a self-loop. With 8-neighbor
    mode, diagonal links are omitted when either orthogonal link already joins
    the corner, preventing common raster corner shortcuts.
    """
    if connectivity not in (4, 8):
        raise ImageInputError("connectivity must be 4 or 8")
    mask = np.asarray(skeleton, dtype=bool)
    if mask.ndim != 2:
        raise ImageInputError("skeleton must be a 2D mask")

    labels = connected_component_labels(mask, connectivity=1 if connectivity == 4 else 2)
    component_count = int(labels.max(initial=0))
    adjacency = _pixel_neighbors(mask, connectivity)
    branch_pixels = {point for point, neighbors in adjacency.items() if len(neighbors) >= 3}
    endpoint_pixels = {point for point, neighbors in adjacency.items() if len(neighbors) <= 1}

    # Merge adjacent branch pixels into a junction node, while keeping endpoints
    # separate so a one- or two-pixel stroke does not disappear in compression.
    clusters: list[tuple[tuple[int, int], ...]] = []
    unseen_branches = set(branch_pixels)
    while unseen_branches:
        seed = min(unseen_branches)
        cluster = {seed}
        stack = [seed]
        unseen_branches.remove(seed)
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if neighbor in unseen_branches:
                    unseen_branches.remove(neighbor)
                    cluster.add(neighbor)
                    stack.append(neighbor)
        clusters.append(tuple(sorted(cluster)))
    clusters.extend((point,) for point in sorted(endpoint_pixels))

    critical_pixels = {point for cluster in clusters for point in cluster}
    node_by_pixel: dict[tuple[int, int], str] = {}
    node_positions: dict[str, tuple[float, float]] = {}
    for node_index, cluster in enumerate(sorted(clusters, key=lambda value: value[0])):
        node_id = f"v{node_index:04d}"
        for point in cluster:
            node_by_pixel[point] = node_id
        node_positions[node_id] = (
            sum(point[1] for point in cluster) / len(cluster),
            sum(point[0] for point in cluster) / len(cluster),
        )

    # Components containing only degree-2 pixels are closed cycles. Pick one
    # stable anchor per such component so that it can be represented by a loop.
    labels_with_nodes = {
        int(labels[point]) for point in critical_pixels if labels[point] != 0
    }
    cycle_anchors = []
    for component_id in range(1, component_count + 1):
        if component_id in labels_with_nodes:
            continue
        points = sorted(point for point in adjacency if labels[point] == component_id)
        if points:
            cycle_anchors.append((component_id, points[0]))
    for component_id, anchor in cycle_anchors:
        node_id = f"v{len(node_by_pixel):04d}"
        node_by_pixel[anchor] = node_id
        node_positions[node_id] = (float(anchor[1]), float(anchor[0]))
        critical_pixels.add(anchor)

    vertices = tuple(dict.fromkeys(node_by_pixel.values()))
    visited_links = set()
    edges: list[Edge] = []

    def add_edge(start_pixel, end_pixel, pixel_path):
        start_node = node_by_pixel[start_pixel]
        end_node = node_by_pixel[end_pixel]
        centerline = tuple((float(column), float(row)) for row, column in pixel_path)
        geometry_points = [node_positions[start_node]]
        for point in centerline:
            if point != geometry_points[-1]:
                geometry_points.append(point)
        if node_positions[end_node] != geometry_points[-1]:
            geometry_points.append(node_positions[end_node])
        geometry = tuple(geometry_points)
        if len(geometry) == 1:
            weight = 0.0
        else:
            weight = _polyline_weight(geometry)
        edges.append(
            Edge(
                id=f"e{len(edges):05d}",
                u=start_node,
                v=end_node,
                weight=weight,
                required=True,
                kind="draw",
                geometry=geometry,
            )
        )

    for start_pixel in sorted(critical_pixels):
        start_node = node_by_pixel[start_pixel]
        for neighbor in adjacency[start_pixel]:
            link = _link_key(start_pixel, neighbor)
            if link in visited_links:
                continue
            if neighbor in node_by_pixel and node_by_pixel[neighbor] == start_node:
                # Internal links in a contracted junction are represented by
                # the junction node rather than separate short graph edges.
                visited_links.add(link)
                continue

            visited_links.add(link)
            path = [start_pixel, neighbor]
            previous, current = start_pixel, neighbor
            while current not in node_by_pixel:
                candidates = [point for point in adjacency[current] if point != previous]
                if not candidates:
                    break
                following = candidates[0]
                next_link = _link_key(current, following)
                if next_link in visited_links:
                    break
                visited_links.add(next_link)
                path.append(following)
                previous, current = current, following

            if current in node_by_pixel:
                add_edge(start_pixel, current, path)

    # A one-pixel foreground component has no pixel links, but retaining it as a
    # zero-length required loop lets the route/rendering layers preserve the dot.
    for point in sorted(endpoint_pixels):
        if not adjacency[point]:
            add_edge(point, point, [point])

    # Trace each all-degree-2 cycle once. The anchor is the first pixel; choose
    # the first neighbor deterministically and continue until returning to it.
    for component_id, anchor in cycle_anchors:
        neighbors = adjacency[anchor]
        if not neighbors:
            continue
        path = [anchor]
        previous = None
        current = anchor
        while True:
            candidates = [point for point in adjacency[current] if point != previous]
            if not candidates:
                break
            following = candidates[0]
            link = _link_key(current, following)
            if link in visited_links:
                break
            visited_links.add(link)
            path.append(following)
            previous, current = current, following
            if current == anchor:
                break
        if len(path) > 1 and path[-1] == anchor:
            add_edge(anchor, anchor, path)

    component_labels = np.array(labels, copy=True)
    component_labels.setflags(write=False)
    return Graph(vertices=vertices, edges=tuple(edges)), component_labels, component_count


def analyze_image(image, *, threshold: int = 128, invert: bool = False, connectivity: int = 8) -> ImageAnalysis:
    """Threshold an image, thin its foreground, label it, and compress its graph."""
    if not isinstance(threshold, int) or not 0 <= threshold <= 255:
        raise ImageInputError("threshold must be an integer in [0, 255]")
    if connectivity not in (4, 8):
        raise ImageInputError("connectivity must be 4 or 8")
    gray = _as_grayscale(image)
    foreground = gray > threshold if invert else gray <= threshold
    skeleton = skeletonize(foreground)
    graph, labels, _ = skeleton_to_graph(skeleton, connectivity=connectivity)
    for array in (foreground, skeleton):
        array.setflags(write=False)
    return ImageAnalysis(foreground, skeleton, labels, graph, connectivity)

