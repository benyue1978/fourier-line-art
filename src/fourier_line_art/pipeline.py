"""End-to-end local solver pipeline and renderer-neutral result files."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np

from .fourier import FourierSeries, RealFourierSeries
from .image import ImageAnalysis, ImageInputError, analyze_image
from .models import Route
from .routes import RouteSamples, add_pen_up_candidates, polyline_from_route
from .solvers import ProblemTooLarge, solve_cpp, solve_rpp_exact, solve_rpp_frederickson


class EmptyDrawingError(ValueError):
    """The input image contains no foreground line art."""


@dataclass(frozen=True, slots=True)
class SolveResult:
    """All data needed by a separate animation/rendering layer."""

    analysis: ImageAnalysis
    route: Route
    samples: RouteSamples
    xy_series: FourierSeries
    x_series: RealFourierSeries
    y_series: RealFourierSeries
    height_series: FourierSeries
    xy_rms_errors: np.ndarray
    height_rms_errors: np.ndarray
    relative_xy_rms_errors: np.ndarray
    solver_name: str
    threshold: int
    invert: bool
    lift_height: float

    @property
    def term_count(self) -> int:
        return self.xy_series.term_count

    def save_npz(self, output_path: str | Path) -> Path:
        """Save compressed samples, coefficients, image masks, and metadata."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema_version": 2,
            "solver": self.solver_name,
            "required_edge_count": sum(edge.required for edge in self.analysis.graph.edges),
            "graph_edge_count": len(self.analysis.graph.edges),
            "component_count": self.analysis.component_count,
            "route_cost": self.route.cost,
            "route_closed": self.route.is_closed,
            "sample_count": int(len(self.samples.points)),
            "fourier_term_count": self.term_count,
            "coordinate_harmonic_count": self.x_series.term_count,
            "threshold": self.threshold,
            "invert": self.invert,
            "connectivity": self.analysis.connectivity,
            "lift_height": self.lift_height,
            "xy_convention": "separate real x cosine and y sine harmonics; cycles per period",
        }
        with output_path.open("wb") as output:
            np.savez_compressed(
                output,
                metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
                foreground=self.analysis.foreground,
                skeleton=self.analysis.skeleton,
                component_labels=self.analysis.component_labels,
                samples_xyz=self.samples.points,
                draw_mask=self.samples.draw,
                xy_frequencies=self.xy_series.ordered_frequencies,
                xy_coefficients=self.xy_series.ordered_coefficients,
                x_mean=np.asarray(self.x_series.mean),
                x_frequencies=self.x_series.ordered_frequencies,
                x_amplitudes=self.x_series.ordered_amplitudes,
                x_phases=self.x_series.ordered_phases,
                x_rms_errors=self.x_series.rms_errors(),
                y_mean=np.asarray(self.y_series.mean),
                y_frequencies=self.y_series.ordered_frequencies,
                y_amplitudes=self.y_series.ordered_amplitudes,
                y_sine_phases=self.y_series.ordered_phases + np.pi / 2,
                y_rms_errors=self.y_series.rms_errors(),
                height_frequencies=self.height_series.ordered_frequencies,
                height_coefficients=self.height_series.ordered_coefficients,
                xy_rms_errors=self.xy_rms_errors,
                height_rms_errors=self.height_rms_errors,
                relative_xy_rms_errors=self.relative_xy_rms_errors,
            )
        return output_path


def solve_image(
    image,
    *,
    threshold: int = 128,
    invert: bool = False,
    connectivity: int = 8,
    sample_count: int = 512,
    lift_height: float = 1.0,
    max_required_edges: int = 15,
    max_rpp_states: int = 2_000_000,
) -> SolveResult:
    """Analyze a line-art image, solve its closed route, and compute its DFT."""
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 2:
        raise ImageInputError("sample_count must be an integer of at least 2")
    try:
        lift_height = float(lift_height)
    except (TypeError, ValueError) as exc:
        raise ImageInputError("lift_height must be finite and positive") from exc
    if not math.isfinite(lift_height) or lift_height <= 0:
        raise ImageInputError("lift_height must be finite and positive")
    analysis = analyze_image(
        image,
        threshold=threshold,
        invert=invert,
        connectivity=connectivity,
    )
    if analysis.component_count == 0 or not analysis.graph.edges:
        raise EmptyDrawingError("image contains no drawable foreground line art")

    if analysis.component_count == 1:
        solver_name = "chinese_postman"
        route = solve_cpp(analysis.graph)
    else:
        augmented_graph = add_pen_up_candidates(analysis.graph, lift_height=lift_height)
        try:
            route = solve_rpp_exact(
                augmented_graph,
                max_required_edges=max_required_edges,
                max_states=max_rpp_states,
            )
            solver_name = "rural_postman_exact_state_dijkstra"
        except ProblemTooLarge:
            route = solve_rpp_frederickson(augmented_graph)
            solver_name = "rural_postman_frederickson_mst_cpp_heuristic"

    polyline = polyline_from_route(route, lift_height=lift_height)
    samples = polyline.resample(sample_count)
    xy_signal = samples.points[:, 0] + 1j * samples.points[:, 1]
    xy_series = FourierSeries.from_samples(xy_signal)
    x_series = RealFourierSeries.from_samples(samples.points[:, 0])
    y_series = RealFourierSeries.from_samples(samples.points[:, 1])
    height_series = FourierSeries.from_samples(samples.points[:, 2])
    xy_errors = xy_series.rms_errors()
    height_errors = height_series.rms_errors()
    image_diagonal = max(
        float(np.hypot(analysis.foreground.shape[1], analysis.foreground.shape[0])),
        1.0,
    )
    relative_xy_errors = xy_errors / image_diagonal
    relative_xy_errors.setflags(write=False)

    return SolveResult(
        analysis=analysis,
        route=route,
        samples=samples,
        xy_series=xy_series,
        x_series=x_series,
        y_series=y_series,
        height_series=height_series,
        xy_rms_errors=xy_errors,
        height_rms_errors=height_errors,
        relative_xy_rms_errors=relative_xy_errors,
        solver_name=solver_name,
        threshold=threshold,
        invert=invert,
        lift_height=lift_height,
    )
