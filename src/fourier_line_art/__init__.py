"""Core algorithms for Fourier line-art drawing animations."""

from .fourier import FourierSeries, FourierValidationError
from .image import ImageAnalysis, ImageInputError, analyze_image, skeleton_to_graph
from .models import Edge, Graph, GraphValidationError, Route, Traversal
from .pipeline import EmptyDrawingError, SolveResult, solve_image
from .routes import (
    RouteGeometryError,
    RoutePolyline,
    RouteSamples,
    add_pen_up_candidates,
    polyline_from_route,
)
from .solvers import (
    NoFeasibleRoute,
    ProblemTooLarge,
    solve_cpp,
    solve_rpp_exact,
    solve_rpp_frederickson,
)

__all__ = [
    "Edge",
    "EmptyDrawingError",
    "FourierSeries",
    "FourierValidationError",
    "Graph",
    "GraphValidationError",
    "ImageAnalysis",
    "ImageInputError",
    "NoFeasibleRoute",
    "ProblemTooLarge",
    "Route",
    "RouteGeometryError",
    "RoutePolyline",
    "RouteSamples",
    "SolveResult",
    "Traversal",
    "analyze_image",
    "add_pen_up_candidates",
    "polyline_from_route",
    "solve_image",
    "solve_rpp_frederickson",
    "skeleton_to_graph",
    "solve_cpp",
    "solve_rpp_exact",
]

