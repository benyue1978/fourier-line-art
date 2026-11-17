# Fourier Line Art

Python algorithms for turning black-and-white line art into a closed route and
approximating that route with rotating Fourier vectors. The project keeps image
solving separate from Manim rendering so the same solver result can later drive
an interactive preview or an exported video.

## Current implementation

- Validated undirected weighted multigraph data structures with required and
  optional edges. Optional `pen_up` edges model hidden travel between strokes.
- Exact Chinese Postman solver (CPP): shortest paths, minimum-weight perfect
  matching of odd-degree vertices, and Hierholzer Euler-tour reconstruction.
- Exact small-instance Rural Postman solver (RPP): Dijkstra over
  `(vertex, covered-required-edge-mask)` states. It is exponential by nature and
  rejects inputs that exceed explicit limits.
- Larger RPP instances fall back to the named Frederickson-style heuristic:
  connect required components with a shortest-path minimum spanning tree, then
  solve the connected augmentation as a CPP. The result is feasible, with no
  optimality claim for image-derived graphs.
- PNG/RGB/RGBA thresholding, transparent-background handling, skeletonization,
  4/8-neighbor component labeling, and degree-2 chain compression into graph
  edges with pixel geometry.
- Complete optional pen-up connectors between disconnected required-edge
  components, 3D lift/move/lower route geometry, and uniform arc-length
  resampling of `(x, y, h)` plus source draw-state samples.
- DFT coefficients, coefficient-energy ordering, partial reconstruction, and
  RMS error reporting.
- `solve_image(...)` chooses CPP for a connected drawing and exact RPP for a
  small disconnected drawing, then switches to the MST/CPP heuristic when the
  exact RPP state limit is exceeded. It saves a renderer-neutral `.npz` file.
- Test matrix in [`docs/TEST_PLAN.md`](docs/TEST_PLAN.md).
- Algorithm choices and limits in [`docs/ALGORITHMS.md`](docs/ALGORITHMS.md).

Manim scenes and interactive rendering remain a separate next layer; the NPZ
file already provides a stable input boundary for that renderer.

## Environment and tests

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest
```

Solve a local line-art image and write the renderer-neutral result:

```bash
fourier-line-art drawing.png --output build/drawing.npz --connectivity 8
```

