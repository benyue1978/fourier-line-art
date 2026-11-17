# Solver Algorithms and Limits

The implementation uses separate stages so image parsing, route solving, and
animation rendering can be tested independently.

## Image and graph

1. Convert PNG/RGB/RGBA pixels to grayscale; transparent pixels count as
   background. Threshold to a binary foreground mask.
2. Thin the mask with `skimage.morphology.skeletonize` and label 4- or
   8-connected components with `skimage.measure.label`.
3. Build the skeleton pixel adjacency graph. Adjacent branch pixels are
   contracted to a junction node; degree-2 pixel chains become weighted edges.
   A component with no endpoint or junction is retained as a self-loop. An
   isolated foreground pixel is retained as a zero-length loop.

This graph contraction is a representation step: CPP/RPP cover the resulting
compressed graph edges. Pixels inside a contracted junction are represented by
its center point, so a junction cluster is not treated as several separate
stroke edges.

## Connected drawings: Chinese Postman Problem

For a connected undirected graph with every drawing edge required:

1. Run Dijkstra from each odd-degree vertex to get shortest paths.
2. Find a minimum-weight perfect matching on odd-degree vertices using
   NetworkX's blossom implementation.
3. Duplicate the matched shortest paths to make all degrees even.
4. Run Hierholzer's algorithm to produce a closed Euler tour.

This is exact for non-negative edge weights on a connected undirected graph.
Parallel edges remain distinct by edge ID; loops contribute degree two.

## Disconnected drawings: Rural Postman Problem

Each skeleton edge is required. Candidate straight-line pen-up edges are
optional, with cost equal to XY distance plus lift and landing cost. For small
compressed graphs, the exact solver runs Dijkstra on states
`(current vertex, mask of covered required edges)`. A transition traverses one
graph edge and sets its required-edge bit when applicable. For each possible
start at a required-edge endpoint, it finds the shortest return to the start
with the full mask. This is exact but exponential in required-edge count.

The default limit is 15 required edges and 2,000,000 explored states. Above the
limit, the pipeline uses the Frederickson-style construction: join required
components with a minimum spanning tree of shortest connector paths, then solve
the augmented connected graph as a CPP. This always constructs a feasible
closed route when the candidate graph connects all required components, but
the image pipeline does not assert optimality for this fallback.

## Fourier series and errors

The closed 3D route is resampled at uniform arc length. XY is represented as
`z(t) = x(t) + i*y(t)`; height is a separate real-valued sample channel. DFT
coefficients use the `1/N` normalization and integer frequencies in cycles per
period. Terms are ordered by coefficient energy, so each nested complex-signal
truncation has non-increasing sample RMS by Parseval's identity. The NPZ output
stores ordered frequencies and coefficients for both channels, the source
draw mask, samples, and RMS curves.

The exact RPP state search is suitable for testing and small inputs, not large
maps. The fallback has no optimality guarantee for a general image-derived
instance. Both behaviors are explicit in the API and tested.

## References

- Frederickson's RPP heuristic family and its MST connection step are described
  in [Heuristics for the Rural Postman Problem](https://www.sciencedirect.com/science/article/abs/pii/S0305054809001993).
- The RPP is NP-hard when required edges induce a disconnected subgraph; see
  [Algorithms for the Rural Postman Problem](https://www.sciencedirect.com/science/article/abs/pii/030505489400070O).

