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

The closed 3D route is resampled at uniform arc length. The renderer transforms
the real x and y signals separately. Each series stores its mean and
positive-frequency harmonics as amplitude/phase terms. Conjugate frequency
pairs become one cosine harmonic; the y phase is shifted by π/2 when expressed
as a sine. For the animation, X and Y are paired by ascending frequency, so
stage N contains frequencies 1 through N on both axes. The first pair therefore
traces an ellipse (or a degenerate ellipse), and every later stage adds a
matched frequency to both coordinates. Each partial error is calculated from
the energy of the omitted frequencies. The NPZ stores the source draw mask,
3D samples, complex coefficients, and separate x/y amplitudes, phases,
frequencies, and error curves.

## Portrait video

`fourier-line-art-render` reads the schema-versioned NPZ and creates one
composite diagram: X's epicycle chain sits above the source drawing and Y's
chain sits to its left. The X wheel's fixed horizontal output axis is aligned
with the map's x coordinates; the Y wheel's fixed vertical output axis is
aligned with the map's y coordinates. Thus the X endpoint and map point have
the exact same x value, while the Y endpoint and map point have the exact same
y value. A vertical dashed line connects the X endpoint to the map point, and
a horizontal dashed line connects the Y endpoint to that point. There are no
coordinate offsets or bridge segments. Pen-up travel is omitted from the
visible route at full precision; partial Fourier curves remain continuous at
earlier stages.

Both coordinate chains use the same image-pixel-to-map scale, with X extent
following the image width and Y extent following its height. Their centers are
fixed at the coordinate means; the harmonic endpoint is therefore the direct
partial-sum coordinate. The X and Y projections meet at exactly the paired
partial-sum point for the same time parameter. At N=1, that point traces an
ellipse (or a degenerate ellipse when the phases or amplitudes collapse it).

The default 512 route samples produce 256 positive-frequency harmonics per
coordinate. Term counts follow a denser milestone sequence from 1 through 256,
with a full period at each stage and 0.8 seconds per stage. The error graph uses
the combined XY RMS error for the same active harmonic count. The current
backend uses Matplotlib and FFmpeg to write H.264 `yuv420p` MP4 at 9:16 by
default. It can be replaced or supplemented without changing the solver or the
NPZ contract.

The exact RPP state search is suitable for testing and small inputs, not large
maps. The fallback has no optimality guarantee for a general image-derived
instance. Both behaviors are explicit in the API and tested.

## References

- Frederickson's RPP heuristic family and its MST connection step are described
  in [Heuristics for the Rural Postman Problem](https://www.sciencedirect.com/science/article/abs/pii/S0305054809001993).
- The RPP is NP-hard when required edges induce a disconnected subgraph; see
  [Algorithms for the Rural Postman Problem](https://www.sciencedirect.com/science/article/abs/pii/030505489400070O).

