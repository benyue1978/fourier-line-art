# Algorithm Test Plan

The solver is tested independently of image processing and Manim. Small graphs
have hand-computable optimum costs; larger fixtures check invariants and limits.

## Data structures

| Area | Cases | Expected contract |
| --- | --- | --- |
| Weighted multigraph | parallel edges, isolated vertices, self-loops | Distinct edge IDs remain distinct; adjacency and degree count undirected incidences correctly |
| Validation | duplicate IDs, missing endpoints, negative/non-finite weights | Reject invalid input with a specific validation error |
| Route | open/closed sequence, repeated edge traversals | Each traversal has a valid orientation; route cost equals the sum of traversed edge weights |

## Chinese Postman Problem (CPP)

| Fixture | Assertion |
| --- | --- |
| Eulerian triangle | Every edge exactly once; cost 3 |
| Three-edge path | Closed route covers every edge; cost 6 |
| Uneven weighted graph | Cost equals required-edge weight plus minimum odd-vertex matching cost |
| Parallel edges and a loop | Euler tour remains valid and does not collapse edge identities |
| Disconnected required graph | Explicit infeasible-route error |
| Same input twice | Deterministic traversal order and cost |

## Rural Postman Problem (RPP)

| Fixture | Assertion |
| --- | --- |
| Two required components with cheap optional bridge | Closed route covers every required edge; exact optimum cost 6 |
| Alternative expensive connector | Solver chooses the lower-cost connector walk |
| Optional edge already joins required components | Route uses it only when it improves a feasible closed route |
| Required component with branches | Every required edge is covered, including repeated traversals where necessary |
| No optional connection between required components | Explicit infeasible-route error |
| State-space limit | Exact solver refuses oversized instances with a documented exception |
| Frederickson-style fallback | MST connector paths join required components; CPP returns a closed feasible route |

The exact RPP implementation uses Dijkstra on an edge-coverage state graph for
small compressed graphs. Its exponential dependence on the number of required
edges is intentional and surfaced through a limit; this is not presented as a
scalable general-purpose RPP implementation.

## Fourier approximation

| Fixture | Assertion |
| --- | --- |
| Constant and single rotating term | Correct coefficients and reconstruction |
| Full set of DFT terms | Reconstructs all input samples within floating-point tolerance |
| Increasing term count | Orthogonal partial reconstruction RMS does not increase |
| Invalid term count or empty input | Clear validation error |
| Multiple channels | Parseval error curve agrees with direct RMS computation |

## Image-to-graph conversion

Image preprocessing tests use tiny synthetic masks rather than screenshots:

- one-pixel line and a plus junction;
- closed loop with no endpoint;
- two disconnected strokes;
- diagonal-only contact under 4-neighbor and 8-neighbor connectivity;
- malformed empty mask.

The tests assert component counts, edge coverage, graph connectivity, and route
geometry. They avoid brittle pixel-by-pixel snapshots where multiple skeletons
are equally valid.

## End-to-end solve result

- A connected drawing selects CPP; a disconnected drawing selects exact RPP
  until the explicit limit, then the MST/CPP fallback.
- The resulting route is closed, has draw and pen-up samples, and yields
  monotone XY/height RMS curves.
- NPZ output contains masks, route samples, ordered Fourier frequencies and
  coefficients, error arrays, and schema-versioned metadata.

## Running tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```

