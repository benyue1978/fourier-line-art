# Fourier Line Art

Turn a black-and-white line drawing into one closed route, approximate the route with a discrete Fourier transform, and render the reconstruction as a phone-portrait MP4. The image solver, saved result, and video renderer are separate stages.

The repository includes two sample images: a disconnected-stroke smiley and a connected county-boundary map of Jilin Province. They demonstrate the two route cases without needing to prepare an input image.

## Quick start

Requirements: Python 3.11 or newer, pip, and FFmpeg on `PATH`. The video extra installs Matplotlib; FFmpeg is a system executable and must be installed separately if it is not already available.

```bash
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev,video]"
```

Solve the sample image into a renderer-neutral `.npz` file:

```bash
fourier-line-art examples/demo_smiley.png \
  --output build/smiley.npz \
  --connectivity 8 \
  --samples 512
```

Render that result as a 9:16, 1080 × 1920, 30 fps MP4:

```bash
fourier-line-art-render build/smiley.npz \
  --output build/smiley-portrait.mp4
```

The default video is about 14 seconds. With 512 route samples, each coordinate has 256 harmonics. The video uses progressively denser counts (1, 2, 3, 4, 6, 8, 12, …, 256), animating a full period for 0.8 seconds at each stage before holding on the final reconstruction. For a faster preview, render smaller dimensions and fewer frames per second:

```bash
fourier-line-art-render build/smiley.npz \
  --output build/preview.mp4 \
  --width 540 --height 960 --fps 15 \
  --stage-seconds 0.5 --end-hold-seconds 0.8
```

Open the MP4 in a desktop video player or copy it to a phone. The file uses H.264 with `yuv420p` pixel format for broad mobile playback support.

## Connected example: Jilin county map

`examples/jilin_counties.png` is a label-free black-and-white map of Jilin's county-level boundaries. It becomes one connected skeleton component, so the solver uses the Chinese Postman solver and does not need pen-up connectors.

Solve and render it with:

```bash
fourier-line-art examples/jilin_counties.png \
  --output build/jilin-counties.npz \
  --threshold 128 \
  --connectivity 8 \
  --samples 4096

fourier-line-art-render build/jilin-counties.npz \
  --output build/jilin-counties-portrait.mp4
```

At 4096 route samples, each coordinate has 2048 harmonics. With the default render settings, the portrait video is about 16 seconds. The RMS readout measures Fourier reconstruction error against the sampled route; it is not a measure of geographic boundary accuracy.

The county outlines were rasterized from GeoJSON in the [ChinaGeoJson repository](https://github.com/zhChuXiao/ChinaGeoJson), which attributes its map data to [Alibaba DataV.GeoAtlas](https://datav.aliyun.com/). This image is an algorithm example, not an official reference map.

The smiley example uses the same commands with `examples/demo_smiley.png`, as shown in Quick start. Its separate strokes are connected by optional pen-up travel; the Jilin example instead exercises a connected branched network.

## How route solving works

### Image to skeleton graph

The solver converts the image to grayscale, treats transparent pixels as background, and thresholds dark pixels into a foreground mask. It then thins the mask to a one-pixel skeleton and labels its connected components using the selected 4- or 8-neighbor rule. Connectivity is determined from this processed raster, so a small gap or a diagonal-only contact can change whether two strokes count as connected.

The skeleton becomes an undirected weighted graph. Endpoints and junctions become vertices; chains of degree-2 pixels are compressed into edges weighted by their pixel length. Every skeleton edge is a required drawing edge. The solver does not ask the user to pair branches at a crossing: it plans a walk that covers every required edge.

### One connected component: Chinese Postman Problem

When the required graph is connected, the solver uses the undirected Chinese Postman Problem (CPP) to find a minimum-cost closed walk that covers every edge. It finds shortest paths between odd-degree vertices, pairs those vertices with a minimum-weight perfect matching, duplicates the matched paths to make all degrees even, then builds an Euler tour with Hierholzer's algorithm. The route can retrace some drawing edges; on the image those retraces lie on top of the same strokes. No pen-up connectors are needed.

### Multiple components: Rural Postman Problem and pen-up travel

When the skeleton has multiple connected components, all drawing edges remain required. The solver adds optional straight pen-up candidates between vertices in different components. Their cost is the XY travel distance plus a lift-and-land cost based on `--lift-height`. A Rural Postman Problem (RPP) solver chooses connectors and repeated edges to produce one closed route covering all required drawing edges.

The route records pen-down travel at height `lift-height` and lifted travel at height zero. Thus separate 2D strokes are joined into one continuous 3D route with a draw-state mask; the program does not make each stroke its own Fourier formula. Small compressed RPP instances are solved exactly with a state-space search, up to the defaults of 15 required edges and 2,000,000 states. If either limit is exceeded, the solver uses a Frederickson-style minimum-spanning-tree/CPP heuristic. That fallback returns a feasible closed route when the candidates connect the components, but it does not promise the global minimum cost.

### One sampled route, then Fourier approximation

The closed route is resampled uniformly by arc length and saved as `(x, y, height)` samples plus a draw-state mask. The solver transforms the real X and Y signals separately (and also preserves the height signal). At each video stage, both coordinates use the same prefix of ascending frequencies, from 1 through `N`; their amplitudes and phases remain independent. In shorthand, the displayed curve is the pair `(x_N(t), y_N(t))`:

$$
x_N(t)=x_0+\sum_{k=1}^{N}A_{x,k}\cos(2\pi kt+\phi_{x,k}),\qquad
y_N(t)=y_0+\sum_{k=1}^{N}A_{y,k}\sin(2\pi kt+\phi_{y,k}).
$$

Its first harmonic pair traces an ellipse, or a degenerate ellipse in special cases.

`--samples` controls route-sample resolution. A 512-sample route has 256 positive-frequency harmonics per coordinate; 4096 samples have 2048. Increasing `N` adds detail to the same closed route. The displayed XY RMS error compares the partial Fourier reconstruction with the solved route samples, including any pen-up XY travel. Its percentage is normalized by the source-image diagonal. It measures Fourier truncation error, not how accurately the skeleton represents the original artwork or a geographic boundary.

## What the video shows

- **One combined diagram:** the X circle chain is above the source drawing, the Y chain is to its left, and the map shows the current blue reconstruction. Their fixed output axes share the map's coordinate frame: the X endpoint has exactly the map point's x coordinate, and the Y endpoint has exactly its y coordinate. Both use the same source-pixel scale.
- **Formula card:** separate real Fourier formulas for `x_N(t)` and `y_N(t)`, plus the live normalized pen-tip coordinate and current term count `N`.
- **Coordinate guides:** one dashed vertical line connects the X endpoint directly to the map pen tip; one dashed horizontal line connects the Y endpoint directly to the same point. These dashed guides show the direct X/Y projections; they are not pen-up paths. There is no origin bridge or coordinate offset.
- **Draw and lift states:** the source skeleton is a light reference, the current pen-down reconstruction is blue, and pen-up travel is a thin pale trace on the map. That pale trace fades as terms are added and disappears at full precision. Pen-up connectors are not drawn in the rotating-circle chains.
- **Fourier stages:** the renderer uses a progressively denser sequence of term counts, animating one full period at each stage. The formula card shows the current `N`; the numeric equation previews the first active terms and uses an ellipsis for the rest.
- **Error graph:** relative XY RMS error versus term count. The readout gives both the relative error and the pixel-space RMS error for the current stage. At full DFT resolution the sampled route reconstructs to numerical precision; this does not remove image-thresholding or skeletonization error.

The NPZ is also useful without rendering: it contains the source masks, sampled `(x, y, height)` route, draw-state mask, ordered Fourier frequencies and coefficients, RMS errors, and solver metadata. Inspect its metadata with:

```bash
python - <<'PY'
import json
import numpy as np

with np.load("build/smiley.npz", allow_pickle=False) as result:
    print(json.loads(str(result["metadata_json"])))
    print("route samples:", result["samples_xyz"].shape)
    print("X harmonics:", result["x_amplitudes"].shape[0])
    print("Y harmonics:", result["y_amplitudes"].shape[0])
PY
```

## Using another image

The input should be a high-contrast black-and-white line drawing. The default treats dark pixels as ink and uses threshold 128. Transparent pixels are treated as background. For light ink on a dark background, pass `--invert`.

```bash
fourier-line-art path/to/drawing.png \
  --output build/drawing.npz \
  --threshold 150 \
  --connectivity 8 \
  --samples 512 \
  --lift-height 1.0

fourier-line-art-render build/drawing.npz \
  --output build/drawing.mp4
```

Useful solver options:

| Option | Default | Meaning |
| --- | ---: | --- |
| `--threshold` | `128` | Grayscale threshold from 0 to 255 |
| `--invert` | off | Use light pixels as foreground |
| `--connectivity` | `8` | Pixel-neighbor connectivity, 4 or 8 |
| `--samples` | `512` | Uniform route samples; 256 positive-frequency harmonics per coordinate at this size |
| `--lift-height` | `1.0` | Height assigned to pen-down travel in the 3D route model |
| `--max-rpp-required-edges` | `15` | Required-edge limit for exact disconnected-route solving |
| `--max-rpp-states` | `2000000` | State limit for the exact rural postman search |

For larger disconnected graphs, the solver switches to a feasible MST/CPP heuristic. The NPZ metadata identifies which solver handled the image. Its route is usable, but the heuristic does not claim a globally shortest route.

## Development

Install the algorithm and test dependencies without the video renderer:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

Install the optional video renderer when working on MP4 output:

```bash
python -m pip install -e ".[video]"
```

Run the tools from a checkout without installing console scripts:

```bash
python -m fourier_line_art examples/demo_smiley.png --output build/smiley.npz
python -m fourier_line_art.rendering build/smiley.npz --output build/smiley-portrait.mp4
```

Regenerate the demo image after changing `examples/make_demo.py`:

```bash
python examples/make_demo.py
```

### Project layout

```text
src/fourier_line_art/
  image.py        thresholding, skeletonization, graph extraction
  models.py       validated graph, edge, and route structures
  solvers.py      CPP and RPP solvers
  routes.py       pen-up connectors and 3D route sampling
  fourier.py      DFT, reconstruction, and RMS error calculations
  pipeline.py     image-to-NPZ solver pipeline
  rendering.py    NPZ-to-portrait-MP4 renderer
tests/            algorithm, pipeline, data-contract, and render tests
docs/             algorithm notes and test plan
examples/         sample input and its generator
```

The renderer reads only the versioned NPZ result. This keeps image solving and route algorithms independent from the video backend and makes it possible to add another renderer later.

## Algorithm limits and further details

The extracted skeleton is a pixel-level estimate of a centerline. Thick, touching, noisy, anti-aliased, or ambiguous strokes can change its topology; inspect the source mask and skeleton in the NPZ when the route looks unexpected. Exact RPP search is intended for small instances, while the MST/CPP fallback has no general optimality guarantee. For implementation details and solver invariants, see [`docs/ALGORITHMS.md`](docs/ALGORITHMS.md); for algorithm and renderer coverage, see [`docs/TEST_PLAN.md`](docs/TEST_PLAN.md).
