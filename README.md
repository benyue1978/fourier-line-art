# Fourier Line Art

Turn a black-and-white line drawing into one closed route, approximate the route with a discrete Fourier transform, and render the reconstruction as a phone-portrait MP4. The image solver, saved result, and video renderer are separate stages.

The repository includes a small disconnected-stroke smiley image so the full workflow can be tried without preparing an input image.

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
  --samples 256
```

Render that result as a 9:16, 1080 × 1920, 30 fps MP4:

```bash
fourier-line-art-render build/smiley.npz \
  --output build/smiley-portrait.mp4
```

The default video shows increasing Fourier term counts (1, 2, 4, 8, …, all terms). Each stage traces one complete period. The final stage holds on the full reconstruction. For a faster preview, render smaller dimensions and fewer frames per second:

```bash
fourier-line-art-render build/smiley.npz \
  --output build/preview.mp4 \
  --width 540 --height 960 --fps 15 \
  --stage-seconds 0.5 --end-hold-seconds 0.8
```

Open the MP4 in a desktop video player or copy it to a phone. The file uses H.264 with `yuv420p` pixel format for broad mobile playback support.

## What the video shows

- **Formula card:** the partial Fourier sum and the current term count `N`.
- **Upper panel:** the extracted centerline in gray and its current Fourier reconstruction in blue. Orange dashed parts show pen-up travel between strokes.
- **Lower panel:** up to 24 rotating Fourier circles and the endpoint trail. When more terms are active, the remaining sum appears as an orange dashed vector; the trail and error still use the complete `N`-term reconstruction.
- **Error graph:** relative XY RMS error versus term count. The readout gives both the relative error and the pixel-space RMS error for the current stage.

The NPZ is also useful without rendering: it contains the source masks, sampled `(x, y, height)` route, draw-state mask, ordered Fourier frequencies and coefficients, RMS errors, and solver metadata. Inspect its metadata with:

```bash
python - <<'PY'
import json
import numpy as np

with np.load("build/smiley.npz", allow_pickle=False) as result:
    print(json.loads(str(result["metadata_json"])))
    print("route samples:", result["samples_xyz"].shape)
    print("Fourier terms:", result["xy_coefficients"].shape[0])
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
| `--samples` | `512` | Uniform route samples and Fourier terms |
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

## Algorithm notes and limits

- Connected line art is converted to a weighted graph and solved as a Chinese Postman Problem, producing a closed walk that covers every required edge.
- Disconnected strokes receive optional pen-up connectors. The route and its sampled `(x, y, height)` points are still represented as one closed parameterized path.
- Small Rural Postman Problems use an exact state-space search. The search has explicit edge and state limits because its state count grows exponentially.
- Larger Rural Postman Problems use the MST/CPP heuristic described in [`docs/ALGORITHMS.md`](docs/ALGORITHMS.md).
- Fourier terms are ordered by coefficient energy. The saved RMS error is calculated from the omitted coefficient energy, and the renderer displays the progressive approximation.
- A raster skeleton is a pixel-level estimate of a centerline. Thick, touching, noisy, or ambiguous strokes can change the extracted topology; inspect the NPZ masks and rendered comparison when an input looks unexpected.

See [`docs/TEST_PLAN.md`](docs/TEST_PLAN.md) for solver and renderer test coverage.
