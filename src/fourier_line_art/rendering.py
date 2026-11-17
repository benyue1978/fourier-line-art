"""Render solver results as a single xy-controlled portrait diagram."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np

from .fourier import RealFourierSeries


class RenderInputError(ValueError):
    """A saved solver result cannot be used as an animation input."""


@dataclass(frozen=True, slots=True)
class RenderData:
    """Validated route and separate real Fourier series for both coordinates."""

    metadata: dict
    skeleton: np.ndarray
    samples_xyz: np.ndarray
    draw_mask: np.ndarray
    x_mean: float
    x_frequencies: np.ndarray
    x_amplitudes: np.ndarray
    x_phases: np.ndarray
    x_rms_errors: np.ndarray
    y_mean: float
    y_frequencies: np.ndarray
    y_amplitudes: np.ndarray
    y_sine_phases: np.ndarray
    y_rms_errors: np.ndarray

    @property
    def term_count(self) -> int:
        return int(len(self.x_frequencies))

    @property
    def sample_count(self) -> int:
        return int(len(self.samples_xyz))


def _axis_values_from_series(series: RealFourierSeries, *, y_axis: bool = False):
    phases = series.ordered_phases
    if y_axis:
        phases = phases + np.pi / 2
    return (
        series.mean,
        series.ordered_frequencies,
        series.ordered_amplitudes,
        phases,
        series.rms_errors(),
    )


def load_render_data(npz_path: str | Path) -> RenderData:
    """Load a solver NPZ, supporting schema v1 and the separate-axis v2."""
    path = Path(npz_path)
    try:
        with np.load(path, allow_pickle=False) as archive:
            if "metadata_json" not in archive.files:
                raise RenderInputError("NPZ has no metadata_json field")
            try:
                metadata = json.loads(str(archive["metadata_json"]))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise RenderInputError("NPZ metadata_json is invalid") from exc
            schema = metadata.get("schema_version")
            if schema not in (1, 2):
                raise RenderInputError(f"unsupported result schema version: {schema!r}")
            required = {"skeleton", "samples_xyz", "draw_mask"}
            if schema == 2:
                required.update({
                    "x_mean", "x_frequencies", "x_amplitudes", "x_phases", "x_rms_errors",
                    "y_mean", "y_frequencies", "y_amplitudes", "y_sine_phases", "y_rms_errors",
                })
            missing = required.difference(archive.files)
            if missing:
                raise RenderInputError(
                    "NPZ is missing renderer fields: " + ", ".join(sorted(missing))
                )

            skeleton = np.array(archive["skeleton"], dtype=bool, copy=True)
            samples = np.array(archive["samples_xyz"], dtype=np.float64, copy=True)
            draw_mask = np.array(archive["draw_mask"], dtype=bool, copy=True)
            if schema == 2:
                axis_data = (
                    float(archive["x_mean"]),
                    np.array(archive["x_frequencies"], dtype=np.int64, copy=True),
                    np.array(archive["x_amplitudes"], dtype=np.float64, copy=True),
                    np.array(archive["x_phases"], dtype=np.float64, copy=True),
                    np.array(archive["x_rms_errors"], dtype=np.float64, copy=True),
                    float(archive["y_mean"]),
                    np.array(archive["y_frequencies"], dtype=np.int64, copy=True),
                    np.array(archive["y_amplitudes"], dtype=np.float64, copy=True),
                    np.array(archive["y_sine_phases"], dtype=np.float64, copy=True),
                    np.array(archive["y_rms_errors"], dtype=np.float64, copy=True),
                )
            else:
                x_series = RealFourierSeries.from_samples(samples[:, 0])
                y_series = RealFourierSeries.from_samples(samples[:, 1])
                axis_data = (
                    *_axis_values_from_series(x_series),
                    *_axis_values_from_series(y_series, y_axis=True),
                )
    except RenderInputError:
        raise
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RenderInputError(f"could not read solver result {path}: {exc}") from exc

    (x_mean, x_freq, x_amp, x_phase, x_err,
     y_mean, y_freq, y_amp, y_phase, y_err) = axis_data
    if skeleton.ndim != 2 or not skeleton.any():
        raise RenderInputError("saved skeleton must be a non-empty 2D mask")
    if samples.ndim != 2 or samples.shape[1] != 3 or len(samples) < 2:
        raise RenderInputError("samples_xyz must have shape (M, 3) with M >= 2")
    if draw_mask.shape != (len(samples),):
        raise RenderInputError("draw_mask must have one value per route sample")
    term_count = len(x_freq)
    if term_count < 1 or len(y_freq) != term_count:
        raise RenderInputError("x and y must have the same positive harmonic count")
    for name, frequencies, amplitudes, phases, errors in (
        ("x", x_freq, x_amp, x_phase, x_err),
        ("y", y_freq, y_amp, y_phase, y_err),
    ):
        if frequencies.shape != (term_count,) or amplitudes.shape != (term_count,):
            raise RenderInputError(f"{name} frequencies and amplitudes must match")
        if phases.shape != (term_count,) or errors.shape != (term_count + 1,):
            raise RenderInputError(f"{name} phases and RMS errors have invalid lengths")
    for name, values in (
        ("samples_xyz", samples), ("x amplitudes", x_amp), ("x phases", x_phase),
        ("x RMS errors", x_err), ("y amplitudes", y_amp), ("y phases", y_phase),
        ("y RMS errors", y_err),
    ):
        if not np.isfinite(values).all():
            raise RenderInputError(f"{name} must contain only finite values")
    if np.any(x_amp < 0) or np.any(y_amp < 0):
        raise RenderInputError("harmonic amplitudes must be non-negative")

    return RenderData(
        metadata, skeleton, samples, draw_mask,
        x_mean, x_freq, x_amp, x_phase, x_err,
        y_mean, y_freq, y_amp, y_phase, y_err,
    )


_TERM_MILESTONES = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512)


def stage_schedule(term_count: int) -> tuple[int, ...]:
    """Choose readable intermediate term counts and always end at full DFT."""
    if isinstance(term_count, bool) or not isinstance(term_count, int) or term_count < 1:
        raise ValueError("term_count must be a positive integer")
    levels = [level for level in _TERM_MILESTONES if level < term_count]
    levels.append(term_count)
    return tuple(levels)


def _evaluate_axis(mean, frequencies, amplitudes, phases, times, count, *, sine=False):
    if count == 0:
        return np.full(times.shape, mean, dtype=np.float64)
    angles = 2 * np.pi * frequencies[:count, None] * times[None, :] + phases[:count, None]
    basis = np.sin(angles) if sine else np.cos(angles)
    return mean + np.sum(amplitudes[:count, None] * basis, axis=0)


def _numeric_formula(symbol, mean, frequencies, amplitudes, phases, count, *, sine=False):
    """Format the leading active harmonics as a compact, live equation."""
    pieces = [f"{mean:.2f}"]
    shown = min(count, 2)
    function = r"\sin" if sine else r"\cos"
    for frequency, amplitude, phase in zip(
        frequencies[:shown], amplitudes[:shown], phases[:shown]
    ):
        sign = "+" if phase >= 0 else "-"
        pieces.append(
            rf"{amplitude:.3f}{function}(2\pi({int(frequency)})t {sign} {abs(phase):.2f})"
        )
    if count > shown:
        pieces.append(r"\cdots")
    return rf"${symbol}_N(t) = " + " + ".join(pieces) + "$"


def _chain_extent(frequencies, amplitudes, phases) -> float:
    if not len(frequencies):
        return 1.0
    time_count = max(256, min(2048, 4 * len(frequencies)))
    times = np.arange(time_count, dtype=np.float64) / time_count
    vectors = amplitudes[:, None] * np.exp(
        2j * np.pi * frequencies[:, None] * times[None, :] + 1j * phases[:, None]
    )
    starts = np.concatenate(
        (np.zeros((1, time_count), dtype=np.complex128), np.cumsum(vectors, axis=0)[:-1]),
        axis=0,
    )
    return max(float(np.max(np.abs(starts) + amplitudes[:, None])), 1e-12)


def render_video(
    npz_path: str | Path,
    output_path: str | Path,
    *,
    width: int = 1080,
    height: int = 1920,
    fps: int = 30,
    stage_seconds: float = 0.8,
    end_hold_seconds: float = 1.2,
    bitrate: int = 8000,
) -> Path:
    """Render the shared xy diagram as a 9:16 H.264 MP4."""
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (width, height, fps, bitrate)):
        raise ValueError("width, height, fps, and bitrate must be integers")
    if width < 270 or height < 480 or width * 16 != height * 9:
        raise ValueError("video dimensions must be a 9:16 portrait size of at least 270x480")
    if fps < 1 or bitrate < 100:
        raise ValueError("fps and bitrate must be positive")
    if not math.isfinite(stage_seconds) or stage_seconds <= 0:
        raise ValueError("stage_seconds must be finite and positive")
    if not math.isfinite(end_hold_seconds) or end_hold_seconds < 0:
        raise ValueError("end_hold_seconds must be finite and non-negative")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg was not found; install FFmpeg before rendering MP4")
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import font_manager
        import matplotlib.pyplot as plt
        from matplotlib.animation import FFMpegWriter, FuncAnimation
        from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
    except ImportError as exc:
        raise RuntimeError(
            "video rendering needs Matplotlib; install it with "
            "python -m pip install 'fourier-line-art[video]'"
        ) from exc

    data = load_render_data(npz_path)
    levels = stage_schedule(data.term_count)
    output = Path(output_path)
    if output.suffix.lower() != ".mp4":
        raise ValueError("output_path must end in .mp4")
    output.parent.mkdir(parents=True, exist_ok=True)

    image_height, image_width = data.skeleton.shape
    map_width = 3.5
    map_scale = map_width / image_width
    map_height = image_height * map_scale
    frames_per_stage = max(2, round(fps * stage_seconds))
    frame_count = max(data.sample_count * 2, frames_per_stage * 12)
    times = np.linspace(0.0, 1.0, frame_count + 1)
    draw_indices = np.minimum(np.floor(times * data.sample_count).astype(int), data.sample_count - 1)
    draw_flags = data.draw_mask[draw_indices]
    stage_curves = {
        count: (
            _evaluate_axis(data.x_mean, data.x_frequencies, data.x_amplitudes,
                           data.x_phases, times, count),
            _evaluate_axis(data.y_mean, data.y_frequencies, data.y_amplitudes,
                           data.y_sine_phases, times, count, sine=True),
        )
        for count in levels
    }
    diagonal = max(float(np.hypot(image_width, image_height)), 1.0)
    xy_errors = np.sqrt(data.x_rms_errors**2 + data.y_rms_errors**2)
    relative_errors = xy_errors / diagonal
    frames_for_stages = len(levels) * frames_per_stage
    hold_frames = round(fps * end_hold_seconds)
    total_frames = frames_for_stages + hold_frames

    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    font_family = next(
        (name for name in ("Noto Sans CJK SC", "Microsoft YaHei", "PingFang SC", "SimHei")
         if name in available_fonts),
        "DejaVu Sans",
    )
    plt.rcParams.update({
        "font.family": font_family,
        "font.size": 10,
        "axes.unicode_minus": False,
        "figure.facecolor": "#F4F8F8",
        "savefig.facecolor": "#F4F8F8",
    })
    dpi = height / 16
    fig = plt.figure(figsize=(9, 16), dpi=dpi, facecolor="#F4F8F8")
    fig.text(0.5, 0.965, "傅里叶线稿", ha="center", va="center", fontsize=25,
             fontweight="bold", color="#102A43")
    formula_box = FancyBboxPatch(
        (0.065, 0.765), 0.87, 0.185,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        transform=fig.transFigure, linewidth=1.2,
        edgecolor="#9DC8C4", facecolor="#FFFFFF", zorder=-1,
    )
    fig.add_artist(formula_box)
    fig.text(
        0.5, 0.916,
        r"$x_N(t)=x_0+\sum_{j=1}^{N}R_{x,j}\cos(2\pi k_{x,j}t+\phi_{x,j})$",
        ha="center", va="center", fontsize=13, color="#176B78",
    )
    fig.text(
        0.5, 0.848,
        r"$y_N(t)=y_0+\sum_{j=1}^{N}R_{y,j}\sin(2\pi k_{y,j}t+\phi_{y,j})$",
        ha="center", va="center", fontsize=13, color="#176B78",
    )
    x_numeric_formula = fig.text(0.5, 0.882, "", ha="center", va="center",
                                 fontsize=10, color="#176B78")
    y_numeric_formula = fig.text(0.5, 0.814, "", ha="center", va="center",
                                 fontsize=10, color="#176B78")
    formula_state = fig.text(0.5, 0.779, "", ha="center", va="center",
                             fontsize=9, fontweight="bold", color="#164E63")

    main_ax = fig.add_axes((0.055, 0.225, 0.89, 0.535), facecolor="#F4F8F8")
    main_ax.set_axis_off()
    main_ax.set_aspect("equal", adjustable="box")
    main_ax.set_xlim(-2.25, map_width + 0.22)
    x_center = (-1.28, map_height * 0.5)
    y_center = (map_width * 0.5, -1.08)
    main_ax.set_ylim(map_height + 0.24, -2.20)

    rgba = np.zeros((image_height, image_width, 4), dtype=np.float32)
    rgba[..., :3] = (0.18, 0.27, 0.32)
    rgba[..., 3] = data.skeleton.astype(np.float32) * 0.54
    main_ax.imshow(
        rgba, extent=(0, map_width, map_height, 0), origin="upper",
        interpolation="nearest", zorder=1,
    )
    main_ax.add_patch(Rectangle((0, 0), map_width, map_height, fill=False,
                                linewidth=0.9, edgecolor="#A8B9BD", zorder=1))
    (map_trace,) = main_ax.plot([], [], color="#087EA4", linewidth=2.4,
                                solid_capstyle="round", zorder=3)
    map_point = main_ax.scatter([], [], s=24, color="#087EA4", edgecolors="white",
                                linewidths=0.5, zorder=6)
    # These two guides are coordinate projections to the active map point.
    (x_guide,) = main_ax.plot([], [], color="#4B9298", linewidth=1.15,
                              linestyle=(0, (4, 3)), alpha=0.85, zorder=4)
    (y_guide,) = main_ax.plot([], [], color="#4B9298", linewidth=1.15,
                              linestyle=(0, (4, 3)), alpha=0.85, zorder=4)

    x_scale = 0.82 / _chain_extent(data.x_frequencies, data.x_amplitudes, data.x_phases)
    y_scale = 0.82 / _chain_extent(data.y_frequencies, data.y_amplitudes, data.y_sine_phases)

    def add_chain(frequencies, amplitudes, phases, center, scale, output_axis, color):
        vectors, circles = [], []
        for _ in frequencies:
            (vector,) = main_ax.plot([], [], color=color, linewidth=0.78, alpha=0.78, zorder=2)
            circle = Circle(center, 0, fill=False, edgecolor="#77A4A6",
                            linewidth=0.68, alpha=0.42, zorder=2)
            main_ax.add_patch(circle)
            vectors.append(vector)
            circles.append(circle)
        endpoint = main_ax.scatter([], [], s=13, color=color, edgecolors="white",
                                   linewidths=0.4, zorder=5)
        (projection,) = main_ax.plot([], [], color="#67868C", linewidth=0.75, alpha=0.68, zorder=3)
        projected_tip = main_ax.scatter([], [], s=8, color="#E07040", zorder=5)
        return {
            "frequencies": frequencies, "amplitudes": amplitudes, "phases": phases,
            "center": center, "scale": scale, "output_axis": output_axis,
            "vectors": vectors, "circles": circles, "endpoint": endpoint,
            "projection": projection, "projected_tip": projected_tip,
        }

    x_center = (-1.28, map_height * 0.5)
    y_center = (map_width * 0.5, -1.08)
    x_chain = add_chain(data.x_frequencies, data.x_amplitudes, data.x_phases,
                        x_center, x_scale, "x", "#287B8E")
    y_chain = add_chain(data.y_frequencies, data.y_amplitudes, data.y_sine_phases,
                        y_center, y_scale, "y", "#2E8B78")
    main_ax.text(x_center[0], x_center[1] - 0.94, "X 分量", ha="center", va="bottom",
                 fontsize=11, fontweight="bold", color="#287B8E")
    main_ax.text(y_center[0], y_center[1] - 0.94, "Y 分量", ha="center", va="bottom",
                 fontsize=11, fontweight="bold", color="#2E8B78")

    point_text = fig.text(0.91, 0.779, "", ha="right", va="center", fontsize=9, color="#527080")
    fig.text(0.08, 0.19, "相对 RMS 误差", ha="left", va="center", fontsize=11,
             fontweight="bold", color="#183B4E")
    error_readout = fig.text(0.92, 0.19, "", ha="right", va="center", fontsize=11,
                             fontweight="bold", color="#087EA4")
    error_ax = fig.add_axes((0.12, 0.095, 0.76, 0.075), facecolor="#F4F8F8")
    error_ax.set_xscale("log")
    counts = np.arange(1, data.term_count + 1)
    error_values = relative_errors[1:] * 100
    error_ax.plot(counts, error_values, color="#9DC8C4", linewidth=1.8)
    (error_marker,) = error_ax.plot([], [], marker="o", markersize=6,
                                    color="#E07040", linestyle="None", zorder=3)
    error_ax.set_xlim(1, max(2, data.term_count))
    error_ax.set_ylim(0, max(float(np.max(error_values)) * 1.12, 0.1))
    error_ax.grid(axis="y", color="#DCE8E8", linewidth=0.7)
    error_ax.tick_params(axis="both", labelsize=7, colors="#607D8B", length=2)
    ticks = sorted(set((1, max(1, data.term_count // 4),
                        max(1, data.term_count // 2), data.term_count)))
    error_ax.set_xticks(ticks, labels=[str(value) for value in ticks])
    error_ax.set_ylabel("项数", fontsize=7, color="#607D8B")
    for spine in error_ax.spines.values():
        spine.set_visible(False)
    fig.text(0.5, 0.035, "蓝色 = 当前描线　·　灰色 = 原图中心线　·　虚线 = X/Y 坐标投影",
             ha="center", va="center", fontsize=8, color="#667F89")

    def update_chain(chain, t, term_count):
        frequencies = chain["frequencies"]
        amplitudes = chain["amplitudes"]
        phases = chain["phases"]
        active_count = min(term_count, len(frequencies))
        rotated = amplitudes[:active_count] * np.exp(
            2j * np.pi * frequencies[:active_count] * t + 1j * phases[:active_count]
        )
        prefix = np.concatenate(([0j], np.cumsum(rotated)))
        center_x, center_y = chain["center"]
        scale = chain["scale"]
        for index, (vector, circle) in enumerate(zip(chain["vectors"], chain["circles"])):
            if index >= active_count:
                vector.set_visible(False)
                circle.set_visible(False)
                continue
            start, end = prefix[index] * scale, prefix[index + 1] * scale
            vector.set_visible(True)
            vector.set_data([center_x + start.real, center_x + end.real],
                            [center_y + start.imag, center_y + end.imag])
            radius = float(amplitudes[index] * scale)
            circle.center = (center_x + start.real, center_y + start.imag)
            circle.radius = radius
            circle.set_visible(radius >= 0.012)

        tip = prefix[-1] * scale
        tip_xy = (center_x + tip.real, center_y + tip.imag)
        chain["endpoint"].set_offsets([tip_xy])
        projection_xy = (tip_xy[0], center_y) if chain["output_axis"] == "x" else (center_x, tip_xy[1])
        chain["projection"].set_data([tip_xy[0], projection_xy[0]],
                                     [tip_xy[1], projection_xy[1]])
        chain["projected_tip"].set_offsets([projection_xy])

    def update(frame: int):
        if frame >= frames_for_stages:
            stage_index, progress = len(levels) - 1, 1.0
        else:
            stage_index = frame // frames_per_stage
            local_frame = frame % frames_per_stage
            progress = local_frame / max(frames_per_stage - 1, 1)
        term_count = levels[stage_index]
        x_curve, y_curve = stage_curves[term_count]
        visible_end = max(1, int(progress * frame_count) + 1)
        partial_x, partial_y = x_curve[:visible_end] * map_scale, y_curve[:visible_end] * map_scale
        flags = draw_flags[:visible_end]
        map_trace.set_data(np.where(flags, partial_x, np.nan), np.where(flags, partial_y, np.nan))

        time_index = min(visible_end - 1, len(times) - 1)
        point_x, point_y = float(x_curve[time_index] * map_scale), float(y_curve[time_index] * map_scale)
        map_point.set_offsets([[point_x, point_y]])
        x_guide.set_data([0, point_x], [point_y, point_y])
        y_guide.set_data([point_x, point_x], [0, point_y])
        t = float(times[time_index])
        update_chain(x_chain, t, term_count)
        update_chain(y_chain, t, term_count)

        x_norm = float(x_curve[time_index] / max(image_width - 1, 1))
        y_norm = float(y_curve[time_index] / max(image_height - 1, 1))
        x_numeric_formula.set_text(_numeric_formula(
            "x", data.x_mean, data.x_frequencies, data.x_amplitudes,
            data.x_phases, term_count,
        ))
        y_numeric_formula.set_text(_numeric_formula(
            "y", data.y_mean, data.y_frequencies, data.y_amplitudes,
            data.y_sine_phases, term_count, sine=True,
        ))
        relative_error = float(relative_errors[term_count] * 100)
        absolute_error = float(xy_errors[term_count])
        formula_state.set_text(f"N = {term_count} / {data.term_count} 项　·　X、Y 分量分别排序")
        point_text.set_text(f"实时笔尖 ({x_norm:.2f}, {y_norm:.2f})")
        error_readout.set_text(f"{relative_error:.2f}%  ({absolute_error:.2f} px)")
        error_marker.set_data([term_count], [relative_error])
        return [map_trace, map_point, x_guide, y_guide, formula_state, point_text,
                x_numeric_formula, y_numeric_formula,
                error_readout, error_marker,
                *x_chain["vectors"], *x_chain["circles"], x_chain["endpoint"],
                x_chain["projection"], x_chain["projected_tip"],
                *y_chain["vectors"], *y_chain["circles"], y_chain["endpoint"],
                y_chain["projection"], y_chain["projected_tip"]]

    animation = FuncAnimation(
        fig, update, frames=total_frames, interval=1000 / fps,
        blit=False, repeat=False, cache_frame_data=False,
    )
    writer = FFMpegWriter(
        fps=fps, codec="libx264", bitrate=bitrate,
        metadata={"title": "Fourier Line Art", "artist": "fourier-line-art"},
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    try:
        animation.save(str(output), writer=writer, dpi=dpi)
    finally:
        plt.close(fig)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fourier-line-art-render",
        description="Render a solver .npz as one xy-controlled 9:16 MP4 diagram.",
    )
    parser.add_argument("result", type=Path, help=".npz produced by fourier-line-art")
    parser.add_argument("--output", type=Path, default=Path("fourier-portrait.mp4"))
    parser.add_argument("--width", type=int, default=1080)
    parser.add_argument("--height", type=int, default=1920)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--stage-seconds", type=float, default=0.8)
    parser.add_argument("--end-hold-seconds", type=float, default=1.2)
    parser.add_argument("--bitrate", type=int, default=8000, help="H.264 video bitrate in kbit/s")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output = render_video(
            args.result, args.output,
            width=args.width, height=args.height, fps=args.fps,
            stage_seconds=args.stage_seconds,
            end_hold_seconds=args.end_hold_seconds, bitrate=args.bitrate,
        )
    except (OSError, RenderInputError, RuntimeError, ValueError) as exc:
        print(f"fourier-line-art-render: error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"output": str(output), "width": args.width,
                      "height": args.height, "fps": args.fps}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
