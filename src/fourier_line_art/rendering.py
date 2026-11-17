"""Render solver results as portrait Fourier epicycle videos."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np


class RenderInputError(ValueError):
    """A saved solver result cannot be used as an animation input."""


@dataclass(frozen=True, slots=True)
class RenderData:
    """Validated, renderer-neutral arrays loaded from a solver NPZ."""

    metadata: dict
    skeleton: np.ndarray
    samples_xyz: np.ndarray
    draw_mask: np.ndarray
    xy_frequencies: np.ndarray
    xy_coefficients: np.ndarray
    xy_rms_errors: np.ndarray
    relative_xy_rms_errors: np.ndarray

    @property
    def term_count(self) -> int:
        return int(len(self.xy_coefficients))

    @property
    def sample_count(self) -> int:
        return int(len(self.samples_xyz))


def load_render_data(npz_path: str | Path) -> RenderData:
    """Load and validate the arrays required for a video render."""
    path = Path(npz_path)
    try:
        with np.load(path, allow_pickle=False) as archive:
            required = {
                "metadata_json",
                "skeleton",
                "samples_xyz",
                "draw_mask",
                "xy_frequencies",
                "xy_coefficients",
                "xy_rms_errors",
                "relative_xy_rms_errors",
            }
            missing = required.difference(archive.files)
            if missing:
                raise RenderInputError(
                    "NPZ is missing renderer fields: " + ", ".join(sorted(missing))
                )
            try:
                metadata = json.loads(str(archive["metadata_json"]))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise RenderInputError("NPZ metadata_json is invalid") from exc
            if metadata.get("schema_version") != 1:
                raise RenderInputError(
                    f"unsupported result schema version: {metadata.get('schema_version')!r}"
                )
            data = RenderData(
                metadata=metadata,
                skeleton=np.array(archive["skeleton"], dtype=bool, copy=True),
                samples_xyz=np.array(archive["samples_xyz"], dtype=np.float64, copy=True),
                draw_mask=np.array(archive["draw_mask"], dtype=bool, copy=True),
                xy_frequencies=np.array(archive["xy_frequencies"], dtype=np.int64, copy=True),
                xy_coefficients=np.array(archive["xy_coefficients"], dtype=np.complex128, copy=True),
                xy_rms_errors=np.array(archive["xy_rms_errors"], dtype=np.float64, copy=True),
                relative_xy_rms_errors=np.array(
                    archive["relative_xy_rms_errors"], dtype=np.float64, copy=True
                ),
            )
    except RenderInputError:
        raise
    except (OSError, ValueError, KeyError) as exc:
        raise RenderInputError(f"could not read solver result {path}: {exc}") from exc

    if data.skeleton.ndim != 2 or not data.skeleton.any():
        raise RenderInputError("saved skeleton must be a non-empty 2D mask")
    if data.samples_xyz.ndim != 2 or data.samples_xyz.shape[1] != 3 or data.sample_count < 2:
        raise RenderInputError("samples_xyz must have shape (M, 3) with M >= 2")
    if data.draw_mask.shape != (data.sample_count,):
        raise RenderInputError("draw_mask must have one value per route sample")
    if data.xy_coefficients.ndim != 1 or data.term_count != data.sample_count:
        raise RenderInputError("xy_coefficients must have one Fourier term per route sample")
    if data.xy_frequencies.shape != (data.term_count,):
        raise RenderInputError("xy_frequencies must match xy_coefficients")
    expected_error_shape = (data.term_count + 1,)
    if data.xy_rms_errors.shape != expected_error_shape:
        raise RenderInputError("xy_rms_errors must include term counts 0 through M")
    if data.relative_xy_rms_errors.shape != expected_error_shape:
        raise RenderInputError("relative_xy_rms_errors must include term counts 0 through M")
    for name, values in (
        ("samples_xyz", data.samples_xyz),
        ("xy_coefficients", data.xy_coefficients),
        ("xy_rms_errors", data.xy_rms_errors),
        ("relative_xy_rms_errors", data.relative_xy_rms_errors),
    ):
        if not np.isfinite(values).all():
            raise RenderInputError(f"{name} must contain only finite values")
    return data


def stage_schedule(term_count: int) -> tuple[int, ...]:
    """Return progressively denser, unique term counts, ending at full DFT."""
    if isinstance(term_count, bool) or not isinstance(term_count, int) or term_count < 1:
        raise ValueError("term_count must be a positive integer")
    levels = []
    value = 1
    while value < term_count:
        levels.append(value)
        value *= 2
    levels.append(term_count)
    return tuple(levels)


def _curve_at_terms(
    coefficients: np.ndarray,
    frequencies: np.ndarray,
    times: np.ndarray,
    term_count: int,
) -> np.ndarray:
    basis = np.exp(2j * np.pi * frequencies[:term_count, None] * times[None, :])
    return np.sum(coefficients[:term_count, None] * basis, axis=0)


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
    """Render a 9:16 H.264 MP4 showing route convergence and epicycles."""
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
        import matplotlib.pyplot as plt
        from matplotlib.animation import FFMpegWriter, FuncAnimation
        from matplotlib.patches import Circle, FancyBboxPatch
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

    # A smooth trace is sampled more densely than the original route samples.
    frames_per_stage = max(2, round(fps * stage_seconds))
    frames_per_period = max(data.sample_count * 2, frames_per_stage * 12)
    times = np.linspace(0.0, 1.0, frames_per_period + 1)
    image_height, image_width = data.skeleton.shape

    plt.rcParams.update(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 10,
            "axes.unicode_minus": False,
            "figure.facecolor": "#F4F8F8",
            "savefig.facecolor": "#F4F8F8",
        }
    )
    dpi = height / 16
    fig = plt.figure(figsize=(9, 16), dpi=dpi, facecolor="#F4F8F8")

    fig.text(0.5, 0.965, "傅里叶线稿", ha="center", va="center", fontsize=25,
             fontweight="bold", color="#102A43")
    fig.text(0.5, 0.94, "从轮廓路径到旋转圆链", ha="center", va="center", fontsize=11,
             color="#527080")

    formula_box = FancyBboxPatch(
        (0.07, 0.795), 0.86, 0.12,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        transform=fig.transFigure, linewidth=1.2,
        edgecolor="#9DC8C4", facecolor="#FFFFFF", zorder=-1,
    )
    fig.add_artist(formula_box)
    fig.text(0.5, 0.878, r"$z_N(t)=\sum_{j=0}^{N-1} c_j\,e^{2\pi i k_j t}$",
             ha="center", va="center", fontsize=16, color="#176B78")
    fig.text(0.5, 0.843,
             r"$c_j=\frac{1}{M}\sum_{m=0}^{M-1} z_m\,e^{-2\pi i k_j m/M}$",
             ha="center", va="center", fontsize=12, color="#3C6873")
    formula_state = fig.text(0.5, 0.812, "N = 1 / M", ha="center", va="center",
                             fontsize=11, fontweight="bold", color="#164E63")

    fig.text(0.09, 0.765, "原图与当前近似路径", ha="left", va="center", fontsize=14,
             fontweight="bold", color="#183B4E")
    main_ax = fig.add_axes((0.08, 0.445, 0.84, 0.295), facecolor="#FFFFFF")
    main_ax.set_axis_off()
    main_ax.set_aspect("equal", adjustable="box")
    main_ax.set_xlim(-0.5, image_width - 0.5)
    main_ax.set_ylim(image_height - 0.5, -0.5)
    rgba = np.zeros((image_height, image_width, 4), dtype=np.float32)
    rgba[..., :3] = (0.18, 0.27, 0.32)
    rgba[..., 3] = data.skeleton.astype(np.float32) * 0.58
    main_ax.imshow(
        rgba,
        extent=(-0.5, image_width - 0.5, image_height - 0.5, -0.5),
        origin="upper",
        interpolation="nearest",
        zorder=1,
    )
    (draw_trace,) = main_ax.plot([], [], color="#087EA4", linewidth=2.3, zorder=3,
                                 solid_capstyle="round")
    (lift_trace,) = main_ax.plot([], [], color="#E49A45", linewidth=1.7, zorder=2,
                                 linestyle=(0, (3, 3)))

    fig.text(0.09, 0.412, "旋转圆链：端点描出同一条近似路径", ha="left", va="center",
             fontsize=14, fontweight="bold", color="#183B4E")
    epi_ax = fig.add_axes((0.08, 0.15, 0.84, 0.235), facecolor="#FFFFFF")
    epi_ax.set_axis_off()
    epi_ax.set_aspect("equal", adjustable="box")
    epi_ax.set_xlim(-1.62, 1.62)
    epi_ax.set_ylim(1.62, -1.62)
    (epi_trace,) = epi_ax.plot([], [], color="#087EA4", linewidth=1.8, alpha=0.9,
                               zorder=2, solid_capstyle="round")
    max_visible_terms = min(data.term_count, 24)
    vectors = []
    circles = []
    for index in range(max_visible_terms):
        (vector,) = epi_ax.plot([], [], color="#4F8490", linewidth=0.9,
                                alpha=0.75, zorder=3)
        circle = Circle((0, 0), 0, fill=False, edgecolor="#7FA9AB",
                        linewidth=0.72, alpha=0.38, zorder=1)
        epi_ax.add_patch(circle)
        vectors.append(vector)
        circles.append(circle)
    endpoint = epi_ax.scatter([], [], s=19, color="#E07040", zorder=4, edgecolors="white",
                              linewidths=0.4)
    (remainder_vector,) = epi_ax.plot([], [], color="#E07040", linewidth=1.4,
                                     linestyle=(0, (2, 2)), alpha=0.9, zorder=3)

    current_state = fig.text(0.5, 0.393, "", ha="center", va="center", fontsize=10,
                             color="#476675")
    fig.text(0.09, 0.113, "相对 RMS 误差", ha="left", va="center", fontsize=11,
             fontweight="bold", color="#183B4E")
    error_readout = fig.text(0.91, 0.113, "", ha="right", va="center", fontsize=11,
                             fontweight="bold", color="#087EA4")
    error_ax = fig.add_axes((0.12, 0.048, 0.76, 0.055), facecolor="#F4F8F8")
    error_ax.set_xscale("log")
    counts = np.arange(1, data.term_count + 1)
    error_values = data.relative_xy_rms_errors[1:] * 100
    error_ax.plot(counts, error_values, color="#9DC8C4", linewidth=1.8)
    (error_marker,) = error_ax.plot([], [], marker="o", markersize=6,
                                    color="#E07040", linestyle="None", zorder=3)
    error_ax.set_xlim(1, max(2, data.term_count))
    upper_error = max(float(np.max(error_values)) * 1.12, 0.1)
    error_ax.set_ylim(0, upper_error)
    error_ax.grid(axis="y", color="#DCE8E8", linewidth=0.7)
    error_ax.tick_params(axis="both", labelsize=7, colors="#607D8B", length=2)
    ticks = sorted(set((1, max(1, data.term_count // 4), max(1, data.term_count // 2), data.term_count)))
    error_ax.set_xticks(ticks, labels=[str(value) for value in ticks])
    error_ax.set_ylabel("%", rotation=0, labelpad=6, fontsize=7, color="#607D8B")
    for spine in error_ax.spines.values():
        spine.set_visible(False)
    fig.text(0.5, 0.018, "蓝线 = 傅里叶重建　 ·　 灰线 = 原始中心线　 ·　 橙虚线 = 提笔连接或剩余项",
             ha="center", va="center", fontsize=8, color="#667F89")

    coefficient_center = np.mean(data.samples_xyz[:, 0] + 1j * data.samples_xyz[:, 1])
    draw_time_indices = np.minimum(
        np.floor(times * data.sample_count).astype(np.int64), data.sample_count - 1
    )
    draw_flags = data.draw_mask[draw_time_indices]
    stage_curves = {
        term_count: _curve_at_terms(
            data.xy_coefficients, data.xy_frequencies, times, term_count
        )
        for term_count in levels
    }
    # Size the epicycle panel from its visible prefix and actual endpoint paths.
    # The omitted high-frequency tail is drawn as one dashed resultant vector.
    display_extent = 0.0
    for term_count, curve in stage_curves.items():
        display_extent = max(
            display_extent,
            float(np.max(np.abs(curve - coefficient_center))),
        )
        shown_count = min(term_count, max_visible_terms)
        if shown_count:
            shown_coefficients = data.xy_coefficients[:shown_count]
            shown_frequencies = data.xy_frequencies[:shown_count]
            constant = np.sum(shown_coefficients[shown_frequencies == 0])
            rotating = np.where(shown_frequencies == 0, 0j, shown_coefficients)
            rotations = rotating[:, None] * np.exp(
                2j * np.pi * shown_frequencies[:, None] * times[None, :]
            )
            prefix = constant + np.cumsum(rotations, axis=0)
            display_extent = max(
                display_extent,
                float(np.max(np.abs(prefix - coefficient_center))),
            )
    display_scale = 1.42 / max(display_extent, 1e-12)
    frames_for_stages = len(levels) * frames_per_stage
    hold_frames = round(fps * end_hold_seconds)
    total_frames = frames_for_stages + hold_frames

    def update(frame: int):
        if frame >= frames_for_stages:
            stage_index = len(levels) - 1
            progress = 1.0
        else:
            stage_index = frame // frames_per_stage
            local_frame = frame % frames_per_stage
            progress = local_frame / max(frames_per_stage - 1, 1)
        term_count = levels[stage_index]
        curve = stage_curves[term_count]
        visible_end = max(1, int(progress * (len(times) - 1)) + 1)
        partial = curve[:visible_end]
        flags = draw_flags[:visible_end]
        draw_x = np.where(flags, partial.real, np.nan)
        draw_y = np.where(flags, partial.imag, np.nan)
        lift_x = np.where(~flags, partial.real, np.nan)
        lift_y = np.where(~flags, partial.imag, np.nan)
        draw_trace.set_data(draw_x, draw_y)
        lift_trace.set_data(lift_x, lift_y)

        displayed_path = (partial - coefficient_center) * display_scale
        epi_trace.set_data(displayed_path.real, -displayed_path.imag)

        phase = progress
        term_coefficients = data.xy_coefficients[:term_count]
        term_frequencies = data.xy_frequencies[:term_count]
        constant = np.sum(term_coefficients[term_frequencies == 0])
        rotating_coefficients = np.where(term_frequencies == 0, 0j, term_coefficients)
        rotating_vectors = rotating_coefficients * np.exp(2j * np.pi * term_frequencies * phase)
        prefix = np.concatenate(([constant], constant + np.cumsum(rotating_vectors)))
        shown_count = min(term_count, max_visible_terms)
        for index in range(max_visible_terms):
            active = index < shown_count
            rotating = active and int(data.xy_frequencies[index]) != 0
            vectors[index].set_visible(rotating)
            circles[index].set_visible(active and int(data.xy_frequencies[index]) != 0)
            if not active:
                continue
            start = (prefix[index] - coefficient_center) * display_scale
            end = (prefix[index + 1] - coefficient_center) * display_scale
            vectors[index].set_data([start.real, end.real], [-start.imag, -end.imag])
            circles[index].center = (start.real, -start.imag)
            circles[index].radius = float(abs(data.xy_coefficients[index]) * display_scale)
        shown_sum = (prefix[shown_count] - coefficient_center) * display_scale
        final_sum = (prefix[-1] - coefficient_center) * display_scale
        if term_count > shown_count:
            remainder_vector.set_data([shown_sum.real, final_sum.real],
                                      [-shown_sum.imag, -final_sum.imag])
            remainder_vector.set_visible(True)
        else:
            remainder_vector.set_data([], [])
            remainder_vector.set_visible(False)
        endpoint.set_offsets([[final_sum.real, -final_sum.imag]])

        relative_error = float(data.relative_xy_rms_errors[term_count] * 100)
        absolute_error = float(data.xy_rms_errors[term_count])
        formula_state.set_text(f"N = {term_count} / {data.term_count} 项")
        if term_count > shown_count:
            current_state.set_text(
                f"显示前 {shown_count} 个圆　·　其余 {term_count - shown_count} 项合成橙色虚线"
            )
        else:
            current_state.set_text(f"当前近似：{term_count} 项　·　每一圈从 t = 0 到 1")
        error_readout.set_text(f"{relative_error:.2f}%  ({absolute_error:.2f} px)")
        error_marker.set_data([term_count], [relative_error])
        return [draw_trace, lift_trace, epi_trace, endpoint, remainder_vector, formula_state,
                current_state, error_readout, error_marker, *vectors, *circles]

    animation = FuncAnimation(
        fig, update, frames=total_frames, interval=1000 / fps,
        blit=False, repeat=False, cache_frame_data=False,
    )
    writer = FFMpegWriter(
        fps=fps,
        codec="libx264",
        bitrate=bitrate,
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
        description="Render a solver .npz as a 9:16 Fourier drawing MP4.",
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
            args.result,
            args.output,
            width=args.width,
            height=args.height,
            fps=args.fps,
            stage_seconds=args.stage_seconds,
            end_hold_seconds=args.end_hold_seconds,
            bitrate=args.bitrate,
        )
    except (OSError, RenderInputError, RuntimeError, ValueError) as exc:
        print(f"fourier-line-art-render: error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"output": str(output), "width": args.width, "height": args.height,
                      "fps": args.fps}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
