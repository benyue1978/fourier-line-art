import json
import shutil

import numpy as np
import pytest
from PIL import Image

from fourier_line_art.pipeline import solve_image
from fourier_line_art.rendering import (
    RenderInputError,
    _chain_extent,
    _evaluate_axis,
    _scene_bounds,
    load_render_data,
    render_video,
    stage_schedule,
)


@pytest.mark.parametrize(
    ("term_count", "expected"),
    [
        (1, (1,)),
        (7, (1, 2, 3, 4, 6, 7)),
        (16, (1, 2, 3, 4, 6, 8, 12, 16)),
    ],
)
def test_stage_schedule_includes_full_precision_once(term_count, expected):
    assert stage_schedule(term_count) == expected


@pytest.mark.parametrize("term_count", [0, -1, True, 1.5])
def test_stage_schedule_rejects_invalid_term_count(term_count):
    with pytest.raises(ValueError):
        stage_schedule(term_count)


def _make_solver_result(tmp_path):
    image = np.full((64, 64), 255, dtype=np.uint8)
    image[8:56, 8] = 0
    image[8:56, 55] = 0
    image[8, 8:56] = 0
    image[55, 8:56] = 0
    image[18:24, 21] = 0
    image[18:24, 43] = 0
    image[42:45, 24:41] = 0
    return solve_image(image, sample_count=32, connectivity=8).save_npz(
        tmp_path / "solution.npz"
    )


def test_load_render_data_reads_npz_and_rejects_invalid_schema(tmp_path):
    result_path = _make_solver_result(tmp_path)
    loaded = load_render_data(result_path)
    assert loaded.term_count == 16
    assert loaded.sample_count == 32
    assert loaded.skeleton.ndim == 2

    missing_fields = tmp_path / "missing-fields.npz"
    np.savez(missing_fields, metadata_json=np.asarray(json.dumps({"schema_version": 1})))
    with pytest.raises(RenderInputError, match="missing renderer fields"):
        load_render_data(missing_fields)

    malformed = tmp_path / "malformed.npz"
    np.savez(
        malformed,
        metadata_json=np.asarray(json.dumps({"schema_version": 999})),
        skeleton=np.ones((2, 2), dtype=bool),
        samples_xyz=np.zeros((2, 3)),
        draw_mask=np.ones(2, dtype=bool),
        xy_frequencies=np.zeros(2, dtype=int),
        xy_coefficients=np.zeros(2, dtype=complex),
        xy_rms_errors=np.zeros(3),
        relative_xy_rms_errors=np.zeros(3),
    )
    with pytest.raises(RenderInputError, match="schema version"):
        load_render_data(malformed)


def test_scene_bounds_scale_both_wheels_in_source_pixel_units(tmp_path):
    data = load_render_data(_make_solver_result(tmp_path))
    levels = stage_schedule(data.term_count)
    times = np.linspace(0.0, 1.0, 257)
    curves = {
        count: (
            _evaluate_axis(data.x_mean, data.x_frequencies, data.x_amplitudes,
                           data.x_phases, times, count),
            _evaluate_axis(data.y_mean, data.y_frequencies, data.y_amplitudes,
                           data.y_sine_phases, times, count, sine=True),
        )
        for count in levels
    }
    map_width = 3.5
    map_scale = map_width / data.skeleton.shape[1]
    map_height = data.skeleton.shape[0] * map_scale

    scene = _scene_bounds(data, levels, curves, times, map_scale, map_width, map_height)

    assert scene[4] == pytest.approx(
        -_chain_extent(data.x_frequencies, data.x_amplitudes, data.x_phases) * map_scale - 0.18
    )
    assert scene[5] == pytest.approx(
        -_chain_extent(data.y_frequencies, data.y_amplitudes, data.y_sine_phases) * map_scale - 0.18
    )
    assert scene[0] < 0 < map_width < scene[1]
    assert scene[2] < 0 < map_height < scene[3]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is needed for MP4 output")
def test_render_video_writes_a_portrait_mp4(tmp_path):
    pytest.importorskip("matplotlib")
    result_path = _make_solver_result(tmp_path)
    output = render_video(
        result_path,
        tmp_path / "preview.mp4",
        width=270,
        height=480,
        fps=2,
        stage_seconds=0.5,
        end_hold_seconds=0,
        bitrate=500,
    )

    assert output.exists()
    assert output.stat().st_size > 0
