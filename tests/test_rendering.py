import json
import shutil

import numpy as np
import pytest
from PIL import Image

from fourier_line_art.pipeline import solve_image
from fourier_line_art.rendering import (
    RenderInputError,
    _chain_extent,
    _coordinate_guides,
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
    np.testing.assert_array_equal(loaded.x_frequencies, loaded.y_frequencies)
    np.testing.assert_array_equal(loaded.x_frequencies, np.arange(1, loaded.term_count + 1))

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
    map_width = 3.5
    map_scale = map_width / data.skeleton.shape[1]
    map_height = data.skeleton.shape[0] * map_scale

    scene = _scene_bounds(data, map_scale, map_width, map_height)
    x_extent = _chain_extent(data.x_frequencies, data.x_amplitudes, data.x_phases) * map_scale
    y_extent = _chain_extent(data.y_frequencies, data.y_amplitudes, data.y_sine_phases) * map_scale

    assert scene[4] == pytest.approx(data.x_mean * map_scale)
    assert scene[5] == pytest.approx(-x_extent - 0.18)
    assert scene[6] == pytest.approx(-y_extent - 0.18)
    assert scene[7] == pytest.approx(data.y_mean * map_scale)
    assert scene[8] == pytest.approx(x_extent)
    assert scene[9] == pytest.approx(y_extent)
    assert scene[0] < scene[6] - y_extent
    assert scene[2] < scene[5] - x_extent
    assert scene[0] < 0 < map_width < scene[1]
    assert scene[2] < 0 < map_height < scene[3]


def test_coordinate_guides_are_perpendicular_and_join_wheels_to_map(tmp_path):
    data = load_render_data(_make_solver_result(tmp_path))
    map_scale = 3.5 / data.skeleton.shape[1]
    _, _, _, _, x_center_x, x_center_y, y_center_x, y_center_y, _, _ = _scene_bounds(
        data, map_scale, 3.5, 3.5
    )
    map_point = (x_center_x + 0.31, y_center_y + 0.42)
    x_tip = (map_point[0], x_center_y - 0.17)
    y_tip = (y_center_x + 0.12, map_point[1])

    guides = _coordinate_guides(x_tip, y_tip, map_point)

    x_guide_x, x_guide_y = guides["x"]
    assert x_guide_x[0] == x_guide_x[1]  # vertical to the horizontal X axis
    assert (x_guide_x[0], x_guide_y[0]) == pytest.approx(x_tip)
    assert (x_guide_x[1], x_guide_y[1]) == pytest.approx(map_point)

    y_guide_x, y_guide_y = guides["y"]
    assert y_guide_y[0] == y_guide_y[1]  # horizontal to the vertical Y axis
    assert (y_guide_x[0], y_guide_y[0]) == pytest.approx(y_tip)
    assert (y_guide_x[1], y_guide_y[1]) == pytest.approx(map_point)

    with pytest.raises(ValueError, match="do not share an x coordinate"):
        _coordinate_guides((x_tip[0] + 0.1, x_tip[1]), y_tip, map_point)
    with pytest.raises(ValueError, match="do not share a y coordinate"):
        _coordinate_guides(x_tip, (y_tip[0], y_tip[1] - 0.1), map_point)


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
