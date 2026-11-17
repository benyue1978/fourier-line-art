import json

import numpy as np
import pytest
from PIL import Image

from fourier_line_art.cli import main
from fourier_line_art.pipeline import EmptyDrawingError, solve_image


def test_pipeline_solves_multiple_strokes_and_saves_renderer_neutral_npz(tmp_path):
    image = np.full((12, 16), 255, dtype=np.uint8)
    image[2, 2:6] = 0
    image[8, 10:14] = 0

    result = solve_image(image, connectivity=4, sample_count=64, lift_height=0.5)

    assert result.solver_name == "rural_postman_exact_state_dijkstra"
    assert result.route.is_closed
    assert result.analysis.component_count == 2
    assert result.samples.draw.any()
    assert (~result.samples.draw).any()
    assert np.all(np.diff(result.xy_rms_errors) <= 1e-12)
    assert np.all(np.diff(result.height_rms_errors) <= 1e-12)
    assert result.xy_rms_errors[-1] == pytest.approx(0, abs=1e-12)

    output = result.save_npz(tmp_path / "solution.npz")
    with np.load(output, allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata_json"]))
        assert metadata["schema_version"] == 2
        assert metadata["solver"] == result.solver_name
        assert saved["samples_xyz"].shape == (64, 3)
        assert saved["xy_coefficients"].shape == (64,)
        assert saved["height_coefficients"].shape == (64,)
        assert saved["draw_mask"].shape == (64,)
        assert saved["x_amplitudes"].shape == (32,)
        assert saved["y_amplitudes"].shape == (32,)
        assert saved["x_rms_errors"].shape == (33,)
        assert saved["y_rms_errors"].shape == (33,)


def test_pipeline_uses_cpp_for_one_connected_drawing_component():
    image = np.full((9, 9), 255, dtype=np.uint8)
    image[4, 2:7] = 0

    result = solve_image(image, connectivity=4, sample_count=32)

    assert result.solver_name == "chinese_postman"
    assert result.analysis.component_count == 1
    assert result.route.is_closed


def test_pipeline_switches_to_mst_cpp_rpp_heuristic_when_exact_limit_is_hit():
    image = np.full((12, 16), 255, dtype=np.uint8)
    image[2, 2:6] = 0
    image[8, 10:14] = 0

    result = solve_image(
        image,
        connectivity=4,
        sample_count=64,
        max_required_edges=1,
    )

    assert result.solver_name == "rural_postman_frederickson_mst_cpp_heuristic"
    assert result.route.is_closed
    assert result.analysis.component_count == 2


def test_pipeline_rejects_empty_line_art():
    with pytest.raises(EmptyDrawingError):
        solve_image(np.full((5, 5), 255, dtype=np.uint8))


def test_cli_reads_png_and_writes_npz(tmp_path, capsys):
    image = np.full((8, 8), 255, dtype=np.uint8)
    image[3, 2:6] = 0
    input_path = tmp_path / "line.png"
    output_path = tmp_path / "solution.npz"
    Image.fromarray(image).save(input_path)

    assert main(
        [
            str(input_path),
            "--output",
            str(output_path),
            "--connectivity",
            "4",
            "--samples",
            "32",
        ]
    ) == 0
    assert output_path.exists()
    assert '"solver": "chinese_postman"' in capsys.readouterr().out

