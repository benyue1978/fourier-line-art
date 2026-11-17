import numpy as np
import pytest

from fourier_line_art.fourier import (
    FourierSeries,
    FourierValidationError,
    RealFourierSeries,
)


def test_constant_signal_reconstructs_with_one_term():
    samples = np.full(32, 2.5 + 1.25j)
    series = FourierSeries.from_samples(samples)

    actual = series.reconstruct(term_count=1)

    np.testing.assert_allclose(actual, samples, atol=1e-12)


def test_full_dft_reconstructs_arbitrary_complex_samples():
    samples = np.array([0 + 1j, 2 - 3j, -1 + 0.5j, 4 + 2j, 1 - 2j])
    series = FourierSeries.from_samples(samples)

    np.testing.assert_allclose(series.reconstruct(len(samples)), samples, atol=1e-12)
    np.testing.assert_allclose(
        series.evaluate(np.arange(len(samples)) / len(samples), len(samples)),
        samples,
        atol=1e-12,
    )


def test_rms_error_does_not_increase_when_terms_are_added():
    t = np.arange(128) / 128
    samples = (
        np.exp(2j * np.pi * 3 * t)
        + 0.4 * np.exp(-2j * np.pi * 11 * t)
        + 0.1 * np.exp(2j * np.pi * 27 * t)
    )
    series = FourierSeries.from_samples(samples)
    errors = [series.rms_error(k) for k in range(len(samples) + 1)]

    assert all(later <= earlier + 1e-12 for earlier, later in zip(errors, errors[1:]))
    assert errors[-1] == pytest.approx(0, abs=1e-12)


@pytest.mark.parametrize("term_count", [-1, 4])
def test_reconstruction_rejects_out_of_range_term_count(term_count):
    series = FourierSeries.from_samples(np.array([1 + 0j, 2 + 1j, 3 - 1j]))

    with pytest.raises(FourierValidationError):
        series.reconstruct(term_count)


def test_fourier_rejects_empty_and_non_finite_samples():
    with pytest.raises(FourierValidationError):
        FourierSeries.from_samples(np.array([], dtype=complex))
    with pytest.raises(FourierValidationError):
        FourierSeries.from_samples(np.array([1 + 0j, complex(float("nan"), 0)]))


def test_fourier_rejects_boolean_term_count_and_non_finite_time():
    series = FourierSeries.from_samples(np.array([1 + 0j, 2 + 1j]))

    with pytest.raises(FourierValidationError):
        series.evaluate(0.5, True)
    with pytest.raises(FourierValidationError):
        series.evaluate(float("nan"), 1)


def test_vector_signal_parseval_errors_match_direct_sample_errors():
    samples = np.array(
        [
            [1 + 2j, 0.5 - 1j],
            [2 - 1j, -1 + 0.25j],
            [-0.5 + 1j, 3 + 2j],
            [4 + 0j, 1 - 2j],
        ]
    )
    series = FourierSeries.from_samples(samples)

    np.testing.assert_allclose(
        series.rms_errors(),
        [series.rms_error(count) for count in range(len(samples) + 1)],
        atol=1e-12,
    )


def test_real_fourier_series_separates_conjugate_pairs_into_cosines():
    times = np.arange(128) / 128
    samples = (
        4.25
        + 2.0 * np.cos(2 * np.pi * 3 * times + 0.4)
        + 0.75 * np.cos(2 * np.pi * 21 * times - 0.9)
        + 0.3 * (-1.0) ** np.arange(128)
    )
    series = RealFourierSeries.from_samples(samples)

    assert series.mean == pytest.approx(4.25)
    assert series.term_count == 64
    np.testing.assert_allclose(series.reconstruct(series.term_count), samples, atol=1e-12)
    np.testing.assert_allclose(
        series.ordered_amplitudes[:2], [2.0, 0.75], atol=1e-12
    )


def test_real_fourier_errors_match_direct_rms_and_never_increase():
    times = np.arange(63) / 63
    samples = 1.3 + np.cos(2 * np.pi * 2 * times + 0.2) + 0.2 * np.sin(2 * np.pi * 8 * times)
    series = RealFourierSeries.from_samples(samples)
    errors = series.rms_errors()

    assert np.all(np.diff(errors) <= 1e-12)
    np.testing.assert_allclose(
        errors,
        [series.rms_error(count) for count in range(series.term_count + 1)],
        atol=1e-12,
    )

    frequency_errors = series.frequency_ordered_rms_errors()
    direct_errors = []
    for count in range(series.term_count + 1):
        if count == 0:
            reconstructed = np.full(times.shape, series.mean)
        else:
            angles = (
                2 * np.pi * series.frequency_ordered_frequencies[:count, None]
                * times[None, :]
                + series.frequency_ordered_phases[:count, None]
            )
            reconstructed = series.mean + np.sum(
                series.frequency_ordered_amplitudes[:count, None] * np.cos(angles),
                axis=0,
            )
        direct_errors.append(np.sqrt(np.mean((samples - reconstructed) ** 2)))
    np.testing.assert_allclose(frequency_errors, direct_errors, atol=1e-12)


def test_first_paired_frequency_reconstructs_an_ellipse():
    times = np.arange(256) / 256
    phase = 0.37
    x_mean, y_mean = 31.0, 27.0
    x_radius, y_radius = 17.0, 11.0
    x_samples = x_mean + x_radius * np.cos(2 * np.pi * times + phase)
    y_samples = y_mean + y_radius * np.sin(2 * np.pi * times + phase)
    x_series = RealFourierSeries.from_samples(x_samples)
    y_series = RealFourierSeries.from_samples(y_samples)

    np.testing.assert_array_equal(
        x_series.frequency_ordered_frequencies,
        y_series.frequency_ordered_frequencies,
    )
    assert x_series.frequency_ordered_frequencies[0] == 1
    x_first = (
        x_series.mean
        + x_series.frequency_ordered_amplitudes[0]
        * np.cos(2 * np.pi * times + x_series.frequency_ordered_phases[0])
    )
    y_first = (
        y_series.mean
        + y_series.frequency_ordered_amplitudes[0]
        * np.sin(
            2 * np.pi * times
            + y_series.frequency_ordered_phases[0]
            + np.pi / 2
        )
    )
    normalized_x = (x_first - x_mean) / x_radius
    normalized_y = (y_first - y_mean) / y_radius
    np.testing.assert_allclose(normalized_x**2 + normalized_y**2, 1.0, atol=1e-12)


def test_real_fourier_rejects_complex_signal():
    with pytest.raises(FourierValidationError, match="real-valued"):
        RealFourierSeries.from_samples(np.array([1 + 1j, 2 + 0j]))

