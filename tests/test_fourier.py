import numpy as np
import pytest

from fourier_line_art.fourier import FourierSeries, FourierValidationError


def test_constant_signal_reconstructs_with_one_term():
    samples = np.full(32, 2.5 + 1.25j)
    series = FourierSeries.from_samples(samples)

    actual = series.reconstruct(term_count=1)

    np.testing.assert_allclose(actual, samples, atol=1e-12)


def test_full_dft_reconstructs_arbitrary_complex_samples():
    samples = np.array([0 + 1j, 2 - 3j, -1 + 0.5j, 4 + 2j, 1 - 2j])
    series = FourierSeries.from_samples(samples)

    np.testing.assert_allclose(series.reconstruct(len(samples)), samples, atol=1e-12)


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

