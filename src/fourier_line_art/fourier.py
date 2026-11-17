"""Discrete Fourier coefficients and nested partial reconstructions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numbers import Integral


class FourierValidationError(ValueError):
    """Invalid samples or coefficient count supplied to the Fourier solver."""


@dataclass(frozen=True, slots=True)
class FourierTerm:
    """One rotating complex Fourier vector."""

    frequency: int
    coefficient: complex | tuple[complex, ...]


@dataclass(frozen=True, slots=True)
class FourierSeries:
    """A DFT whose terms are ordered by coefficient energy.

    `samples` may be a complex 1D signal, such as x + i*y, or a two-dimensional
    array whose rows are samples and columns are channels. Since DFT basis
    vectors are orthogonal, adding terms in any order cannot increase the
    sample-space RMS error.
    """

    _samples: np.ndarray
    _coefficients: np.ndarray
    _frequencies: np.ndarray
    _order: tuple[int, ...]

    @classmethod
    def from_samples(cls, samples) -> "FourierSeries":
        values = np.asarray(samples)
        if values.ndim not in (1, 2) or values.shape[0] == 0 or (values.ndim == 2 and values.shape[1] == 0):
            raise FourierValidationError(
                "samples must be a non-empty 1D or 2D array with at least one channel"
            )
        if not np.issubdtype(values.dtype, np.number):
            raise FourierValidationError("samples must be numeric")
        if not np.all(np.isfinite(values)):
            raise FourierValidationError("samples must be finite")

        values = np.array(values, dtype=np.complex128, copy=True)
        coefficients = np.fft.fft(values, axis=0) / values.shape[0]
        frequencies = np.rint(np.fft.fftfreq(values.shape[0]) * values.shape[0]).astype(int)
        magnitudes = np.linalg.norm(coefficients.reshape(values.shape[0], -1), axis=1)
        order = tuple(
            sorted(
                range(values.shape[0]),
                key=lambda index: (
                    -float(magnitudes[index]),
                    abs(int(frequencies[index])),
                    int(frequencies[index]),
                ),
            )
        )
        for array in (values, coefficients, frequencies):
            array.setflags(write=False)
        return cls(values, coefficients, frequencies, order)

    @property
    def sample_count(self) -> int:
        return int(self._samples.shape[0])

    @property
    def term_count(self) -> int:
        return len(self._order)

    @property
    def ordered_frequencies(self) -> np.ndarray:
        values = self._frequencies[list(self._order)]
        values.setflags(write=False)
        return values

    @property
    def ordered_coefficients(self) -> np.ndarray:
        values = self._coefficients[list(self._order)]
        values.setflags(write=False)
        return values

    @property
    def ordered_terms(self) -> tuple[FourierTerm, ...]:
        terms = []
        for index in self._order:
            value = self._coefficients[index]
            if value.ndim == 0:
                coefficient: complex | tuple[complex, ...] = complex(value)
            else:
                coefficient = tuple(complex(component) for component in value)
            terms.append(FourierTerm(int(self._frequencies[index]), coefficient))
        return tuple(terms)

    def reconstruct(self, term_count: int) -> np.ndarray:
        if (
            isinstance(term_count, bool)
            or not isinstance(term_count, Integral)
            or not 0 <= term_count <= self.term_count
        ):
            raise FourierValidationError(
                f"term_count must be an integer between 0 and {self.term_count}"
            )
        return self.evaluate(np.arange(self.sample_count) / self.sample_count, int(term_count))

    def evaluate(self, t, term_count: int) -> np.ndarray:
        """Evaluate a partial Fourier sum at normalized time(s) in one period."""
        if (
            isinstance(term_count, bool)
            or not isinstance(term_count, Integral)
            or not 0 <= term_count <= self.term_count
        ):
            raise FourierValidationError(
                f"term_count must be an integer between 0 and {self.term_count}"
            )
        try:
            times = np.asarray(t, dtype=float)
        except (TypeError, ValueError) as exc:
            raise FourierValidationError("t must contain finite numeric values") from exc
        if not np.all(np.isfinite(times)):
            raise FourierValidationError("t must contain finite numeric values")
        indices = self._order[: int(term_count)]
        if not indices:
            shape = (
                times.shape
                if self._samples.ndim == 1
                else (*times.shape, self._samples.shape[1])
            )
            return np.zeros(shape, dtype=np.complex128)
        frequencies = self._frequencies[list(indices)]
        coefficients = self._coefficients[list(indices)]
        basis = np.exp(2j * np.pi * times[..., None] * frequencies)
        if self._samples.ndim == 1:
            return np.sum(basis * coefficients, axis=-1)
        return np.sum(basis[..., None] * coefficients, axis=-2)

    def rms_error(self, term_count: int) -> float:
        difference = self._samples - self.reconstruct(term_count)
        return float(np.sqrt(np.mean(np.abs(difference) ** 2)))

    def rms_errors(self) -> np.ndarray:
        """Return RMS errors for term counts 0..N using Parseval's identity."""
        channel_count = 1 if self._samples.ndim == 1 else self._samples.shape[1]
        energies = (
            np.sum(np.abs(self._coefficients.reshape(self.sample_count, -1)) ** 2, axis=1)
            / channel_count
        )
        ordered = energies[list(self._order)]
        remaining = np.concatenate((np.cumsum(ordered[::-1])[::-1], [0.0]))
        errors = np.sqrt(np.maximum(remaining, 0.0))
        errors.setflags(write=False)
        return errors


@dataclass(frozen=True, slots=True)
class RealFourierSeries:
    """A real signal represented by an offset and ordered cosine harmonics.

    Positive-frequency conjugate pairs are combined into a single cosine
    component. The Nyquist component, when present, is kept at its correct
    single-sided amplitude. Harmonics are ordered by their contribution to
    sample-space squared error.
    """

    _samples: np.ndarray
    mean: float
    _frequencies: np.ndarray
    _amplitudes: np.ndarray
    _phases: np.ndarray
    _energies: np.ndarray
    _order: tuple[int, ...]

    @classmethod
    def from_samples(cls, samples) -> "RealFourierSeries":
        values = np.asarray(samples)
        if values.ndim != 1 or values.shape[0] == 0:
            raise FourierValidationError("real samples must be a non-empty 1D array")
        if not np.issubdtype(values.dtype, np.number):
            raise FourierValidationError("samples must be numeric")
        if not np.all(np.isfinite(values)):
            raise FourierValidationError("samples must be finite")
        if np.iscomplexobj(values) and np.any(np.imag(values) != 0):
            raise FourierValidationError("RealFourierSeries requires real-valued samples")

        values = np.array(np.real(values), dtype=np.float64, copy=True)
        sample_count = len(values)
        coefficients = np.fft.rfft(values) / sample_count
        maximum_frequency = sample_count // 2
        frequencies = np.arange(1, maximum_frequency + 1, dtype=np.int64)
        amplitudes = 2 * np.abs(coefficients[1 : maximum_frequency + 1])
        phases = np.angle(coefficients[1 : maximum_frequency + 1])
        is_nyquist = sample_count % 2 == 0 and maximum_frequency > 0
        if is_nyquist:
            amplitudes[-1] /= 2

        energies = amplitudes**2 / 2
        if is_nyquist:
            energies[-1] = amplitudes[-1] ** 2
        order = tuple(
            sorted(
                range(len(frequencies)),
                key=lambda index: (
                    -float(energies[index]),
                    int(frequencies[index]),
                ),
            )
        )
        mean = float(coefficients[0].real)
        for array in (values, frequencies, amplitudes, phases, energies):
            array.setflags(write=False)
        return cls(values, mean, frequencies, amplitudes, phases, energies, order)

    @property
    def sample_count(self) -> int:
        return int(len(self._samples))

    @property
    def term_count(self) -> int:
        return len(self._order)

    @property
    def ordered_frequencies(self) -> np.ndarray:
        values = self._frequencies[list(self._order)]
        values.setflags(write=False)
        return values

    @property
    def ordered_amplitudes(self) -> np.ndarray:
        values = self._amplitudes[list(self._order)]
        values.setflags(write=False)
        return values

    @property
    def ordered_phases(self) -> np.ndarray:
        values = self._phases[list(self._order)]
        values.setflags(write=False)
        return values

    def evaluate(self, t, term_count: int) -> np.ndarray:
        """Evaluate the offset plus the requested highest-energy harmonics."""
        if (
            isinstance(term_count, bool)
            or not isinstance(term_count, Integral)
            or not 0 <= term_count <= self.term_count
        ):
            raise FourierValidationError(
                f"term_count must be an integer between 0 and {self.term_count}"
            )
        try:
            times = np.asarray(t, dtype=float)
        except (TypeError, ValueError) as exc:
            raise FourierValidationError("t must contain finite numeric values") from exc
        if not np.all(np.isfinite(times)):
            raise FourierValidationError("t must contain finite numeric values")
        if term_count == 0:
            return np.full(times.shape, self.mean, dtype=np.float64)
        indices = self._order[: int(term_count)]
        frequencies = self._frequencies[list(indices)]
        amplitudes = self._amplitudes[list(indices)]
        phases = self._phases[list(indices)]
        angles = 2 * np.pi * times[..., None] * frequencies + phases
        return self.mean + np.sum(amplitudes * np.cos(angles), axis=-1)

    def reconstruct(self, term_count: int) -> np.ndarray:
        """Reconstruct the original sample times with a partial harmonic sum."""
        if (
            isinstance(term_count, bool)
            or not isinstance(term_count, Integral)
            or not 0 <= term_count <= self.term_count
        ):
            raise FourierValidationError(
                f"term_count must be an integer between 0 and {self.term_count}"
            )
        return self.evaluate(np.arange(self.sample_count) / self.sample_count, int(term_count))

    def rms_error(self, term_count: int) -> float:
        difference = self._samples - self.reconstruct(term_count)
        return float(np.sqrt(np.mean(difference**2)))

    def rms_errors(self) -> np.ndarray:
        """Return the RMS errors for term counts zero through full precision."""
        ordered = self._energies[list(self._order)]
        remaining = np.concatenate((np.cumsum(ordered[::-1])[::-1], [0.0]))
        errors = np.sqrt(np.maximum(remaining, 0.0))
        errors.setflags(write=False)
        return errors

