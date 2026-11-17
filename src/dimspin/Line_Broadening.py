"""Line-shape profiles and spectrum broadening.

Public entry point: :func:`broaden`. Two evaluation paths are available:
direct summation (exact for any grid) and FFT convolution (uniform grid, fast
for many transitions). The FFT path adds out-of-grid transitions by direct
evaluation, so their tails inside the grid are retained.

All widths use the same units as the field grid.
"""

from functools import lru_cache

import numpy as np
from scipy.special import wofz


def _line_values(x, sigma_l, sigma_g):
    """Return the area-normalized line profile at offsets ``x``."""
    sqrt2 = np.sqrt(2.0)
    if sigma_g == 0:
        return (sigma_l / np.pi) / (x * x + sigma_l * sigma_l)
    z = (x + 1j * sigma_l) / (sigma_g * sqrt2)
    return np.real(wofz(z)) / (sigma_g * np.sqrt(2.0 * np.pi))


def line_shape(fieldrange, position, intensity, sigma_l, sigma_g):
    """Return one area-normalized line.

    Inputs: field grid, center, integrated intensity, Lorentzian half-width,
    Gaussian standard deviation. At least one width must be nonzero.

    Returns the line values; their integral over the grid equals ``intensity``.
    ``sigma_g = 0`` gives a Lorentzian; ``sigma_l = 0`` gives a Gaussian.
    """
    fieldrange = np.asarray(fieldrange, dtype=float)
    return intensity * _line_values(fieldrange - position, sigma_l, sigma_g)


def broaden_direct(fieldrange, positions, intensities, sigma_l, sigma_g):
    """Sum line shapes directly.

    Inputs: field grid, positions, intensities, Lorentzian half-width, Gaussian
    standard deviation.

    Returns the summed spectrum. Exact for any grid; positions outside the grid
    contribute their tails inside it. Cost is O(M * N).
    """
    fieldrange = np.asarray(fieldrange, dtype=float)
    positions = np.asarray(positions, dtype=float).ravel()
    intensities = np.asarray(intensities, dtype=float).ravel()
    if positions.size != intensities.size:
        raise ValueError("positions and intensities must have the same length")
    if positions.size == 0:
        return np.zeros(fieldrange.size)

    x = fieldrange[None, :] - positions[:, None]
    y = _line_values(x, sigma_l, sigma_g)
    return (y * intensities[:, None]).sum(axis=0)


@lru_cache(maxsize=None)
def _kernel_fft(spacing, N, sigma_l, sigma_g, oversample):
    """Return the cached FFT of a line-shape kernel for a uniform grid."""
    Nf = N * oversample
    spacing_f = spacing / oversample
    Mf = 2 * Nf
    offsets = (np.arange(Mf) - Nf) * spacing_f
    return np.fft.fft(_line_values(offsets, sigma_l, sigma_g))


def _voigt_kernel_fft(fieldrange, sigma_l, sigma_g, oversample=1):
    """Return the kernel FFT for ``fieldrange`` and the given widths."""
    fieldrange = np.asarray(fieldrange, dtype=float)
    N = fieldrange.size
    spacing = fieldrange[1] - fieldrange[0]
    return _kernel_fft(spacing, N, sigma_l, sigma_g, oversample)


def _fft_inside(fieldrange, positions, intensities, sigma_l, sigma_g,
                kernel_fft=None, oversample=1):
    """FFT convolution for transitions inside the field grid."""
    fieldrange = np.asarray(fieldrange, dtype=float)
    N = fieldrange.size
    spacing = fieldrange[1] - fieldrange[0]
    lo = fieldrange[0]

    if kernel_fft is None:
        kernel_fft = _voigt_kernel_fft(fieldrange, sigma_l, sigma_g, oversample)
    Nf = kernel_fft.shape[0] // 2
    spacing_f = spacing * N / Nf

    idx = (positions - lo) / spacing_f
    i0 = np.floor(idx).astype(int)
    frac = idx - i0
    valid = (i0 >= 0) & (i0 <= Nf - 1)
    i0 = i0[valid]
    frac = frac[valid]
    inten = intensities[valid]
    if i0.size == 0:
        return np.zeros(N)

    stick = np.zeros(Nf)
    np.add.at(stick, i0, inten * (1.0 - frac))
    np.add.at(stick, np.minimum(i0 + 1, Nf - 1), inten * frac)

    Mf = 2 * Nf
    stick_pad = np.zeros(Mf)
    stick_pad[:Nf] = stick
    full = np.fft.ifft(np.fft.fft(stick_pad) * kernel_fft).real
    return full[Nf:Nf + Nf][::(Nf // N)]


def broaden_fft(fieldrange, positions, intensities, sigma_l, sigma_g,
                kernel_fft=None, oversample=1):
    """Sum line shapes by FFT convolution, including out-of-grid tails.

    Inputs: uniform field grid, positions (may lie outside the grid),
    intensities, Lorentzian half-width, Gaussian standard deviation.
    ``kernel_fft`` may be supplied for repeated calls; ``oversample`` refines
    the internal grid.

    Returns the summed spectrum. Inside transitions use FFT; outside
    transitions are added by direct evaluation. Sub-grid error is about 0.3%
    of the peak at ``oversample=1`` and scales as 1/F^2.
    """
    fieldrange = np.asarray(fieldrange, dtype=float)
    N = fieldrange.size
    if N < 2:
        raise ValueError("fieldrange must contain at least two points")
    spacing = fieldrange[1] - fieldrange[0]
    if not np.allclose(np.diff(fieldrange), spacing, rtol=1e-6, atol=1e-12):
        raise ValueError("broaden_fft requires a uniformly spaced field grid")

    positions = np.asarray(positions, dtype=float).ravel()
    intensities = np.asarray(intensities, dtype=float).ravel()
    if positions.size != intensities.size:
        raise ValueError("positions and intensities must have the same length")
    if positions.size == 0:
        return np.zeros(N)

    lo = fieldrange[0]
    hi = fieldrange[-1]
    inside = (positions >= lo) & (positions <= hi)
    spectrum = np.zeros(N)
    if np.any(inside):
        spectrum += _fft_inside(
            fieldrange, positions[inside], intensities[inside],
            sigma_l, sigma_g, kernel_fft, oversample,
        )
    if np.any(~inside):
        spectrum += broaden_direct(
            fieldrange, positions[~inside], intensities[~inside],
            sigma_l, sigma_g,
        )
    return spectrum


def broaden(fieldrange, positions, intensities, sigma_l, sigma_g,
            method="auto", oversample=1, threshold=2):
    """Broaden all transitions and return the summed spectrum.

    Inputs: field grid, positions, intensities, Lorentzian half-width, Gaussian
    standard deviation. ``method`` is "auto", "direct" or "fft"; "auto" uses
    direct summation for at most ``threshold`` transitions and FFT otherwise.

    Returns the summed spectrum. Lines are area-normalized, so its integral
    equals the sum of the intensities.
    """
    if method not in ("auto", "direct", "fft"):
        raise ValueError("method must be 'auto', 'direct' or 'fft'")

    count = np.asarray(positions).size
    if method == "direct" or (method == "auto" and count <= threshold):
        return broaden_direct(fieldrange, positions, intensities, sigma_l, sigma_g)
    return broaden_fft(
        fieldrange, positions, intensities, sigma_l, sigma_g,
        oversample=oversample,
    )
