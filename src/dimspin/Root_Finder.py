"""Resonance-field root finding and transition intensities.

Public entry points: find_root_Bmod, find_root_Emod and find_root_param, plus
transition_probability and population_difference for line intensities. The
module is shared by the high-field, adaptive-grid and eigenfield methods.

Energies are in MHz and fields in G.
"""

import numpy as np

from .Spin_Operators import Sx_get, Sy_get, Sz_get


def transition_probability(S, I, g, eigvecs, jj, kk):
    """
    Inputs: electron and nuclear spins; 3x3 g tensor; eigenvectors at the
    resonance field; level indices jj and kk.

    Returns: |<kk|H1|jj>|^2, with H1 = g[0, :] . S the microwave coupling.

    Notes: the first row of ``g`` is used because the microwave field is taken
    along the laboratory x axis.
    """
    Sx = Sx_get(S, I)
    Sy = Sy_get(S, I)
    Sz = Sz_get(S, I)
    H1 = g[0][0] * Sx + g[0][1] * Sy + g[0][2] * Sz
    element = np.dot(np.dot(eigvecs[:, kk].T.conjugate(), H1), eigvecs[:, jj])
    return np.absolute(element) ** 2


def population_difference(eigvals, jj, kk, Temperature):
    """
    Inputs: eigenvalues at the resonance field (MHz); level indices jj and kk;
    temperature (K).

    Returns: the normalized Boltzmann population difference between jj and kk.

    Notes: ``h / k_B`` is applied per MHz.
    """
    k_B = 1.380649e-23
    h_per_mhz = 6.626e-28
    beta = h_per_mhz / (k_B * Temperature)
    factors = np.exp(-(np.asarray(eigvals, dtype=float) - np.min(eigvals)) * beta)
    return (factors[jj] - factors[kk]) / factors.sum()


def _brentq(f, a, b, xtol, rtol, maxiter=100):
    """
    Inputs: bracketing function f; endpoints a and b with residuals of opposite
    sign; absolute and relative tolerances; iteration limit.

    Returns: a root of f in [a, b].

    Notes: self-contained Brent implementation; it avoids importing scipy.optimize
    inside worker processes,thereby significantly improving computational efficiency.
    """
    xpre = a * 1.0
    xcur = b * 1.0
    fpre = f(xpre)
    fcur = f(xcur)
    if fpre * fcur > 0:
        raise ValueError("f(a) and f(b) must have different signs")
    if fpre == 0:
        return xpre
    if fcur == 0:
        return xcur

    xblk = 0.0
    fblk = 0.0
    spre = 0.0
    scur = 0.0

    for _ in range(maxiter):
        if fpre * fcur < 0:
            xblk = xpre
            fblk = fpre
            spre = scur = xcur - xpre
        if abs(fblk) < abs(fcur):
            xpre = xcur
            xcur = xblk
            xblk = xpre
            fpre = fcur
            fcur = fblk
            fblk = fpre

        delta = (xtol + rtol * abs(xcur)) / 2
        sbis = (xblk - xcur) / 2
        if fcur == 0 or abs(sbis) < delta:
            return xcur

        if abs(spre) > delta and abs(fcur) < abs(fpre):
            if xpre == xblk:
                stry = -fcur * (xcur - xpre) / (fcur - fpre)
            else:
                dpre = (fpre - fcur) / (xpre - xcur)
                dblk = (fblk - fcur) / (xblk - xcur)
                stry = -fcur * (fblk * dblk - fpre * dpre) / (dblk * dpre * (fblk - fpre))
            if 2 * abs(stry) < min(abs(spre), 3 * abs(sbis) - delta):
                spre = scur
                scur = stry
            else:
                spre = sbis
                scur = sbis
        else:
            spre = sbis
            scur = sbis

        xpre = xcur
        fpre = fcur
        if abs(scur) > delta:
            xcur += scur
        else:
            xcur += (delta if sbis > 0 else -delta)
        fcur = f(xcur)

    return xcur


def _find_root(func, field_left, field_right, fa, fb, xtol, rtol):
    """
    Inputs: residual function; bracket endpoints; precomputed endpoint
    residuals fa and fb, which may be None; tolerances.

    Returns: the root found by :func:`_brentq`.

    Notes: fa/fb are computed by the caller from the end eigenvalues,
    so the bracket ends are not diagonalized again.
    """
    def f(x):
        if fa is not None and x == field_left:
            return fa
        if fb is not None and x == field_right:
            return fb
        return func(x)
    return _brentq(f, field_left, field_right, xtol, rtol)


def _root_record(spin, mwFreq, set_field, field_left, field_right, jj, kk,
                 Temperature, fa, fb, xtol):
    """
    Inputs: Spin object; microwave frequency (MHz); callable that sets the
    field-dependent Hamiltonian; bracket endpoints; level indices jj and kk;
    temperature (K); endpoint residuals; absolute tolerance.

    Returns: (field, jj, kk, intensity), or None when the transition
    probability is zero.
    """
    def residual(x):
        set_field(x)
        eigvals = np.linalg.eigvalsh(spin.H_total)
        return np.abs(eigvals[kk] - eigvals[jj]) - mwFreq

    field = _find_root(residual, field_left, field_right, fa, fb, xtol=xtol, rtol=1e-6)
    set_field(field)
    eigvals, eigvecs = np.linalg.eigh(spin.H_total)
    probability = transition_probability(spin.S, spin.I, spin.g1, eigvecs, jj, kk)
    if probability > 0.0:
        difference = population_difference(eigvals, jj, kk, Temperature)
        return (field, jj, kk, probability * difference)
    return None


def find_root_Emod(spin, mwFreq, EF, field_left, field_right, jj, kk,
                   modulation_symbol, Temperature, fa=None, fb=None):
    """
    Inputs: Spin object; microwave frequency (MHz); electric-field vector EF;
    bracket endpoints; level indices jj and kk; modulation_symbol ("positive"
    or "negative"); temperature (K); optional endpoint residuals.

    Returns: (field, jj, kk, intensity), or None when the transition
    probability is zero.

    Notes: the absolute tolerance is 0.01 G, well below the field resolution.
    """
    return _root_record(
        spin, mwFreq,
        lambda x: spin.modulate_E_T(x, EF, modulation_symbol),
        field_left, field_right, jj, kk, Temperature, fa, fb, xtol=0.01,
    )


def find_root_Bmod(spin, mwFreq, Bmod, field_left, field_right, jj, kk,
                   modulation_symbol, Temperature, fa=None, fb=None):
    """
    Inputs: Spin object; microwave frequency (MHz); modulated field Bmod (G);
    bracket endpoints; level indices jj and kk; modulation_symbol ("positive"
    or "negative"); temperature (K); optional endpoint residuals.

    Returns: (field, jj, kk, intensity), or None when the transition
    probability is zero.

    Notes: the absolute tolerance is 0.05 G, about 0.14 MHz.
    """
    return _root_record(
        spin, mwFreq,
        lambda x: spin.modulate_B(x, Bmod, modulation_symbol),
        field_left, field_right, jj, kk, Temperature, fa, fb, xtol=0.05,
    )


def find_root_param(spin, mwFreq, field_left, field_right, jj, kk,
                    modulation_symbol, Temperature, fa=None, fb=None):
    """
    Inputs: Spin object; microwave frequency (MHz); bracket endpoints; level
    indices jj and kk; modulation_symbol ("positive" or "negative");
    temperature (K); optional endpoint residuals.

    Returns: (field, jj, kk, intensity), or None when the transition
    probability is zero.

    Notes: the absolute tolerance is 0.01 G,  well below the field resolution.
    """
    return _root_record(
        spin, mwFreq,
        lambda x: spin.modulate_param(x, modulation_symbol),
        field_left, field_right, jj, kk, Temperature, fa, fb, xtol=0.01,
    )
