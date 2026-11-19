"""Adaptive-grid resonance search with Hermite cubic spline roots.

Public entry point: :func:`adaptive_grid_method`. The energy-level diagram is
refined adaptively until a cubic Hermite spline reproduces the levels to a
given tolerance; the roots of each level-gap spline are then solved
analytically and backfilled with transition intensities. This follows the
same approach as EasySpin: the spline roots are trusted as returned,
without an iterative refinement step.

The method makes no high-field assumption: it evaluates all level pairs that
can cross the microwave frequency and therefore keeps forbidden and
level-crossing transitions.

Reference: S. Stoll and A. Schweiger, EasySpin, a comprehensive software
package for spectral simulation and analysis in EPR, J. Magn. Reson. 178
(2006) 42-55, doi:10.1016/j.jmr.2005.08.013.
"""

import numpy as np


def _feynman_derivatives(eigvecs, H1):
    """
    Inputs: eigenvectors at one field; field-gradient matrix H1 = dH/dB.

    Returns: the level derivatives dE_u/dB = <u|H1|u>, one per level.

    """
    return np.sum(eigvecs.conj() * (H1 @ eigvecs), axis=0).real


def _cubic_roots_t(dL, dR, dLp, dRp):
    """
    Inputs: Hermite endpoint values dL and dR of the level gap minus the
    microwave frequency, and endpoint derivatives dLp and dRp with respect to
    the normalized parameter t.

    Returns: the real roots in [0, 1] of the cubic Hermite polynomial.

    Notes: the roots inherit the cubic interpolation error; the caller uses
    them directly.
    """
    a3 = 2.0 * (dL - dR) + dLp + dRp
    a2 = 3.0 * (dR - dL) - 2.0 * dLp - dRp
    a1 = dLp
    a0 = dL
    roots = np.roots([a3, a2, a1, a0])
    ts = []
    for r in roots:
        if abs(r.imag) < 1e-8 and -1e-9 <= r.real <= 1.0 + 1e-9:
            ts.append(float(r.real))
    return ts


def _adaptive_level_diagram(H0, H1, field_lo, field_hi, tol=0.5, max_depth=20,
                            min_width=1.0):
    """
    Inputs: field-independent Hamiltonian H0 and gradient H1, so that
    H(B) = H0 + B * H1; field range; spline tolerance (MHz); maximum
    subdivision depth; minimum segment width (G).

    Returns: a list of segments (B_L, B_R, E_L, E_R, Ep_L, Ep_R), each holding
    the endpoint fields, sorted eigenvalues and level derivatives.

    Notes: a segment splits while the Hermite midpoint prediction differs from
    the exact eigenvalues by more than ``tol``. Near level crossings the
    sorted eigenvalues have a discontinuous derivative, so subdivision stops
    at ``max_depth`` or at ``min_width``; the endpoint eigenvalues are always
    exact, so the spline accuracy is not needed by the final roots.
    """
    segments = []

    def eig_at(B):
        vals, vecs = np.linalg.eigh(H0 + B * H1)
        return vals, _feynman_derivatives(vecs, H1)

    E_lo, Ep_lo = eig_at(field_lo)
    E_hi, Ep_hi = eig_at(field_hi)

    def subdivide(B_L, B_R, E_L, E_R, Ep_L, Ep_R, depth):
        if depth >= max_depth or (B_R - B_L) <= min_width:
            segments.append((B_L, B_R, E_L, E_R, Ep_L, Ep_R))
            return
        B_M = 0.5 * (B_L + B_R)
        E_pred = 0.5 * (E_L + E_R) + 0.125 * (B_R - B_L) * (Ep_L - Ep_R)
        E_M, vecs_M = np.linalg.eigh(H0 + B_M * H1)
        if float(np.max(np.abs(E_M - E_pred))) <= tol:
            segments.append((B_L, B_R, E_L, E_R, Ep_L, Ep_R))
            return
        Ep_M = _feynman_derivatives(vecs_M, H1)
        subdivide(B_L, B_M, E_L, E_M, Ep_L, Ep_M, depth + 1)
        subdivide(B_M, B_R, E_M, E_R, Ep_M, Ep_R, depth + 1)

    subdivide(field_lo, field_hi, E_lo, E_hi, Ep_lo, Ep_hi, 0)
    return segments


def _resonating_pairs(segments, mwfreq, N):
    """
    Inputs: spline segments; microwave frequency (MHz); number of levels.

    Returns: the (u, v) pairs whose gap lies between the global minimum and
    maximum of E_v - E_u over all segment endpoints.

    Notes: the gap is continuous in the field, so a resonance is accompanied
    by a segment endpoint whose gap crosses ``mwfreq``; only a gap that
    crosses the frequency twice inside one segment can be missed, which the
    adaptive subdivision avoids in practice.
    """
    gap_min = np.full((N, N), np.inf)
    gap_max = np.full((N, N), -np.inf)
    for _, _, E_L, E_R, _, _ in segments:
        for E in (E_L, E_R):
            G = E[:, None] - E[None, :]
            np.minimum(gap_min, G, out=gap_min)
            np.maximum(gap_max, G, out=gap_max)
    return [(u, v) for u in range(N - 1) for v in range(u + 1, N)
            if gap_min[v, u] <= mwfreq <= gap_max[v, u]]


def _dedup_roots(roots, tol=0.05):
    """
    Inputs: roots as (field, u, v); merge tolerance in G.

    Returns: the roots with repeats of the same (u, v) closer than ``tol``
    merged, keeping one.

    Notes: only repeats of the same pair are merged; degenerate resonances of
    different pairs at the same field are distinct transitions and are kept.
    """
    if not roots:
        return []
    roots.sort(key=lambda r: (r[1], r[2], r[0]))
    out = []
    for B, u, v in roots:
        if out and out[-1][1] == u and out[-1][2] == v and (B - out[-1][0]) <= tol:
            continue
        out.append((B, u, v))
    return out


def _backfill(H0, H1, mwfreq, spin, spec, roots, tol_freq=0.5):
    """
    Inputs: Hamiltonian constants; microwave frequency (MHz); Spin and Spec
    objects; roots as (field, u, v); gap tolerance (MHz).

    Returns: transitions (field, u, v, intensity) with intensity equal to the
    transition probability times the population difference.

    Notes: each root field is diagonalized once. A root whose actual gap
    differs from ``mwfreq`` by more than ``tol_freq`` is discarded; this only
    happens when the pair indices are misassigned at a level crossing.
    """
    from .Root_Finder import population_difference, transition_probability
    transitions = []
    for B, u, v in roots:
        eigvals, eigvecs = np.linalg.eigh(H0 + B * H1)
        if abs(abs(eigvals[v] - eigvals[u]) - mwfreq) > tol_freq:
            continue
        p = transition_probability(spin.S, spin.I, spin.g1, eigvecs, u, v)
        if p > 0.0:
            pd = population_difference(eigvals, u, v, spec.Temperature)
            transitions.append((B, u, v, p * pd))
    return transitions


def _spline_roots(segments, mwfreq, pairs):
    """
    Inputs: spline segments; microwave frequency (MHz); level pairs.

    Returns: roots (field, u, v) from the analytic cubic roots on each segment.

    Notes: no refinement is applied to the analytic roots.
    """
    roots = []
    for B_L, B_R, E_L, E_R, Ep_L, Ep_R in segments:
        delta = B_R - B_L
        if delta == 0:
            continue
        for u, v in pairs:
            dL = (E_L[v] - E_L[u]) - mwfreq
            dR = (E_R[v] - E_R[u]) - mwfreq
            if dL * dR > 0.0:
                continue
            dLp = (Ep_L[v] - Ep_L[u]) * delta
            dRp = (Ep_R[v] - Ep_R[u]) * delta
            for t in _cubic_roots_t(dL, dR, dLp, dRp):
                roots.append((B_L + t * delta, u, v))
    return roots


def adaptive_grid_method(spin, spec, modulation_symbol, tol=0.1, max_depth=20,
                         tol_freq=0.5):
    """
    Inputs: Spin and Spec objects; modulation_symbol ("positive" or
    "negative"); spline tolerance (MHz); maximum subdivision depth; gap
    tolerance for the intensity backfill (MHz).

    Returns: transitions (field, u, v, intensity), with intensity equal to the
    transition probability times the population difference.

    Notes: all resonating level pairs are used, without a high-field label
    selection, so forbidden and level-crossing transitions are included.
    """
    from .Eigenfield_Method import _build_H0_H1
    H0, H1 = _build_H0_H1(spin, spec, modulation_symbol)

    segments = _adaptive_level_diagram(H0, H1, spec.fieldrange[0], spec.fieldrange[-1],
                                       tol=tol, max_depth=max_depth)
    pairs = _resonating_pairs(segments, spec.mwfreq, H0.shape[0])
    roots = _dedup_roots(_spline_roots(segments, spec.mwfreq, pairs))
    return _backfill(H0, H1, spec.mwfreq, spin, spec, roots, tol_freq=tol_freq)
