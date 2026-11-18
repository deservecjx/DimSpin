"""Resonance fields from the generalized eigenvalue problem.

The spin Hamiltonian is linear in the field, H(B) = H0 + B * H1. The
eigenvalues of the commutator superoperator L = I (x) H - H (x) H are all
level differences, so the resonance condition E_j - E_i = mwfreq is the
generalized eigenvalue problem (mwfreq * I - L0) v = B * L1 v.

Public entry point: :func:`eigenfield_method`. It returns every resonance in
the field window at once, including forbidden and level-crossing
transitions, without labeling or bracketing. The cost grows as the square of
the level count, so it serves as an independent reference for the iterative
methods rather than a production path.
"""

import numpy as np
from scipy.linalg import eigvals


def _build_H0_H1(spin, spec, modulation_symbol):
    """
    Inputs: Spin and Spec objects; modulation_symbol ("positive" or
    "negative").

    Returns: the constant matrices H0 and H1 with H(B) = H0 + B * H1.

    Notes: the Spin must already have its field-independent terms and field
    gradients prepared. Electric-field terms are built here from the Spec.
    """
    if modulation_symbol not in ('positive', 'negative'):
        raise ValueError("modulation_symbol must be 'positive' or 'negative'.")
    sign = 1 if modulation_symbol == 'positive' else -1

    if spec.modtype == 'Bmod':
        H0 = spin.H_initial + sign * spec.Bmod * spin.H_g_grad
        H1 = spin.H_g_grad
    elif spec.modtype == 'Emod':
        spin.update_Etensor(spec.E1)
        spin.modulate_E_initial(spec.E1)
        H0 = spin.H_initial + sign * spin.H_Einitial
        H1 = spin.H_g_grad + sign * spin.H_T_grad
    else:
        if not (spin.dg1.any() or spin.dD1.any() or spin.dA1.any()):
            raise ValueError('Parameter modulation is selected, but no '
                             'Delta g / Delta D / Delta A is given on the Spin.')
        H0 = spin.H_initial + sign * spin.H_delta_const
        H1 = spin.H_g_grad + sign * spin.H_dg_grad
    return H0, H1


def _resonance_fields(H0, H1, mwfreq, field_lo, field_hi):
    """
    Inputs: Hamiltonian constants; microwave frequency (MHz); field window.

    Returns: the resonance fields inside the window, sorted and deduplicated.

    Notes: the superoperator matrices have dimension N^2 with N the number of
    levels, so the dense QZ solve is much slower than the iterative methods.
    L1 is singular, hence non-finite eigenvalues are filtered out.
    """
    N = H0.shape[0]
    I = np.eye(N)
    L0 = np.kron(I, H0) - np.kron(H0, I)
    L1 = np.kron(I, H1) - np.kron(H1, I)
    A = mwfreq * np.eye(N * N) - L0
    C = L1
    w = eigvals(A, C)
    tol = 1e-6
    real = np.real(w)
    mask = (np.isfinite(w)
            & (np.abs(np.imag(w)) <= tol * (1.0 + np.abs(real)))
            & (real >= field_lo - tol) & (real <= field_hi + tol))
    fields = np.sort(real[mask])
    if fields.size:
        keep = [fields[0]]
        for f in fields[1:]:
            if f - keep[-1] > 1e-3:
                keep.append(f)
        fields = np.array(keep)
    return fields


def eigenfield_method(spin, spec, modulation_symbol):
    """
    Inputs: Spin and Spec objects; modulation_symbol ("positive" or
    "negative").

    Returns: transitions (field, jj, kk, intensity) for every resonance in the
    field window, with intensity equal to the transition probability times the
    population difference.

    Notes: the intensities are backfilled by diagonalizing H(B) once per
    resonance field, so they match the other methods. The field tolerance for
    matching a gap to mwfreq is 1e-2 MHz.
    """
    H0, H1 = _build_H0_H1(spin, spec, modulation_symbol)

    fields = _resonance_fields(H0, H1, spec.mwfreq,
                               spec.fieldrange[0], spec.fieldrange[-1])

    from .Root_Finder import population_difference, transition_probability

    tol_freq = 1e-2
    transitions = []
    for B in fields:
        eigvals, eigvecs = np.linalg.eigh(H0 + B * H1)
        for jj in range(spin.total_Level):
            for kk in range(jj + 1, spin.total_Level):
                if abs(abs(eigvals[kk] - eigvals[jj]) - spec.mwfreq) > tol_freq:
                    continue
                p = transition_probability(spin.S, spin.I, spin.g1, eigvecs, jj, kk)
                if p > 0.0:
                    pd = population_difference(eigvals, jj, kk, spec.Temperature)
                    transitions.append((B, jj, kk, p * pd))
    return transitions
