"""High-field method: label transitions by matrix elements and solve the roots.

Public entry point: :func:`high_field_method`. Candidate (jj, kk) pairs are
chosen from the microwave transition matrix at the high-field end of the
sweep; the resonance field of each pair is then bracketed by the eigenvalue
gaps at both ends and solved with the Brent routine from Root_Finder.
Note that this method may miss spectral peaks when energy level crossings or
avoided crossings occur.
"""

import numpy as np


def _transition_matrix(spin, eigvecs):
    """
    Inputs: Spin object; eigenvectors at the high-field end.

    Returns: M = U^dagger H1 U, with H1 = g1[0, :] . S the microwave coupling.

    Notes: |M[kk, jj]|^2 is the transition probability used by
    Root_Finder.transition_probability.
    """
    from .Spin_Operators import Sx_get, Sy_get, Sz_get
    Sx = Sx_get(spin.S, spin.I)
    Sy = Sy_get(spin.S, spin.I)
    Sz = Sz_get(spin.S, spin.I)
    H1 = spin.g1[0][0] * Sx + spin.g1[0][1] * Sy + spin.g1[0][2] * Sz
    return eigvecs.T.conj() @ H1 @ eigvecs


def _select_transition_pairs(spin, eigvecs, epsilon=1e-2):
    """
    Inputs: Spin object; eigenvectors at the high-field end; relative threshold
    epsilon.

    Returns: the (jj, kk) pairs with |<kk|H1|jj>|^2 above epsilon times the
    largest matrix element.

    Notes: the threshold is insensitive to anisotropy, large E or A, and
    Delta m_s >= 2 transitions from high-order ZFS, because all of these enter
    through the matrix elements. The default 1e-2 sits in the gap between
    allowed and forbidden intensities.
    """
    M = _transition_matrix(spin, eigvecs)
    N = spin.total_Level
    p = np.abs(M) ** 2
    pmax = float(p.max())
    pairs = []
    for jj in range(N - 1):
        for kk in range(jj + 1, N):
            if p[kk, jj] > epsilon * pmax:
                pairs.append((jj, kk))
    return pairs


def _find_transitions(spin, spec, field_left, field_right, eigvals1, eigvals2,
                      pairs, modulation_symbol, find_root, root_args):
    """
    Inputs: Spin and Spec objects; bracket endpoints; eigenvalues at both ends;
    candidate pairs; modulation_symbol; root-finder function for the modulation
    type and its extra positional arguments.

    Returns: the list of transitions whose gap crosses mwFreq between the ends.

    Notes: only pairs whose gap crosses mwFreq are passed to the root finder;
    both endpoint residuals are supplied so the ends are not diagonalized
    again.
    """
    transitions = []
    for jj, kk in pairs:
        left = eigvals1[kk] - eigvals1[jj]
        right = eigvals2[kk] - eigvals2[jj]
        if (left < spec.mwfreq and right > spec.mwfreq) or (left > spec.mwfreq and right < spec.mwfreq):
            transition = find_root(spin, spec.mwfreq, *root_args, field_left, field_right,
                                   jj, kk, modulation_symbol, spec.Temperature,
                                   fa=abs(left) - spec.mwfreq, fb=abs(right) - spec.mwfreq)
            if transition is not None:
                transitions.append(transition)
    return transitions


def high_field_method(spin, spec, modulation_symbol):
    """
    Inputs: Spin and Spec objects; modulation_symbol ("positive" or
    "negative").

    Returns: a list of (field, jj, kk, intensity) transitions.

    Notes: pairs are selected from the microwave transition matrix at the
    high-field end; if fewer than 80% of them yield a root, all pairs are
    retried so that transitions interrupted by level crossings are recovered.
    """
    field_left = spec.fieldrange[0]
    field_right = spec.fieldrange[-1]

    if spec.modtype == 'Bmod':
        from .Root_Finder import find_root_Bmod
        spin.modulate_B(field_left, spec.Bmod, modulation_symbol)
        eigvals1 = np.linalg.eigvalsh(spin.H_total)
        spin.modulate_B(field_right, spec.Bmod, modulation_symbol)
        eigvals2, eigvecs2 = np.linalg.eigh(spin.H_total)
        find_root, root_args = find_root_Bmod, (spec.Bmod,)
    elif spec.modtype == 'Emod':
        from .Root_Finder import find_root_Emod
        spin.update_Etensor(spec.E1)
        spin.modulate_E_initial(spec.E1)
        spin.modulate_E_T(field_left, spec.E1, modulation_symbol)
        eigvals1 = np.linalg.eigvalsh(spin.H_total)
        spin.modulate_E_T(field_right, spec.E1, modulation_symbol)
        eigvals2, eigvecs2 = np.linalg.eigh(spin.H_total)
        find_root, root_args = find_root_Emod, (spec.E1,)
    else:
        from .Root_Finder import find_root_param
        if not (spin.dg1.any() or spin.dD1.any() or spin.dA1.any()):
            raise ValueError('Parameter modulation is selected, but no '
                             'Delta g / Delta D / Delta A is given on the Spin.')
        spin.modulate_param(field_left, modulation_symbol)
        eigvals1 = np.linalg.eigvalsh(spin.H_total)
        spin.modulate_param(field_right, modulation_symbol)
        eigvals2, eigvecs2 = np.linalg.eigh(spin.H_total)
        find_root, root_args = find_root_param, ()

    fast_pairs = _select_transition_pairs(spin, eigvecs2)
    transitions = _find_transitions(spin, spec, field_left, field_right, eigvals1, eigvals2,
                                    fast_pairs, modulation_symbol, find_root, root_args)

    if len(transitions) < 0.8 * len(fast_pairs):
        full_pairs = [(jj, kk) for jj in range(spin.total_Level - 1)
                      for kk in range(jj + 1, spin.total_Level)]
        transitions = _find_transitions(spin, spec, field_left, field_right, eigvals1, eigvals2,
                                        full_pairs, modulation_symbol, find_root, root_args)

    return transitions
