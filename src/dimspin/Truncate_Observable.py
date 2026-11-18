"""Selection of observable transitions after the resonance search.

The search methods return every transition they find. This module decides
which of them to keep for the spectrum: the strongest allowed transitions are
always kept, and forbidden transitions are kept only while their intensity is
above a relative threshold.
"""


def truncate_observable(transitions, num_allowed, rel_thresh=1e-3):
    """
    Inputs: transitions (field, jj, kk, intensity); number of allowed
    transitions to keep; relative intensity threshold.

    Returns: the kept transitions, sorted by decreasing intensity magnitude.

    Notes: the ``num_allowed`` strongest transitions are always kept; beyond
    them, a transition is kept only while its intensity exceeds
    ``rel_thresh`` times the maximum, which keeps observable forbidden lines
    in systems with large anisotropy.
    """
    if not transitions:
        return []
    ordered = sorted(transitions, key=lambda t: -abs(t[3]))
    max_int = abs(ordered[0][3])
    if max_int <= 0.0:
        return []
    keep = list(ordered[:num_allowed])
    for t in ordered[num_allowed:]:
        if abs(t[3]) > rel_thresh * max_int:
            keep.append(t)
        else:
            break
    return keep
