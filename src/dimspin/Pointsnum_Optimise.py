"""Estimate the number of field points needed for a spectrum."""


def get_points_spectrum(fieldRange, lw_lorentzian, lw_gaussian):
    """Return a point count giving roughly five samples per linewidth."""
    # Use the smaller nonzero linewidth as the resolution scale.
    if lw_lorentzian != 0 and lw_gaussian != 0:
        lw = min(lw_lorentzian, lw_gaussian)
    elif lw_lorentzian != 0:
        lw = lw_lorentzian
    elif lw_gaussian != 0:
        lw = lw_gaussian
    else:
        lw = 1.0

    width = fieldRange[-1] - fieldRange[0]
    return max(int(5 * width / lw), 100)
