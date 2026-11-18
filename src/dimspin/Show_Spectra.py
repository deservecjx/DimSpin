"""Plotting helpers for simulated spectra.

Both functions draw on a matplotlib axes and call ``plt.show()``. The
AutoSim entry point uses them when ``show=True``; they can also be called
directly with an existing axes.
"""

import numpy as np


def show_spectrum(spec, spectrum, ax=None):
    """
    Inputs: Spec object; one-dimensional spectrum; optional axes.

    Returns: the axes used, after showing the figure.

    Notes: matplotlib is imported only when this function is called.
    """
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(spec.fieldrange, spectrum, color=spec.linecolor, linewidth=0.5)
    ax.set_xlabel('Magnetic field (G)')
    ax.set_ylabel('Intensity (arb. units)')
    plt.show()
    return ax


def show_rotation(spec, spectra, ax=None):
    """
    Inputs: Spec object; (angle, field) spectrum array; optional axes.

    Returns: the axes used, after showing the figure.

    Notes: the map is drawn with the field on the horizontal axis and the
    rotation angle on the vertical axis; matplotlib is imported only when this
    function is called.
    """
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots()
    angles = np.linspace(spec.anglerange[0], spec.anglerange[1], spectra.shape[0])
    mesh = ax.pcolormesh(spec.fieldrange, angles, spectra, shading='auto', cmap='viridis')
    ax.set_xlabel('Magnetic field (G)')
    ax.set_ylabel('Angle (deg)')
    ax.figure.colorbar(mesh, ax=ax)
    plt.show()
    return ax
