"""Single entry point for spectrum simulations.

``autosim`` takes one or more Spin objects and one Spec object and returns
the spectrum: a one-dimensional array for the ``Crystal_Oneside`` and
``Powder`` samples, and a two-dimensional (angle, field) array for
``Crystal_Rotation``. Plotting is opt-in through ``show=True`` and is done by
the helpers in Show_Spectra.
"""

import numpy as np

from .Class_Spec import Spec
from .Class_Spin import Spin


def autosim(*class_name, show=True, parallel=True):
    """
    Inputs: one or more Spin objects followed by one Spec object; whether to
    plot the result; whether to use a process pool.

    Returns: the spectrum array (see the module docstring for its shape).

    Notes: when ``Spec.group_number`` is set, each Spin is expanded into the
    magnetically inequivalent orientations of the space group and the spectra
    are summed with their multiplicities. Parameter checks and the field grid
    expansion are done by ``Spec.check_simpara``. With ``show=True`` all three
    sample types are drawn: a line for one-side and powder, an angle-field map
    for rotation.
    """
    if len(class_name) < 2:
        raise ValueError("Autosim requires at least one Spin and one Spec to be entered as parameters")
    spec = class_name[-1]
    if not isinstance(spec, Spec):
        raise ValueError("The last parameter must be a Spec class!")
    spec.check_simpara()

    spins = []
    multiplicity = []
    for item in class_name[:-1]:
        if isinstance(item, Spec):
            raise ValueError("Cannot accept two independent experimental conditions! "
                             "You can only transfer one 'Spec' class to autosim!")
        if not isinstance(item, Spin):
            raise ValueError("Cannot accept other classes! You can only transfer "
                             "'Spin' and 'Spec' classes to autosim!")
        if spec.group_number is not None:
            from .Space_Group_Expand import build_space_group_orientations
            orientations, multiplicities = build_space_group_orientations(
                item, spec.group_number, spec.frac_pos)
            for R_cart, m in zip(orientations, multiplicities):
                sj = item.clone()
                sj.identity_tensor(spec.crystal_orientation, R_cart)
                spins.append(sj)
                multiplicity.append(m)
        else:
            sj = item.clone()
            sj.identity_tensor(spec.crystal_orientation, np.eye(3))
            spins.append(sj)
            multiplicity.append(1)

    if spec.sample == 'Crystal_Oneside':
        spectrum = _oneside(spins, multiplicity, spec, parallel)
        if show:
            from .Show_Spectra import show_spectrum
            show_spectrum(spec, spectrum)
        return spectrum

    if spec.sample == 'Crystal_Rotation':
        from .Crystal_Rotation import crystal_rotation
        angnum = int((spec.anglerange[1] - spec.anglerange[0]) / spec.anglerange[2]) + 1
        spectra = crystal_rotation(spins, spec, angnum, parallel=parallel)
        result = np.zeros((angnum, spec.specpoints))
        for i in range(len(spins)):
            result = result + multiplicity[i] * spectra[i]
        if show:
            from .Show_Spectra import show_rotation
            show_rotation(spec, result)
        return result

    if spec.sample == 'Powder':
        from .Powder import powder
        spectra = powder(spins, spec, parallel=parallel)
        result = np.zeros(spec.specpoints)
        for i in range(len(spins)):
            result = result + multiplicity[i] * spectra[i]
        if show:
            from .Show_Spectra import show_spectrum
            show_spectrum(spec, result)
        return result


def _oneside(spins, multiplicity, spec, parallel):
    """
    Inputs: expanded Spin list; multiplicities; Spec object; parallel flag.

    Returns: the multiplicity-weighted sum of the one-side spectra.
    """
    from .Spec_Oneside import spec_oneside
    if len(spins) == 1:
        return multiplicity[0] * spec_oneside(spins[0], spec)

    from functools import partial

    from .Parallel import parallel_map
    if parallel:
        results = parallel_map(partial(spec_oneside, spec=spec), spins, max_workers=spec.max_workers)
    else:
        results = (spec_oneside(spin, spec) for spin in spins)
    spectrum = np.zeros(spec.specpoints)
    for i, res in enumerate(results):
        spectrum = spectrum + multiplicity[i] * res
    return spectrum

