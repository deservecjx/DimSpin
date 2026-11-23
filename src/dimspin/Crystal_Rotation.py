"""Single-crystal rotation spectra.

The crystal is turned about the laboratory x axis; the initial orientation is
given by the spin's crystal orientation. Each angle is an independent
spectrum, computed in parallel over (spin index, angle) tasks.
"""

from math import pi

import numpy as np

from .Parallel import parallel_map
from .Rodrigues_Rotation import rotation_matrix
from .Spec_Oneside import spec_oneside

_worker_spins = None
_worker_spec = None


def _init_worker(spins, spec):
    global _worker_spins, _worker_spec
    _worker_spins = spins
    _worker_spec = spec


def _angle_task(spin_angle):
    spin_idx, angle = spin_angle
    return _one_angle(_worker_spins[spin_idx], _worker_spec, angle)


def _one_angle(spin, spec, angle):
    """
    Inputs: Spin and Spec objects; rotation angle in degrees.

    Returns: the spectrum at that angle.

    Notes: spin and spec are copied so that parallel workers never share
    mutable state.
    """
    spin = spin.clone()
    spec = spec.clone()
    radian = angle * pi / 180
    axis = np.array([1, 0, 0])
    Rotangle = rotation_matrix(axis, radian)
    spin.R_tf2lf_rotated = Rotangle @ spin.R_tf2lf
    spin.R_etf2lf_rotated = Rotangle @ spin.R_etf2lf
    spin.R_dtf2lf_rotated = Rotangle @ spin.R_dtf2lf
    if spec.modtype == 'Emod':
        spec.E1 = spec.Emod * np.dot(Rotangle, spec.Edirection)
    return spec_oneside(spin, spec)


def crystal_rotation(spins, spec, angnum, parallel=True):
    """
    Inputs: list of Spin objects; Spec object; number of angles; whether to
    use a process pool.

    Returns: an array of shape (len(spins), angnum, specpoints).

    Notes: tasks are (spin index, angle) pairs, so the parallel load does not
    depend on the number of spins. The angles are evenly spaced over
    ``spec.anglerange``.
    """
    n_spins = len(spins)
    anglerange = np.linspace(spec.anglerange[0], spec.anglerange[1], angnum)
    tasks = [(si, angle) for si in range(n_spins) for angle in anglerange]
    if parallel:
        rows = parallel_map(_angle_task, tasks, initializer=_init_worker,
                            initargs=(spins, spec), max_workers=spec.max_workers)
    else:
        rows = (_one_angle(spins[si], spec, angle) for si, angle in tasks)
    spectra = np.zeros((n_spins, angnum, spec.specpoints))
    for k, ((si, angle), row) in enumerate(zip(tasks, rows)):
        spectra[si, k % angnum, :] = row
    return spectra
