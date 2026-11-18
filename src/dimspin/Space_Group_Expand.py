"""Expand one spin centre into the magnetically inequivalent orientations of a crystal.

Given a representative spin centre and a space group number, the space group
maps the centre onto its orbit. Each site in the orbit rotates the molecular
frame by

    R_cart = L @ R @ L^-1,

with R the integer fractional-coordinate rotation of that site and L the
standard lattice matrix. The resulting ``R_cart`` is the molecular
orientation used when assembling tensors.

Output: ``(orientations, multiplicities)``. The total spectrum is the sum of
``multiplicity[i] * spectrum(orientations[i])``.

Two sites with identical (g, D, A) tensors in the crystal frame give the same
spectrum for every crystal orientation and are merged into one magnetic
equivalence class, weighted by its multiplicity. For a special position the
operations are first grouped by the position they produce, so the stabilizer
is not counted more than once.

Notes: only second-order tensors g, D and A are transformed. High-order ZFS
coefficients and the third-order T/R/F tensors are not rotated here. The
molecular frame of the reference site is assumed to coincide with the crystal
frame; any fixed tilt belongs to the tensor orientation. Fractional
coordinates must use the same standard setting as the operations: unique axis
b for monoclinic groups and hexagonal axes for rhombohedral groups.
"""

from __future__ import annotations

import warnings

import numpy as np

from .Space_Group import get_space_group
from .Space_Group_L import get_L


def _round_key(arr, tol):
    """
    Inputs: array; rounding tolerance.

    Returns: a hashable tuple of the rounded values.
    """
    n = max(0, int(round(-np.log10(tol))))
    return tuple(np.round(np.asarray(arr, dtype=float), n).ravel().tolist())


def _group_by(items, keyfunc):
    """
    Inputs: items; key function.

    Returns: groups of items, preserving first-appearance order of the keys.
    """
    groups = {}
    order = []
    for it in items:
        k = keyfunc(it)
        if k not in groups:
            groups[k] = []
            order.append(k)
        groups[k].append(it)
    return [groups[k] for k in order]


def build_space_group_orientations(ref_spin, group_number, frac_pos=None, tol=1e-8):
    """
    Inputs: representative spin centre; space group number (1-230); optional
    fractional coordinates of the site; grouping tolerance.

    Returns: (orientations, multiplicities), with each orientation a 3x3
    orthogonal matrix giving the molecular frame relative to the crystal
    frame.

    Notes: the spin centre's tensors are normalized in place by its
    ``identity_tensor`` method. Without ``frac_pos`` every symmetry operation
    is treated as a distinct site (general position); with ``frac_pos`` the
    operations are grouped by the position they produce, which handles
    special positions correctly. A mirror, inversion or glide operation gives
    det(R_cart) = -1; second-order tensors transform as R @ T @ R.T for either
    sign.
    """
    number = int(group_number)
    sg = get_space_group(number)
    L = get_L(number)

    ref_spin.identity_tensor()

    g = np.asarray(ref_spin.g, dtype=float)
    D = np.asarray(ref_spin.D, dtype=float)
    A = np.asarray(ref_spin.A, dtype=float)

    x0 = None if frac_pos is None else np.asarray(frac_pos, dtype=float)

    entries = []
    for op in sg.operations:
        R = np.asarray(op.rotation, dtype=float)
        t = np.array([float(v) for v in op.translation])
        R_cart = L @ R @ np.linalg.inv(L)
        pos = (R @ x0 + t) % 1.0 if x0 is not None else None
        entries.append({
            "pos": pos,
            "R_cart": R_cart,
            "g": R_cart @ g @ R_cart.T,
            "D": R_cart @ D @ R_cart.T,
            "A": R_cart @ A @ R_cart.T,
        })

    if x0 is not None:
        def _site_key(e):
            pos = np.where(e["pos"] >= 1.0 - tol, 0.0, e["pos"])
            return _round_key(pos, tol)

        site_groups = _group_by(entries, _site_key)
        for grp in site_groups:
            g0 = grp[0]["g"]
            for e in grp[1:]:
                if not np.allclose(g0, e["g"], atol=tol):
                    warnings.warn(
                        "The tensors are inconsistent with the site symmetry: "
                        "operations that map the site onto itself give different "
                        "tensors. Check that the representative tensors satisfy "
                        "the local symmetry of this Wyckoff position."
                    )
                    break
        entries = [grp[0] for grp in site_groups]

    def _tensor_key(e):
        return (_round_key(e["g"], tol), _round_key(e["D"], tol), _round_key(e["A"], tol))

    classes = _group_by(entries, _tensor_key)

    orientations = [cls[0]["R_cart"] for cls in classes]
    multiplicities = [len(cls) for cls in classes]

    return orientations, multiplicities


__all__ = ["build_space_group_orientations"]
