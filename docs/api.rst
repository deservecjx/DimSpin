API reference
=============

Core objects
------------

.. autoclass:: dimspin.Class_Spin.Spin
   :members:

.. autoclass:: dimspin.Class_Spec.Spec
   :members:

Entry points
------------

.. autofunction:: dimspin.AutoSim.autosim

.. autofunction:: dimspin.Levels_Plot.plot_energy_levels

.. autofunction:: dimspin.Show_Spectra.show_spectrum

.. autofunction:: dimspin.Show_Spectra.show_rotation

Resonance search
----------------

.. autofunction:: dimspin.High_Field_Method.high_field_method

.. autofunction:: dimspin.Adaptive_Grid_Method.adaptive_grid_method

.. autofunction:: dimspin.Eigenfield_Method.eigenfield_method

.. autofunction:: dimspin.Root_Finder.transition_probability

.. autofunction:: dimspin.Root_Finder.population_difference

.. autofunction:: dimspin.Truncate_Observable.truncate_observable

Hamiltonian terms
-----------------

.. autofunction:: dimspin.Team_Zeeman.H_Zeeman

.. autofunction:: dimspin.Team_ZFS.H_ZFS

.. autofunction:: dimspin.Team_Hyperfine.H_Hyperfine

.. autofunction:: dimspin.Team_ZFS_HighOrder.H_ZFS_HighOrder

.. autofunction:: dimspin.Team_ZFS_HighOrder.B_arrays_to_dict

.. autofunction:: dimspin.Steven_Operator.stev_op

Electric-field modulation
-------------------------

.. autofunction:: dimspin.ETeam_Zeeman.H_EMod_T

.. autofunction:: dimspin.ETeam_ZFS.H_EMod_R

.. autofunction:: dimspin.ETeam_Hyperfine.H_EMod_F

Tensors and rotations
---------------------

.. autofunction:: dimspin.Tensor.gtensor

.. autofunction:: dimspin.Tensor.Dtensor

.. autofunction:: dimspin.Tensor.Atensor

.. autofunction:: dimspin.Tensor.Tijk2Txyz

.. autofunction:: dimspin.Tensor.unpack_tensor_3x6

.. autofunction:: dimspin.Tensor.unpack_tensor_3x5

.. autofunction:: dimspin.Rodrigues_Rotation.rotation_matrix

.. autofunction:: dimspin.Eular2Rot.eular2rotzyz

.. autofunction:: dimspin.Eular2Rot.resolve_orientation

Line broadening
---------------

.. autofunction:: dimspin.Line_Broadening.line_shape

.. autofunction:: dimspin.Line_Broadening.broaden_direct

.. autofunction:: dimspin.Line_Broadening.broaden_fft

.. autofunction:: dimspin.Line_Broadening.broaden

Space groups
------------

.. autoclass:: dimspin.Space_Group.SymmetryOperator
   :members:

.. autoclass:: dimspin.Space_Group.SpaceGroup
   :members:

.. autofunction:: dimspin.Space_Group.get_space_group

.. autofunction:: dimspin.Space_Group_L.get_L

.. autofunction:: dimspin.Space_Group_L.get_lattice_info

.. autofunction:: dimspin.Space_Group_Expand.build_space_group_orientations

Parallel evaluation
-------------------

.. autofunction:: dimspin.Parallel.parallel_map

.. autofunction:: dimspin.Parallel.check_worker_count
