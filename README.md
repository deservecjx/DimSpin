# DimSpin

DimSpin is an EPR/ESR spectrum simulator for spin Hamiltonians and
spin-electric coupling (SEC) tensors (T, R, F), featuring electric-field
modulation (EFM), magnetic-field modulation (MFM), and 2D angle-dependent
single-crystal analysis.

> **Status:** 0.1.0 development. The package skeleton is in place; the physics
> modules are added in subsequent releases.

## Planned scope

- Spin systems with electron spin `S` and nuclear spin `I`
- Zeeman, zero-field splitting, hyperfine and high-order zero-field terms
- Magnetic-field modulation, electric-field modulation and parameter modulation
- Single-crystal one-side, single-crystal rotation and powder modes
- Resonance search with `High_Field_Method` and `Adaptive_Grid_Method`
- Voigt/Lorentzian/Gaussian line broadening, parallel evaluation and energy-level plots

`Fused_Method` is not included in this release.

## Development setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

## Tests

```bash
pytest
```

## License

MIT. See [LICENSE](LICENSE).

## Citation

If you use DimSpin in your work, please cite it. Citation metadata is
provided in [CITATION.cff](CITATION.cff).

## Third-party data

The space-group operations in `dimspin.Space_Group` are generated with
[spglib](https://spglib.readthedocs.io) (BSD-3 license). Please cite
A. Togo, K. Shinohara and I. Tanaka, *Sci. Technol. Adv. Mater. Methods* 4,
2384822 (2024), doi:10.1080/27660400.2024.2384822.
