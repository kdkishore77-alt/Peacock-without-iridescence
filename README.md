# Disorder-Driven Photonic Homogenization in Peacock Feather Barbules

Python implementation of the effective-medium/transfer-matrix model used in
*"Designing angle-robust structural colour through disorder: a bioinspired
principle from peacock feathers."* Reproduces all figures and numerical
results in the manuscript for Region 4 (and comparative Regions 1–6) of
peacock tail feather barbules.

## Contents

| File | Description |
|---|---|
| `peacock_optics.py` | Core physics: material dispersion, EMT mixing rule, transfer-matrix reflectance, disordered/continuum stack builders, CIE 1931 colorimetry, sRGB conversion. |
| `generate_figures.py` | Generates all manuscript figures (Figs. 1–7) and console-printed numerical results; entry point for reproduction. |
| `fig1_*.png` – `fig7_*.png`, `fig_coherent_incoherent_split.png` | Output figures (regenerated on each run; not tracked as "source"). |
| `*.csv` / `*.npz` *(if present)* | Cached intermediate spectra/chromaticity arrays for figures that are expensive to regenerate. |

## Requirements

```
python >= 3.10
numpy, matplotlib
```

Install via:
```bash
pip install numpy matplotlib
```

## Usage

```bash
python generate_figures.py
```

Runs in **full manuscript-resolution mode** by default (`FAST = False`,
several minutes runtime: N=100 disorder realizations, 351-point wavelength
grid, 10-step σ sweep for the tradeoff analysis). For a quick correctness
check (~1–2 min), set `FAST = True` at the top of `generate_figures.py`.

All figures are saved as `.png` in the working directory; key numerical
results (peak reflectance, chromaticity standard deviations, stabilization
factor 𝒮, η(σ) fit statistics, diffuse fraction) are printed to stdout.

## Figure ↔ Result Map

| Figure | Script function | Key quantity |
|---|---|---|
| Fig. 1 | `figure1()` | Ordered multilayer reflectance spectrum, Region 4 |
| Fig. 2 | `figure2_strict()` | Disorder-ensemble-averaged reflectance (N=100) |
| Fig. 3 | `figure3()` | Continuum (EMT) reflectance spectrum |
| Fig. 4 | `figure4()` | Angle-averaged (0°–40°) continuum reflectance |
| Fig. 5 | `figure5()` | Chromaticity: ordered angular spread, disorder-ensemble spread, EMT angular spread, stabilization factor 𝒮 |
| Fig. 6 | `figure6()` | sRGB palette across feather Regions 1–6 |
| Fig. 7 | `disorder_iridescence_tradeoff()` | η(σ) disorder–iridescence tradeoff, 1/σ fit diagnostic |
| — | `coherent_incoherent_check()` | Coherent/diffuse reflectance decomposition |

## Known Model Limitations

- Disorder is implemented as independent, per-layer melanosome/air-channel
  **diameter** perturbations only; lateral positional/connectivity disorder
  (documented by TEM) is **not** represented, since the underlying model is
  a 1D transfer-matrix/EMT treatment (laterally homogeneous per depth slice).
  Reported stabilization factors should be read as lower bounds.
- Region 2 and 4 sRGB swatches (Fig. 6) are computed from **disorder-averaged,
  normal-incidence** spectra (not angle-averaged); Regions 1, 3, 5, 6 use
  representative observed colors and are not independent model outputs.
- The η(σ) vs. 1/σ fit in Fig. 7 (R² ≈ 0.997) may be partly a normalization
  artifact of η's definition; treat the functional form as provisional
  pending full-wave (FDTD) cross-validation.

## Citation

If you use this code, please cite the associated manuscript (see repository
`CITATION.cff` or the paper's DOI once assigned).

## License

Specify a license (e.g. MIT) before making the repository public.
