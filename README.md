# Analytical solution for solidification with melt convection

This repository contains analytical and numerical models for one-dimensional
solidification of pure aluminum with melt-convection corrections. The numerical
reference uses a finite-volume enthalpy formulation with Voller-type subcell
front interpolation. The default comparison uses the same mold heat-transfer
coefficient in both models.

## Main programs

- `analytical_model_2024_Al_melt_convection_corrected.py`: main analytical
  model and recommended entry point for analytical-versus-numerical comparison.
- `numerical_stefan_model_pure_al.py`: numerical Stefan model and comparison
  figure generator.
- `publication_studies.py`: supporting verification and parametric studies.
- `help.html`: detailed execution guide in Portuguese.

## Requirements

- Python 3.11 or newer
- NumPy
- SciPy
- Matplotlib

Install the Python dependencies with:

```powershell
py -3.11 -m pip install numpy scipy matplotlib
```

## Run

Recommended analytical entry point:

```powershell
py -3.11 analytical_model_2024_Al_melt_convection_corrected.py
```

Direct numerical execution:

```powershell
py -3.11 numerical_stefan_model_pure_al.py
```

Example changing the common mold coefficient and first thermocouple position:

```powershell
py -3.11 analytical_model_2024_Al_melt_convection_corrected.py --h-bar 5000 --tc-positions 0.003 0.010 0.015 0.030 0.050 0.090
```

The generated analytical-versus-numerical figures and tables are written to
`outputs/numerical_comparison/`.

## Generated comparisons

The tracked baseline results include:

- cooling curves;
- interface position versus time;
- interface velocity versus position;
- liquid-side thermal gradient versus position;
- cooling rate at the fusion temperature versus position.

## Local-only research material

The manuscript, its DOCX versions, publication build artifacts, reference PDFs,
Origin project files, and experimental source material are intentionally ignored
by Git and remain only in the local working directory.

