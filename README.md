# Analytical solution for solidification with melt convection

This repository contains analytical and numerical models for one-dimensional
solidification of pure aluminum with melt-convection corrections. The numerical
reference uses a finite-volume enthalpy formulation with Voller-type subcell
front interpolation. The default comparison uses the same mold heat-transfer
coefficient in both models.

## Clone the repository

Using HTTPS:

```powershell
git clone https://github.com/ileaof/analytical-solution-for-solidification-with-melt-convection.git
cd analytical-solution-for-solidification-with-melt-convection
```

Using SSH, if an SSH key is already configured in GitHub:

```powershell
git clone git@github.com:ileaof/analytical-solution-for-solidification-with-melt-convection.git
cd analytical-solution-for-solidification-with-melt-convection
```

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

## Published paper

The published article is available locally as `ajeassp.2026.88.116.pdf`. The PDF
is intentionally not tracked in this repository; use the publisher links below
to access and cite the version of record.

> Santos Júnior, G. E. M., Rocha, F. S., Silva, A. B. S., Carmo, D. A. R.,
> Silva, M. O., & Ferreira, I. L. (2026). On a Novel Closed-Form Analytical
> Solution for Unsteady Solidification: Theory and Application. *American
> Journal of Engineering and Applied Sciences*, 19(1), 88-116.
> https://doi.org/10.3844/ajeassp.2026.88.116

- [Article page](https://u.thescipub.com/abstract/ajeassp.2026.88.116)
- [Publisher PDF](https://thescipub.com/pdf/ajeassp.2026.88.116.pdf)

## License

The source code and repository documentation are distributed under the
[MIT License](LICENSE). The published paper is a separate scholarly work and is
distributed by the publisher under the Creative Commons Attribution license
(CC BY); the MIT license does not replace the article's publication license.

