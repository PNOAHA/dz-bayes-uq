# MANIFEST — files in this deposit

Curated 2026-07-04. Source directory:
`Projects/dz-unmix-bounds/d107_paper2_addon/` (+ input CSVs from
`Projects/dz-global-db/data/realdata_k1_samples/`).

⚠ The source directory is **shared** with a separate methods paper
(dz-unmix-bounds, identifiability bounds). The `probe_*`, `probe2_*`,
`probe3_*`, `applicability_*`, `sigma_err_*`, `interaction_*`, `bigjob_*`,
`phase3d_*`, `diag_*`, `conv_probe*`, `compute_*`, `merge_*`, `*.slurm`, and
`realdata.*` files belong to **that** paper and are **excluded** here.
`assemble.ps1` copies only the files below.

---

## INCLUDE

### `code/` — GPU Bayesian sampler + reference + validation
| file | description |
|---|---|
| `bayes_gpu.py` | GPU-batched Goodman–Weare ensemble sampler (core method); mirrors first-paper end-members read-only |
| `run_scan_gpu.py` | driver for the 15 × 121 × 100 = 181,500-fit prior×bootstrap scan |
| `validate_gpu.py` | fit-by-fit GPU-vs-emcee equivalence check (run first; PASS) |
| `bayesian_realdata.py` | CPU emcee reference implementation |
| `geo_primitives.py` | map primitives bundled for Fig 1 (copied from the geo-figures toolkit) |

### `analysis/` — posterior-uncertainty decomposition
| file | description |
|---|---|
| `m3_eigen_probe.py` | per-sample posterior covariance eigenstructure (soft-direction diagnosis) |
| `m3_nscaling_probe.py` | within-sample grain-subsampling test of the n^(−1/2) counting-noise law |
| `m3_within_vs_between.py` | law-of-total-variance split (within-fit credible vs between-fit scatter; prior vs bootstrap) |

### `figures/` — one script per manuscript figure
| file | figure | notes |
|---|---|---|
| `Fig1_sample_locations_v1.py` | Fig 1 | sample location map; needs `geo_primitives.py` (bundled in `code/`) |
| `fig_bayes_uncertainty.py` | Fig 2 | ternary + dot-whisker |
| `m3_variance_sources_figure.py` | Fig 3 | variance-source stacked bars |
| `FigY_nmf_comparison.py` | Fig 4 | supervised-NMF vs Bayesian; needs first-paper `dz_unmixing_experiment.py` (DOI 10.5281/zenodo.20371892) |
| `m3_mu_sigma_posterior_figure.py` | Fig 5 | end-member shape prior-vs-posterior |
| `m3_acceptance_diagnostics_figure.py` | Fig 6 | MCMC acceptance diagnostics |

### `data/`
| path | description |
|---|---|
| `grains_bayes_gpu.parquet` | **full 181,500-fit scan** (posterior weight/mu/sigma means+stds+quantiles, acceptance, per fit) — the reviewer truth source (~40 MB) |
| `realdata_k1_samples/*.csv` | 15 input grain-age spectra (age_Ma column; `source_doi` in `#` headers); from dz-global-db |
| `results/m3_eigen_results.csv` | eigenstructure per sample |
| `results/m3_nscaling_results.csv`, `results/m3_nscaling_curves.csv` | n-scaling fit + per-m curves |
| `results/m3_within_vs_between_results.csv`, `results/m3_within_vs_between_soft.csv` | variance decomposition (per weight; soft axis) |
| `results/m3_bootstrap_convergence_table.csv` | Table 2 source (σ_b/σ_100 per sample×weight×b) |
| `results/FigY_nmf_comparison_data.csv` | Fig 4 source (NMF vs Bayesian, 4 NCC samples) |

### `figures_rendered/`
Published PNG + PDF (300 dpi) for all six figures; `Fig1_sample_locations` also as SVG.

### `deps/wheels/` (optional)
`emcee-3.1.6-*.whl`, `scipy-1.14.1-*.whl` — third-party (BSD), for air-gapped install only.

---

## EXCLUDE (belong to the second paper / internal working files)

- All `probe_*.py`, `probe2_*.py`, `probe3_*.py`
- `applicability_*`, `sigma_err_*`, `interaction_*`, `bigjob_*`, `phase3d_*`,
  `diag_*`, `conv_probe*`, `compute_factors.py`, `compute_sandwich.py`,
  `analytic_worst_mode.py`, `judge_rho_exponent.py`, `fig_sigma_err*.py`,
  `merge_*` (except none needed), `realdata_scan*.py`, `validate_sigma_err.py`,
  `multi_init_smoke*`, `_inspect.py`
- Their outputs: `realdata.csv/json`, `applicability_*.csv/json`,
  `interaction.csv/json`, `bigjob.json`, `probe_excess_growth_out.json`
- All `*.slurm`, `done_ids`, `missing_ids`, `paper2_jobids.txt`
- Internal working docs: `PAPER3_bayes_materials.md` (drafting notes),
  `00_RESUME_Bayesian.md`, `zenodo_paper3/` staging itself
