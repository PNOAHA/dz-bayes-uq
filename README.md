# Bayesian detrital-zircon unmixing with full uncertainty quantification — code & data

Supplementary code and data for the manuscript *"Which mixing weights can you
trust? GPU-accelerated Bayesian uncertainty quantification for detrital-zircon
provenance unmixing of Lower Cretaceous samples from NE Asia"* (Pan Lujia,
submitted to *Computers & Geosciences*).

This deposit lets a reader (i) re-analyse the full posterior scan without
re-running the 181,500-fit computation, and (ii) reproduce every figure and
table in the paper from the released data.

Each tagged release of this repository is archived on **Zenodo** with a citable
DOI.

---

## What is here

```
├── code/            GPU Bayesian sampler + emcee reference + equivalence check
├── analysis/        posterior-uncertainty decomposition (eigenstructure, n-scaling, within-vs-between)
├── figures/         one script per manuscript figure
├── data/
│   ├── grains_bayes_gpu.parquet      full 181,500-fit scan (reviewer truth source, ~40 MB)
│   ├── realdata_k1_samples/          15 input grain-age spectra (with source DOIs in headers)
│   └── results/                      per-figure derived tables (CSV)
├── figures_rendered/                 the figures as published (PNG + PDF, 300 dpi)
└── deps/wheels/                      optional offline wheels for air-gapped install
```

See `MANIFEST.md` for the exact file list and one-line descriptions.

---

## Reproducing the analysis (CPU, minutes)

Everything except the original GPU scan reads the released
`grains_bayes_gpu.parquet` and runs on a CPU in seconds to minutes.

```bash
python -m venv venv && source venv/bin/activate      # or conda
pip install -r requirements.txt

# re-derive the uncertainty decomposition tables
python analysis/m3_eigen_probe.py
python analysis/m3_nscaling_probe.py
python analysis/m3_within_vs_between.py

# re-render the figures (each writes PNG+PDF next to itself)
python figures/fig_bayes_uncertainty.py            # Fig 2
python figures/m3_variance_sources_figure.py       # Fig 3
python figures/m3_mu_sigma_posterior_figure.py     # Fig 5
python figures/m3_acceptance_diagnostics_figure.py # Fig 6
python figures/FigY_nmf_comparison.py              # Fig 4  (see cross-deps below)
python figures/Fig1_sample_locations_v1.py         # Fig 1  (see cross-deps below)
```

The figure scripts expect `grains_bayes_gpu.parquet` and
`realdata_k1_samples/` to sit where `MANIFEST.md` places them (paths are
resolved relative to each script); adjust the two module-level path constants
at the top of a script if you relocate the data.

### Cross-dependencies (two figures)

- **Fig 1** (`Fig1_sample_locations_v1.py`) imports map primitives from
  `geo_primitives.py`; a copy is bundled in `code/` by the assembler. It is
  self-contained matplotlib (no GIS stack).
- **Fig 4** (`FigY_nmf_comparison.py`) applies the *first paper's* supervised-NMF
  routine to the same real samples. It imports `dz_unmixing_experiment.py` from
  the first paper's repository, https://github.com/PNOAHA/dz-unmix-cg
  (preprint: https://ssrn.com/abstract=6955308); place that file on the Python
  path to run Fig 4. (The first paper's end-members are identical to those used
  here.)

## Reproducing the original GPU scan (optional, GPU)

The 181,500-fit scan was produced by `code/run_scan_gpu.py`, which batches
`code/bayes_gpu.py` (a GPU re-implementation of the Goodman–Weare ensemble
sampler) over the 15 samples × 121 prior configurations × 100 bootstraps.
Its numerical equivalence to the CPU emcee reference (`code/bayesian_realdata.py`)
is established fit-by-fit by `code/validate_gpu.py` (**run this first; verdict
PASS**). The GPU sampler requires a CUDA GPU and the GPU array library imported
at the top of `bayes_gpu.py`; the CPU reference requires only `emcee`. Bootstrap
resampling is made deterministic by a SHA-1-derived seed (`boot_seed`), so the
scan is exactly reproducible.

---

## Dependencies

See `requirements.txt`. Core: `numpy`, `pandas`, `pyarrow`, `matplotlib`,
`scipy`, `emcee`, `scikit-learn`. Offline `.whl` files for `emcee` and `scipy`
are provided under `deps/wheels/` for air-gapped machines (e.g. the compute
cluster used here); install with `pip install --no-index --find-links deps/wheels ...`.

---

## Data provenance and licence

- **Code** (`code/`, `analysis/`, `figures/`) is released under the **MIT
  licence** — see `LICENSE`.
- **Data** (`data/`, `figures_rendered/`) are released under **CC-BY-4.0** —
  see `LICENSE_DATA.md`.
- **Input grain ages** (`data/realdata_k1_samples/*.csv`) are compiled from two
  open per-grain detrital-zircon databases — the OneDZ compilation and the
  Puetz database — both distributed under **CC-BY**. Each sample CSV carries
  its `source_doi` in the `#` header; please retain these on reuse and cite the
  original sources.

## Citation

See `CITATION.cff` for the preferred citation (the manuscript) and the archive
citation. Related deposits: first paper (supervised NMF method) —
https://github.com/PNOAHA/dz-unmix-cg, preprint
https://ssrn.com/abstract=6955308; second paper (identifiability bounds) —
doi:10.5281/zenodo.20768132, https://github.com/PNOAHA/dz-unmix-bounds.
