# -*- coding: utf-8 -*-
"""M3 first step: posterior-weight covariance eigenstructure per sample.

Reads grains_bayes_gpu.parquet (181,500 fits, 15 samples x 121 priors x 100
bootstraps) and, per sample, eigendecomposes the 3x3 covariance of
(w_S1, w_S2, w_S3) posterior means across the scan grid.

  - lam1/lam2         : degeneracy strength (anisotropy in the simplex plane)
  - angle_deg         : angle between principal eigenvector and the S1-S2
                        edge direction (1,-1,0)/sqrt(2); 0 deg = "degeneracy
                        direction = ill-conditioned direction" confirmed
  - corr_w1w2         : sanity check (strong degeneracy -> corr ~ -1)

Cross-checks (zero compute, analytical):
  - M1 Gram anchor: Gaussian overlap integrals of the reference endmembers
    S1(130,15) S2(280,30) S3(1900,100); weakest eigenvector of the
    normalized Gram = predicted ill-conditioned direction.
  - rho scaling law sigma_err = C/(sqrt(N)*rho): rho is shared by all samples
    (same endmember triple), so sqrt(lam1) should scale ~ n^(-1/2). Tested on
    the bootstrap-only spread at the central prior (mean_shift=sigma_shift=0),
    which isolates data-resampling variance from prior-perturbation variance.

NOT part of the M1 probe2_* series (those eigendecompose Gram/Toeplitz
matrices for the excess(K) derivation, not Bayesian posteriors).
"""
import numpy as np
import pandas as pd

PARQUET = "grains_bayes_gpu.parquet"
W_COLS = ["w_S1_mean", "w_S2_mean", "w_S3_mean"]
# S1-S2 edge direction in weight space: trade w1 vs w2, w3 fixed
E_S1S2 = np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0)
# reference endmembers (mu, sigma) from the first paper / Methods draft
ENDMEMBERS = [(130.0, 15.0), (280.0, 30.0), (1900.0, 100.0)]


def eig_summary(W):
    """Covariance eigenstructure of an (N,3) weight-mean cloud."""
    C = np.cov(W.T)
    lam, V = np.linalg.eigh(C)          # ascending
    lam, V = lam[::-1], V[:, ::-1]      # descending: lam1 >= lam2 >= lam3
    v1 = V[:, 0]
    cosang = min(1.0, abs(float(v1 @ E_S1S2)))
    angle = np.degrees(np.arccos(cosang))
    corr = np.corrcoef(W[:, 0], W[:, 1])[0, 1]
    return lam, v1, angle, corr


def gram_gaussian(endmembers):
    """Analytical Gram of Gaussian endmember densities: G_ij = integral N_i N_j."""
    K = len(endmembers)
    G = np.empty((K, K))
    for i, (mi, si) in enumerate(endmembers):
        for j, (mj, sj) in enumerate(endmembers):
            s2 = si**2 + sj**2
            G[i, j] = np.exp(-0.5 * (mi - mj) ** 2 / s2) / np.sqrt(2 * np.pi * s2)
    return G


def main():
    df = pd.read_parquet(PARQUET)
    dev = (df[W_COLS].sum(axis=1) - 1.0).abs().max()
    print(f"simplex check: max |sum(w)-1| = {dev:.2e}")

    rows = []
    for name, g in df.groupby("sample_name"):
        parts = name.split("__")
        group, short = parts[0], parts[-1]
        n = int(g["n_zircons"].iloc[0])

        # full-grid covariance (prior perturbation + bootstrap combined)
        lam, v1, angle, corr = eig_summary(g[W_COLS].to_numpy())

        # bootstrap-only spread at central prior config -> data variance
        central = g[(g["mean_shift"] == 0.0) & (g["sigma_shift"] == 0.0)]
        lam_c, _, angle_c, _ = eig_summary(central[W_COLS].to_numpy())

        rows.append(dict(
            sample=short, group=group, n=n,
            lam1=lam[0], lam2=lam[1], lam3=lam[2],
            aniso=lam[0] / lam[1],
            soft_std=np.sqrt(lam[0]),
            angle_deg=angle,
            v1_w1=v1[0], v1_w2=v1[1], v1_w3=v1[2],
            corr_w1w2=corr,
            soft_std_boot=np.sqrt(lam_c[0]),
            angle_deg_boot=angle_c,
            n_central=len(central),
        ))

    res = pd.DataFrame(rows).sort_values(["group", "sample"]).reset_index(drop=True)
    res.to_csv("m3_eigen_results.csv", index=False)

    with pd.option_context("display.width", 200, "display.max_columns", None):
        print("\n=== per-sample eigenstructure (full 121x100 grid) ===")
        cols = ["sample", "group", "n", "lam1", "lam2", "lam3", "aniso",
                "soft_std", "angle_deg", "corr_w1w2",
                "soft_std_boot", "angle_deg_boot"]
        print(res[cols].to_string(index=False,
              float_format=lambda x: f"{x:.4g}"))

    print("\n=== summary ===")
    print(f"angle to S1-S2 edge : median {res.angle_deg.median():.1f} deg, "
          f"max {res.angle_deg.max():.1f} deg (n={len(res)})")
    print(f"anisotropy lam1/lam2: median {res.aniso.median():.1f}, "
          f"range {res.aniso.min():.1f} - {res.aniso.max():.1f}")
    print(f"corr(w1,w2)         : median {res.corr_w1w2.median():.3f}")

    # --- M1 Gram anchor -------------------------------------------------
    G = gram_gaussian(ENDMEMBERS)
    Gn = G / np.sqrt(np.outer(np.diag(G), np.diag(G)))   # cosine-normalized
    for label, M in (("raw", G), ("normalized", Gn)):
        lam, V = np.linalg.eigh(M)
        vweak = V[:, 0]                                   # smallest eigenvalue
        ang = np.degrees(np.arccos(min(1.0, abs(float(vweak @ E_S1S2)))))
        print(f"\nGram ({label}): kappa = {lam[-1]/lam[0]:.3g}, "
              f"weakest eigvec = {np.round(vweak, 3)}, "
              f"angle to S1-S2 edge = {ang:.1f} deg")

    # --- n-scaling check (rho shared across samples) --------------------
    for label, col in (("full grid", "soft_std"), ("bootstrap-only", "soft_std_boot")):
        x, y = np.log(res["n"].to_numpy(float)), np.log(res[col].to_numpy())
        slope, icept = np.polyfit(x, y, 1)
        r = np.corrcoef(x, y)[0, 1]
        print(f"n-scaling ({label:14s}): slope {slope:+.2f} (predict -0.50), "
              f"r = {r:+.2f}")


if __name__ == "__main__":
    main()
