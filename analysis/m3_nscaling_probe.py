# -*- coding: utf-8 -*-
"""M3 step 2: clean within-sample test of the n-scaling law sigma_soft ~ n^(-1/2).

The cross-sample regression in m3_eigen_probe.py was confounded (simplex
position dominates over n). Clean test: per sample, bootstrap-subsample the
raw grain ages at varying size m and refit weights each time; the spread of
the refit weights along the soft direction should scale as m^(-1/2).

Fitting paradigm: supervised EM with FIXED endmember shapes — the same
setting in which M1's sigma_err = C/(sqrt(N)*rho) law was derived (and in
which _proto_danger_boundary.py measured rho_S1S2(KS) ~ 0.999). Comparing
the EM soft direction/magnitude at full n against the Bayesian posterior
(m3_eigen_results.csv, shapes co-inferred +-10%/20%) separates "counting
noise" from "shape-co-inference degeneracy".

Raw ages: Projects/bayes_gpu/realdata_k1_samples/*.csv (age_Ma column,
'#' comment headers). Zero cluster compute; runs local in seconds.
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS_DIR = ROOT / "data" / "results"
CSV_DIR = ROOT / "data" / "realdata_k1_samples"

ENDMEMBERS = np.array([[130.0, 15.0], [280.0, 30.0], [1900.0, 100.0]])
E_S1S2 = np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0)
B = 400          # bootstrap resamples per subsample size
N_SIZES = 6      # geometric grid of subsample sizes from M_MIN to n
M_MIN = 10
EM_ITERS = 300
SEED = 20260704


def log_dens(a):
    """[..., m] ages -> [..., m, 3] log Gaussian densities, fixed endmembers."""
    mu, sig = ENDMEMBERS[:, 0], ENDMEMBERS[:, 1]
    z = (a[..., None] - mu) / sig
    return -0.5 * z**2 - np.log(sig) - 0.5 * np.log(2 * np.pi)


def em_weights(a_batch):
    """Batched EM for mixture weights, fixed components. a_batch: [B, m] -> [B, 3].

    Log-likelihood is concave in w on the simplex, so EM reaches the
    global maximum regardless of init.
    """
    lp = log_dens(a_batch)                       # [B, m, 3]
    lp = lp - lp.max(axis=-1, keepdims=True)     # stabilize: relative densities
    L = np.exp(lp)
    w = np.full((a_batch.shape[0], 3), 1.0 / 3.0)
    for _ in range(EM_ITERS):
        R = w[:, None, :] * L                    # [B, m, 3]
        R /= R.sum(axis=-1, keepdims=True)
        w_new = R.mean(axis=1)
        if np.abs(w_new - w).max() < 1e-10:
            w = w_new
            break
        w = w_new
    return w


def soft_axis(W):
    """[B,3] weight fits -> (sigma_soft, v1) from covariance eigenstructure."""
    lam, V = np.linalg.eigh(np.cov(W.T))
    return np.sqrt(lam[-1]), V[:, -1]


def angle_deg(v, e):
    return np.degrees(np.arccos(min(1.0, abs(float(v @ e)))))


def main():
    rng = np.random.default_rng(SEED)
    bayes = pd.read_csv(RESULTS_DIR / "m3_eigen_results.csv").set_index("sample")

    rows, curves = [], []
    for csv in sorted(CSV_DIR.glob("*.csv")):
        ages = pd.read_csv(csv, comment="#")["age_Ma"].to_numpy(float)
        n = len(ages)
        parts = csv.stem.split("__")
        group, short = parts[0], parts[-1]

        # full-data supervised point estimate (reference)
        w_full = em_weights(ages[None, :])[0]

        sizes = np.unique(np.round(np.geomspace(M_MIN, n, N_SIZES)).astype(int))
        sig_soft, axes = [], []
        for m in sizes:
            sub = rng.choice(ages, size=(B, m), replace=True)
            s, v1 = soft_axis(em_weights(sub))
            sig_soft.append(s)
            axes.append(v1)
            curves.append(dict(sample=short, m=int(m), sigma_soft=s,
                               angle_S1S2=angle_deg(v1, E_S1S2)))
        sig_soft = np.array(sig_soft)

        slope, _ = np.polyfit(np.log(sizes), np.log(sig_soft), 1)
        r = np.corrcoef(np.log(sizes), np.log(sig_soft))[0, 1]

        # direction at largest m: vs S1-S2 edge and vs Bayesian principal axis
        v_top = axes[-1]
        b = bayes.loc[short]
        v_bayes = np.array([b.v1_w1, b.v1_w2, b.v1_w3])
        # multinomial counting-statistics floor along the observed soft axis:
        # perfectly separable endmembers -> grain labels are multinomial(w),
        # cov(w_hat) = (diag(w) - w w^T)/n; project onto v_top
        S_multi = (np.diag(w_full) - np.outer(w_full, w_full)) / n
        sig_multi = np.sqrt(max(float(v_top @ S_multi @ v_top), 1e-30))

        rows.append(dict(
            sample=short, group=group, n=n,
            w1_em=w_full[0], w2_em=w_full[1], w3_em=w_full[2],
            slope=slope, r=r,
            sigma_soft_fulln=sig_soft[-1],
            angle_S1S2=angle_deg(v_top, E_S1S2),
            angle_vs_bayes=angle_deg(v_top, v_bayes),
            sigma_soft_bayes=b.soft_std_boot,
            ratio_bayes_em=b.soft_std_boot / sig_soft[-1],
            ratio_em_multinom=sig_soft[-1] / sig_multi,
            clipped=bool((w_full < 0.02).any()),
        ))

    res = pd.DataFrame(rows).sort_values(["group", "sample"]).reset_index(drop=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(RESULTS_DIR / "m3_nscaling_results.csv", index=False)
    pd.DataFrame(curves).to_csv(RESULTS_DIR / "m3_nscaling_curves.csv", index=False)

    with pd.option_context("display.width", 220, "display.max_columns", None):
        print("=== within-sample n-scaling (supervised EM, fixed shapes) ===")
        print(res.to_string(index=False, float_format=lambda x: f"{x:.3g}"))

    print("\n=== summary ===")
    print(f"slope: median {res.slope.median():+.3f}  "
          f"IQR [{res.slope.quantile(.25):+.3f}, {res.slope.quantile(.75):+.3f}]  "
          f"(prediction -0.500)")
    for flag, lab in [(False, "interior (no weight < 0.02)"),
                      (True, "boundary-clipped")]:
        sub = res[res.clipped == flag]
        if len(sub):
            print(f"  {lab:28s}: median slope {sub.slope.median():+.3f}  (n={len(sub)})")
    print(f"EM soft axis vs S1-S2 edge : median {res.angle_S1S2.median():.1f} deg")
    print(f"EM soft axis vs Bayes axis : median {res.angle_vs_bayes.median():.1f} deg")
    print(f"sigma_soft Bayes/EM at full n: median x{res.ratio_bayes_em.median():.1f} "
          f"(shape co-inference inflation)")
    print(f"sigma_soft EM/multinomial  : median x{res.ratio_em_multinom.median():.2f} "
          f"(vs counting-statistics floor along soft axis)")


if __name__ == "__main__":
    main()
