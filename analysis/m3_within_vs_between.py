# -*- coding: utf-8 -*-
r"""M3 step 3: single-fit posterior width (credible) vs between-fit scatter of
means (bootstrap + prior sensitivity), per sample.

Steps 1-2 (m3_eigen_probe.py, m3_nscaling_probe.py) studied only the ESTIMATOR-
level uncertainty: how the posterior *mean* moves across the 121x100 prior-
perturbation x bootstrap grid. Each individual MCMC fit ALSO carries its own
CREDIBLE-interval width (parquet columns w_S{1,2,3}_std), never yet compared.

The two are the two terms of the law of total variance for the marginal
posterior pooled over the scan grid (balanced 121 priors x 100 bootstraps):

    Var_pool(w_k) = E_fit[ Var_within(w_k) ]  +  Var_fit[ E_within(w_k) ]
                    \_______ WITHIN ________/     \_______ BETWEEN _______/
                     credible width, per fit       scan scatter of the mean
                     = mean(w_k_std^2)             = var(w_k_mean)  == Table 1 std^2

Question answered: is the reported uncertainty carried by the single-fit
credible interval (WITHIN dominates -> one Bayesian fit is already honest), or
by the sensitivity of the point estimate to prior/data (BETWEEN dominates ->
a single credible interval is OVERCONFIDENT and the scan is load-bearing)?

Secondary (cheap, exact for the balanced design): split BETWEEN into
data-resampling vs prior-sensitivity components,
    Var_fit[mean] = E_prior[ Var_boot(mean) ] + Var_prior[ E_boot(mean) ],
connecting to step 2's finding that the S1<->S2 softness is a counting-noise
floor (=> bootstrap-driven, not prior-driven).

Built-in verification: BETWEEN std recomputed here must equal the std already
reported in PAPER3 Table 1 / m3_eigen soft-direction inputs.

Reads grains_bayes_gpu.parquet (this dir). Zero cluster compute; local seconds.
NOT part of the M1 probe2_* series.
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PARQUET = HERE / "grains_bayes_gpu.parquet"
WK = ["S1", "S2", "S3"]
MEAN = {k: f"w_{k}_mean" for k in WK}
STD = {k: f"w_{k}_std" for k in WK}
E_S1S2 = np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0)


def totvar_decomp(g):
    """Law-of-total-variance decomposition for one sample's 12,100 fits.

    Returns per-weight dict of between (std of means), within_rms
    (sqrt mean of credible var), and the balanced-design ANOVA split of the
    between-variance into bootstrap vs prior components.
    """
    out = {}
    # prior config = (mean_shift, sigma_shift); 121 groups x 100 bootstraps
    by_cfg = g.groupby(["mean_shift", "sigma_shift"])
    n_cfg = by_cfg.ngroups
    for k in WK:
        m = g[MEAN[k]].to_numpy(float)          # 12,100 posterior means
        s = g[STD[k]].to_numpy(float)           # 12,100 credible sds

        v_total = m.var(ddof=0)                 # BETWEEN: var of the means
        within_rms2 = np.mean(s ** 2)           # WITHIN : mean credible variance

        # ANOVA split of v_total (exact for the balanced 121x100 design)
        cfg_mean = by_cfg[MEAN[k]].mean().to_numpy()     # per-prior mean over 100 boots
        cfg_var = by_cfg[MEAN[k]].var(ddof=0).to_numpy()  # per-prior boot variance
        v_boot = cfg_var.mean()                 # data-resampling component
        v_prior = cfg_mean.var(ddof=0)          # prior-sensitivity component

        grand = v_total + within_rms2
        out[k] = dict(
            between=np.sqrt(v_total),
            within_rms=np.sqrt(within_rms2),
            within_med=float(np.median(s)),
            total=np.sqrt(grand),
            frac_between=v_total / grand if grand > 0 else np.nan,
            ratio_bt_wn=(np.sqrt(v_total) / np.sqrt(within_rms2)
                         if within_rms2 > 0 else np.inf),
            boot_sd=np.sqrt(max(v_boot, 0.0)),
            prior_sd=np.sqrt(max(v_prior, 0.0)),
            frac_boot=v_boot / v_total if v_total > 0 else np.nan,
            anova_resid=abs(v_total - (v_boot + v_prior)),  # must be ~0
            n_cfg=n_cfg,
        )
    return out


def soft_axis_within_proxy(g):
    """Typical single-fit credible width along the S1-S2 soft axis (proxy).

    The full within-fit posterior covariance is not stored (only marginal
    w_k_std). In the degenerate limit the single-fit posterior is fully
    anti-correlated along S1-S2 (corr_within(w1,w2) -> -1; supported by the
    eigenstructure), giving soft width ~ (w1_std + w2_std)/sqrt(2). We report
    this as a labelled proxy, compared against the exact BETWEEN soft width
    sqrt(lam1) from m3_eigen (covariance of the means, anti-correlation exact).
    """
    s1 = g[STD["S1"]].to_numpy(float)
    s2 = g[STD["S2"]].to_numpy(float)
    soft_within = (s1 + s2) / np.sqrt(2.0)      # per-fit proxy
    return float(np.sqrt(np.mean(soft_within ** 2)))  # RMS over grid


def main():
    df = pd.read_parquet(PARQUET)
    missing = [c for c in list(MEAN.values()) + list(STD.values())
               if c not in df.columns]
    if missing:
        raise SystemExit(f"parquet missing expected columns: {missing}\n"
                         f"available: {list(df.columns)}")

    eig = None
    eig_path = HERE / "m3_eigen_results.csv"
    if eig_path.exists():
        eig = pd.read_csv(eig_path).set_index("sample")

    rows, soft_rows = [], []
    for name, g in df.groupby("sample_name"):
        short = name.split("__")[-1]
        group = name.split("__")[0]
        n = int(g["n_zircons"].iloc[0])
        dec = totvar_decomp(g)

        for k in WK:
            d = dec[k]
            rows.append(dict(sample=short, group=group, n=n, weight=k, **d))

        # soft-axis: between (exact, from eigen) vs within proxy
        soft_within = soft_axis_within_proxy(g)
        soft_between = float(np.sqrt(eig.loc[short, "lam1"])) if eig is not None \
            else float("nan")
        soft_rows.append(dict(
            sample=short, group=group, n=n,
            soft_between=soft_between,          # sqrt(lam1), exact anti-corr
            soft_within_proxy=soft_within,      # (s1+s2)/sqrt2 RMS, corr=-1 limit
            soft_ratio_bt_wn=(soft_between / soft_within
                              if soft_within > 0 else np.inf),
        ))

    res = pd.DataFrame(rows)
    soft = pd.DataFrame(soft_rows).sort_values(["group", "sample"]).reset_index(drop=True)
    res.to_csv(HERE / "m3_within_vs_between_results.csv", index=False)
    soft.to_csv(HERE / "m3_within_vs_between_soft.csv", index=False)

    # ---- verification: ANOVA identity + Table-1 cross-check ----------------
    max_resid = res["anova_resid"].max()
    print(f"[verify] max |Var_total - (Var_boot + Var_prior)| = {max_resid:.2e}"
          f"  (balanced-design identity, should be ~1e-18)")

    chk = res[(res["sample"] == "05FW007")].set_index("weight")["between"]
    print("[verify] 05FW007 BETWEEN std (recomputed) vs Table 1 "
          "(0.118 / 0.121 / 0.070):")
    print(f"         S1={chk.get('S1', float('nan')):.3f}  "
          f"S2={chk.get('S2', float('nan')):.3f}  "
          f"S3={chk.get('S3', float('nan')):.3f}")

    # ---- per-weight table ---------------------------------------------------
    with pd.option_context("display.width", 220, "display.max_columns", None):
        show = ["sample", "weight", "n", "between", "within_rms", "within_med",
                "ratio_bt_wn", "frac_between", "boot_sd", "prior_sd", "frac_boot"]
        print("\n=== per-sample per-weight within-vs-between decomposition ===")
        print(res.sort_values(["group", "sample", "weight"])[show]
              .to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    # ---- summary: hard (S3) vs soft (S1,S2) --------------------------------
    print("\n=== summary: ratio BETWEEN / WITHIN (credible) ===")
    print("  ratio < 1  -> single-fit credible interval already dominates (honest/conservative)")
    print("  ratio > 1  -> point estimate wanders more than any single posterior is wide (overconfident)")
    soft_mask = res["weight"].isin(["S1", "S2"])
    for lab, sub in [("HARD  w_S3      ", res[res["weight"] == "S3"]),
                     ("SOFT  w_S1,w_S2 ", res[soft_mask])]:
        r = sub["ratio_bt_wn"].replace([np.inf, -np.inf], np.nan).dropna()
        print(f"  {lab}: median ratio {r.median():.2f}  "
              f"IQR [{r.quantile(.25):.2f}, {r.quantile(.75):.2f}]  "
              f"median frac_between {sub['frac_between'].median():.2f}")

    print("\n=== summary: within BETWEEN, bootstrap vs prior share ===")
    for lab, sub in [("HARD  w_S3      ", res[res["weight"] == "S3"]),
                     ("SOFT  w_S1,w_S2 ", res[soft_mask])]:
        fb = sub["frac_boot"].replace([np.inf, -np.inf], np.nan).dropna()
        print(f"  {lab}: median frac of BETWEEN var from bootstrap "
              f"(vs prior) = {fb.median():.2f}")

    print("\n=== soft-axis (S1-S2): exact BETWEEN vs within proxy ===")
    print(soft.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    r = soft["soft_ratio_bt_wn"].replace([np.inf, -np.inf], np.nan).dropna()
    print(f"\nsoft-axis BETWEEN/WITHIN: median {r.median():.2f}  "
          f"range {r.min():.2f}-{r.max():.2f}")


if __name__ == "__main__":
    main()
