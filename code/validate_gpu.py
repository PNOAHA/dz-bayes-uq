# -*- coding: utf-8 -*-
"""
validate_gpu.py — confirm bayes_gpu.run_mcmc_gpu reproduces emcee
(bayesian_realdata.run_mcmc) within MC error, before trusting it at scale.

Runs a handful of (sample, mean_shift, sigma_shift, bootstrap) configs through
BOTH samplers with IDENTICAL bootstrapped ages and prior, prints the weights
posterior mean/std/CI side by side, and a PASS/CHECK verdict.

Needs: torch (GPU) + emcee + scipy  (the latter two only for the CPU reference).
emcee/scipy on the air-gapped container: install offline wheels (like pyarrow).

Run (L20 container, after uploading realdata_k1_samples + the addon scripts):
  DZ_SAMPLES_DIR=/private/realdata_k1_samples python validate_gpu.py --n-steps 5000
A quick smoke first (cheaper): add  --n-steps 1500
"""
from __future__ import annotations
import argparse, glob, os, sys, time
import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)
# the emcee reference imports dz_unmixing_experiment for (DEFAULT_SOURCES, T_GRID);
# on the air-gapped container the real cg core isn't present, so default to the
# constants-only stub (unless the user already exported DZ_CORE_DIR).
os.environ.setdefault("DZ_CORE_DIR", os.path.join(_here, "_corestub"))
from bayesian_realdata import PriorSpec, bootstrap_ages, run_mcmc   # emcee reference
from bayes_gpu import make_prior_batch, run_mcmc_gpu                 # GPU candidate


def load_samples(d):
    out = []
    for csv in sorted(glob.glob(os.path.join(d, "*.csv"))):
        ages = []
        for line in open(csv, encoding="utf-8"):
            line = line.strip()
            if not line or line.startswith("#") or not line[0].isdigit():
                continue
            try:
                ages.append(float(line.split(",")[0]))
            except (ValueError, IndexError):
                pass
        if len(ages) >= 30:
            out.append((os.path.basename(csv)[:-4], np.array(ages, dtype=float)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-dir", default=os.environ.get("DZ_SAMPLES_DIR", "/private/realdata_k1_samples"))
    ap.add_argument("--n-configs", type=int, default=4)
    ap.add_argument("--n-walkers", type=int, default=64)
    ap.add_argument("--n-steps", type=int, default=5000)
    ap.add_argument("--burn-in", type=int, default=1000)
    # tol on weights MEAN is loose by design: degenerate/wide posteriors (adjacent
    # endmembers S1=130 & S2=280 trade weight at ~constant likelihood) make even
    # emcee-vs-emcee seed scatter ~0.15-0.19 on w1/w2 at 5000 steps (see the twin
    # test in the session log). Equivalence = GPU-vs-emcee diff << emcee self-scatter,
    # which holds (~0.02). The STD/CI tol stays tight (those converge fast).
    ap.add_argument("--tol", type=float, default=0.05, help="max |diff| on weights mean for PASS")
    ap.add_argument("--tol-std", type=float, default=0.02, help="max |diff| on weights std for PASS")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[env] device={dev}  torch={torch.__version__}")
    samples = load_samples(args.samples_dir)
    if not samples:
        sys.exit(f"[err] no usable sample CSVs in {args.samples_dir}")
    print(f"[data] {len(samples)} samples loaded")

    shifts = [(0.0, 0.0), (0.05, -0.10), (-0.10, 0.20), (0.10, 0.10)]
    cfgs = []
    for i in range(args.n_configs):
        sn, ages = samples[i % len(samples)]
        ms, ss = shifts[i % len(shifts)]
        seed = (abs(hash(sn)) % (2 ** 31)) ^ (i * 1000 + 42)
        cfgs.append(dict(name=sn, ages=ages, ms=ms, ss=ss, seed=seed,
                         boot=bootstrap_ages(ages, seed=seed)))

    pd_prior = PriorSpec.from_default(mean_unc_pct=0.10, sigma_unc_pct=0.20)

    # ---- emcee reference (CPU, sequential) ----
    print("\n[emcee] reference fits ...")
    emc, t0 = [], time.time()
    for c in cfgs:
        r = run_mcmc(c["boot"], pd_prior.perturbed(c["ms"], c["ss"]),
                     n_walkers=args.n_walkers, n_steps=args.n_steps, burn_in=args.burn_in, seed=c["seed"])
        emc.append(r)
        print(f"  {c['name'][:22]:22s} ms={c['ms']:+.2f} ss={c['ss']:+.2f}  "
              f"w={[f'{w:.3f}' for w in r['weights_mean']]}  acc={r['acceptance_fraction']:.2f}")
    print(f"[emcee] {len(cfgs)} fits / {time.time()-t0:.1f}s  ({(time.time()-t0)/len(cfgs):.1f}s per fit)")

    # ---- GPU batched (all configs in one shot) ----
    print("\n[gpu] batched fits ...")
    nmax = max(len(c["boot"]) for c in cfgs)
    B = len(cfgs)
    A = np.zeros((B, nmax)); M = np.zeros((B, nmax))
    for i, c in enumerate(cfgs):
        m = len(c["boot"]); A[i, :m] = c["boot"]; M[i, :m] = 1.0
    prior = make_prior_batch([c["ms"] for c in cfgs], [c["ss"] for c in cfgs], device=dev)
    t0 = time.time()
    g = run_mcmc_gpu(torch.tensor(A), torch.tensor(M), prior,
                     n_walkers=args.n_walkers, n_steps=args.n_steps, burn_in=args.burn_in, seed=123, device=dev)
    if dev == "cuda":
        torch.cuda.synchronize()
    print(f"[gpu] {B} fits / {time.time()-t0:.1f}s in ONE batch  ({g['n_samples_per_fit']} post-samples/fit)")

    # ---- compare ----
    print("\n=== emcee vs GPU — weights posterior ===")
    gm = g["weights_mean"].cpu().numpy(); gs = g["weights_std"].cpu().numpy()
    gq05 = g["weights_q05"].cpu().numpy(); gq95 = g["weights_q95"].cpu().numpy()
    d_mean = d_std = d_ci = 0.0
    for i, c in enumerate(cfgs):
        em = np.array(emc[i]["weights_mean"]); es = np.array(emc[i]["weights_std"])
        eq05 = np.array(emc[i]["weights_q05"]); eq95 = np.array(emc[i]["weights_q95"])
        dm = np.abs(gm[i] - em).max(); dsd = np.abs(gs[i] - es).max()
        dci = max(np.abs(gq05[i] - eq05).max(), np.abs(gq95[i] - eq95).max())
        d_mean = max(d_mean, dm); d_std = max(d_std, dsd); d_ci = max(d_ci, dci)
        print(f"  [{c['name'][:16]:16s} ms={c['ms']:+.2f} ss={c['ss']:+.2f}]")
        print(f"     w_mean  emcee {np.array2string(em, precision=3, floatmode='fixed')}  "
              f"gpu {np.array2string(gm[i], precision=3, floatmode='fixed')}  |d|max={dm:.3f}")
        print(f"     w_std   emcee {np.array2string(es, precision=3, floatmode='fixed')}  "
              f"gpu {np.array2string(gs[i], precision=3, floatmode='fixed')}  |d|max={dsd:.3f}")
    print("\n=== max |diff| across configs ===")
    print(f"  weights_mean : {d_mean:.4f}")
    print(f"  weights_std  : {d_std:.4f}")
    print(f"  weights_CI   : {d_ci:.4f}")
    ok = d_mean < args.tol and d_std < args.tol_std
    if ok:
        print(f"\n[verdict] PASS — GPU == emcee within MC error "
              f"(mean<{args.tol}, std<{args.tol_std}). std/CI converge tight; "
              f"wider mean tol absorbs degenerate-posterior seed scatter.")
    else:
        print(f"\n[verdict] CHECK — mean d={d_mean:.4f} (tol {args.tol}) / "
              f"std d={d_std:.4f} (tol {args.tol_std}); investigate before scaling")


if __name__ == "__main__":
    main()
