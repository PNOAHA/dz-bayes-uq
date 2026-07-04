# -*- coding: utf-8 -*-
"""
run_scan_gpu.py — full real-data Bayesian uncertainty scan on GPU.

Reproduces realdata_scan.py's grid (sample x mean_shift x sigma_shift x bootstrap)
but runs every fit through bayes_gpu.run_mcmc_gpu in large GPU batches instead of
one-at-a-time emcee on CPU. Equivalence to emcee is established by validate_gpu.py
(run that FIRST — verdict PASS).

Grid (defaults match realdata_scan):
  15 samples x 11 mean_shift x 11 sigma_shift x 100 bootstrap = 181,500 fits
  each = 64 walkers x 5000 steps (burn-in 1000), 8-D posterior.

Batching: per sample (uniform n_zircons, no padding), chunked to fit GPU memory.
Output: one parquet per sample in --out-dir (resumable — existing files skipped),
        then a merged grains_bayes_gpu.parquet. Schema expands weights/mus/sigmas
        into numeric columns (w1/w2/w3 etc.) keyed to S1/S2/S3 endmembers.

Does NOT import emcee or dz-unmix-cg; uses only bayes_gpu (+ torch/numpy/pandas).
Endmember order/labels mirror dz-unmix-cg DEFAULT_SOURCES (S1 130, S2 280, S3 1900).

Run (L20 container, after validate PASS):
  python run_scan_gpu.py --samples-dir /private/bayes_gpu/realdata_k1_samples \
      --out-dir /private/bayes_gpu/out --chunk 3000
  # smoke: --n-bootstrap 2 --n-steps 1500   (tiny, ~1 min, sanity the plumbing)
"""
from __future__ import annotations
import argparse, glob, hashlib, os, sys, time
import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)
from bayes_gpu import make_prior_batch, run_mcmc_gpu, K  # noqa: E402

SRC_LABELS = ["S1", "S2", "S3"]   # mirror DEFAULT_SOURCES order (130 / 280 / 1900 Ma)


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
            out.append((os.path.basename(csv)[:-4], np.asarray(ages, dtype=np.float64)))
    return out


def boot_seed(name, b):
    """Deterministic, reproducible bootstrap seed (stable across runs/machines —
    unlike realdata_scan's hash(), which used non-deterministic PYTHONHASHSEED)."""
    return int(hashlib.sha1(f"{name}:{b}".encode()).hexdigest()[:8], 16)


def run_sample(name, ages, *, mean_shifts, sigma_shifts, n_boot, chunk,
               n_walkers, n_steps, burn_in, thin, device, dtype):
    """Run the full prior x bootstrap grid for one sample. Returns a list of row dicts."""
    n = len(ages)
    # precompute the n_boot resamples once (bootstrap depends only on (sample,b))
    boots = np.stack([np.random.default_rng(boot_seed(name, b)).choice(ages, size=n, replace=True)
                      for b in range(n_boot)])                      # (n_boot, n)
    # flat config list: (ms, ss, b) cartesian — order matches realdata_scan
    cfgs = [(float(ms), float(ss), b)
            for ms in mean_shifts for ss in sigma_shifts for b in range(n_boot)]
    rows = []
    t_s = time.time()
    for c0 in range(0, len(cfgs), chunk):
        cc = cfgs[c0:c0 + chunk]
        Bc = len(cc)
        ages_mat = np.stack([boots[b] for (_, _, b) in cc])         # (Bc, n)
        ms_arr = [ms for (ms, _, _) in cc]; ss_arr = [ss for (_, ss, _) in cc]
        prior = make_prior_batch(ms_arr, ss_arr, device=device, dtype=dtype)
        res = run_mcmc_gpu(torch.tensor(ages_mat), torch.ones(Bc, n),
                           prior, n_walkers=n_walkers, n_steps=n_steps,
                           burn_in=burn_in, thin=thin, seed=20260609,
                           device=device, dtype=dtype)
        if device == "cuda":
            torch.cuda.synchronize()
        # pull to cpu numpy
        npres = {k: (v.cpu().numpy() if torch.is_tensor(v) else v) for k, v in res.items()}
        for j, (ms, ss, b) in enumerate(cc):
            row = dict(sample_name=name, n_zircons=n,
                       mean_shift=round(ms, 4), sigma_shift=round(ss, 4), bootstrap_idx=b,
                       acceptance_fraction=float(npres["acceptance_fraction"][j]),
                       log_evidence_proxy=float(npres["log_evidence_proxy"][j]),
                       n_walkers=n_walkers, n_steps=n_steps)
            for k in range(K):
                lab = SRC_LABELS[k]
                row[f"w_{lab}_mean"] = float(npres["weights_mean"][j, k])
                row[f"w_{lab}_std"] = float(npres["weights_std"][j, k])
                row[f"w_{lab}_q05"] = float(npres["weights_q05"][j, k])
                row[f"w_{lab}_q95"] = float(npres["weights_q95"][j, k])
                row[f"mu_{lab}_mean"] = float(npres["mus_mean"][j, k])
                row[f"mu_{lab}_std"] = float(npres["mus_std"][j, k])
                row[f"sig_{lab}_mean"] = float(npres["sigmas_mean"][j, k])
                row[f"sig_{lab}_std"] = float(npres["sigmas_std"][j, k])
            rows.append(row)
        done = min(c0 + chunk, len(cfgs))
        el = time.time() - t_s
        eta = el / done * (len(cfgs) - done)
        print(f"    [{name[:24]:24s}] {done:5d}/{len(cfgs)} fits  "
              f"({el:.0f}s, ETA {eta:.0f}s)  acc~{npres['acceptance_fraction'].mean():.2f}",
              flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-dir", default=os.environ.get("DZ_SAMPLES_DIR",
                    "/private/bayes_gpu/realdata_k1_samples"))
    ap.add_argument("--out-dir", default="/private/bayes_gpu/out")
    ap.add_argument("--n-mean-shifts", type=int, default=11)
    ap.add_argument("--n-sigma-shifts", type=int, default=11)
    ap.add_argument("--n-bootstrap", type=int, default=100)
    ap.add_argument("--n-walkers", type=int, default=64)
    ap.add_argument("--n-steps", type=int, default=5000)
    ap.add_argument("--burn-in", type=int, default=1000)
    ap.add_argument("--thin", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=3000, help="fits per GPU batch (raise to use more VRAM)")
    ap.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    args = ap.parse_args()

    import pandas as pd
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    print(f"[env] device={dev} torch={torch.__version__} dtype={args.dtype}")
    if dev == "cuda":
        p = torch.cuda.get_device_properties(0)
        print(f"[env] GPU={torch.cuda.get_device_name(0)} {p.total_memory/1e9:.0f}GB")

    mean_shifts = np.linspace(-0.10, 0.10, args.n_mean_shifts)
    sigma_shifts = np.linspace(-0.20, 0.20, args.n_sigma_shifts)
    samples = load_samples(args.samples_dir)
    if not samples:
        sys.exit(f"[err] no samples in {args.samples_dir}")
    per_sample = args.n_mean_shifts * args.n_sigma_shifts * args.n_bootstrap
    print(f"[grid] {len(samples)} samples x {args.n_mean_shifts} ms x {args.n_sigma_shifts} ss "
          f"x {args.n_bootstrap} boot = {len(samples)*per_sample:,} fits "
          f"({args.n_walkers}w x {args.n_steps}step)")
    os.makedirs(args.out_dir, exist_ok=True)

    t0 = time.time()
    for si, (name, ages) in enumerate(samples):
        out_pq = os.path.join(args.out_dir, f"{name}.parquet")
        if os.path.isfile(out_pq):
            print(f"  [{si+1}/{len(samples)}] {name[:30]:30s}  SKIP (exists)")
            continue
        print(f"  [{si+1}/{len(samples)}] {name[:30]:30s}  n={len(ages)}  ({per_sample} fits)")
        rows = run_sample(name, ages, mean_shifts=mean_shifts, sigma_shifts=sigma_shifts,
                          n_boot=args.n_bootstrap, chunk=args.chunk, n_walkers=args.n_walkers,
                          n_steps=args.n_steps, burn_in=args.burn_in, thin=args.thin,
                          device=dev, dtype=dtype)
        pd.DataFrame(rows).to_parquet(out_pq, index=False)
        print(f"      -> {out_pq}  ({len(rows)} rows)")

    # merge
    parts = sorted(glob.glob(os.path.join(args.out_dir, "*.parquet")))
    parts = [p for p in parts if os.path.basename(p) != "grains_bayes_gpu.parquet"]
    if parts:
        big = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        merged = os.path.join(args.out_dir, "grains_bayes_gpu.parquet")
        big.to_parquet(merged, index=False)
        print(f"\n[merge] {len(big):,} rows -> {merged}")
    print(f"[done] total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
