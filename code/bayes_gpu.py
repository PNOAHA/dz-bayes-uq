# -*- coding: utf-8 -*-
"""
bayes_gpu.py — GPU-batched Goodman-Weare sampler for the real-data Bayesian
DZ-unmix posterior. Numerical drop-in for bayesian_realdata.run_mcmc (emcee),
but runs THOUSANDS of independent fits in parallel on one GPU.

Lives in dz-unmix-bounds (active repo). Does NOT import or modify dz-unmix-cg;
the endmember constants below MIRROR dz-unmix-cg DEFAULT_SOURCES (read-only copy).

Model (identical to bayesian_realdata.py):
  theta = [w_1..w_{K-1}, mu_1..mu_K, sig_1..sig_K], w_K = 1 - sum(w_free)   (D=3K-1=8)
  log_prior : Dirichlet(1) on w (uniform; constant), Gaussian on mu & sigma,
              hard constraints  w in [0,1],  sigma > 0
  log_like  : sum_z logsumexp_k [ log w_k - 0.5((a_z-mu_k)/sig_k)^2
                                  - log sig_k - 0.5 log(2pi) ]
Sampler   : affine-invariant ensemble stretch move (Goodman & Weare 2010, a=2),
            two-half (red/black) update — exactly what emcee does.

Run validate_gpu.py to confirm equivalence with emcee BEFORE trusting at scale.
"""
from __future__ import annotations
import math
import torch

K = 3
D = 3 * K - 1                 # 8
LOG2PI = math.log(2.0 * math.pi)

# MIRROR of dz-unmix-cg DEFAULT_SOURCES (S1 130/15, S2 280/30, S3 1900/100). Read-only copy.
SRC_MEANS = (130.0, 280.0, 1900.0)
SRC_SIGMAS = (15.0, 30.0, 100.0)


def make_prior_batch(mean_shift, sigma_shift, *, mean_unc_pct=0.10, sigma_unc_pct=0.20,
                     device="cuda", dtype=torch.float64):
    """Per-fit perturbed prior, matching PriorSpec.from_default(...).perturbed(...).
    mean_shift, sigma_shift: (B,) tensors/arrays/floats. Returns dict of (B,K) tensors."""
    mc0 = torch.tensor(SRC_MEANS, device=device, dtype=dtype)
    sc0 = torch.tensor(SRC_SIGMAS, device=device, dtype=dtype)
    ms_pr = mc0 * mean_unc_pct          # prior std on mu — based on ORIGINAL center, NOT perturbed
    ss_pr = sc0 * sigma_unc_pct
    ms = torch.as_tensor(mean_shift, device=device, dtype=dtype).reshape(-1)
    ss = torch.as_tensor(sigma_shift, device=device, dtype=dtype).reshape(-1)
    B = ms.shape[0]
    return {
        "mc": mc0[None, :] * (1.0 + ms[:, None]),   # perturbed mu center (B,K)
        "sc": sc0[None, :] * (1.0 + ss[:, None]),   # perturbed sigma center (B,K)
        "ms": ms_pr[None, :].expand(B, K).contiguous(),
        "ss": ss_pr[None, :].expand(B, K).contiguous(),
    }


def theta_to_params(theta):
    """theta (...,D) -> weights (...,K), mus (...,K), sigmas (...,K)."""
    w_free = theta[..., :K - 1]
    w_last = 1.0 - w_free.sum(-1, keepdim=True)
    weights = torch.cat([w_free, w_last], dim=-1)
    return weights, theta[..., K - 1:2 * K - 1], theta[..., 2 * K - 1:3 * K - 1]


def log_posterior(theta, ages, mask, prior):
    """theta (B,W,D); ages (B,n); mask (B,n) {0,1}; prior dict of (B,K).
    Returns (B,W) log-posterior; -inf where w in/out of [0,1] or sigma<=0."""
    weights, mus, sigmas = theta_to_params(theta)                      # (B,W,K)
    bad = (weights < 0).any(-1) | (weights > 1).any(-1) | (sigmas <= 0).any(-1)
    mc = prior["mc"][:, None, :]; ms = prior["ms"][:, None, :]
    sc = prior["sc"][:, None, :]; ss = prior["ss"][:, None, :]
    lp = (-0.5 * (((mus - mc) / ms) ** 2).sum(-1)
          - 0.5 * (((sigmas - sc) / ss) ** 2).sum(-1))                 # (B,W)  Gaussian prior
    a = ages[:, None, :, None]                                         # (B,1,n,1)
    diff = (a - mus[:, :, None, :]) / sigmas[:, :, None, :]            # (B,W,n,K)
    log_comp = (torch.log(weights[:, :, None, :].clamp_min(1e-300))
                - 0.5 * diff * diff - torch.log(sigmas[:, :, None, :]) - 0.5 * LOG2PI)
    per_z = torch.logsumexp(log_comp, dim=-1) * mask[:, None, :]       # (B,W,n), padded->0
    out = lp + per_z.sum(-1)                                           # (B,W)
    return torch.where(bad, torch.full_like(out, float("-inf")), out)


def run_mcmc_gpu(ages, mask, prior, *, n_walkers=64, n_steps=5000, burn_in=1000,
                 a=2.0, seed=42, device="cuda", dtype=torch.float64, thin=4):
    """Batched affine-invariant ensemble sampler.
    ages (B,n) float, mask (B,n) {0,1}, prior dict of (B,K). Returns per-fit
    posterior summaries as (B,K) / (B,) tensors. Mirrors run_mcmc(emcee)."""
    g = torch.Generator(device=device); g.manual_seed(seed)
    B, n = ages.shape; W = n_walkers
    ages = ages.to(device, dtype); mask = mask.to(device, dtype)

    def randn(*s): return torch.randn(*s, device=device, generator=g, dtype=dtype)
    def rand(*s): return torch.rand(*s, device=device, generator=g, dtype=dtype)

    # init walkers (mirror initial_positions): w ~ Dirichlet(1) [first K-1]; mu/sig = center + 0.1*prior_std*N(0,1)
    theta = torch.empty(B, W, D, device=device, dtype=dtype)
    e = -torch.log(rand(B, W, K).clamp_min(1e-300))                    # Exp(1)
    theta[..., :K - 1] = (e / e.sum(-1, keepdim=True))[..., :K - 1]    # Dirichlet(1)
    theta[..., K - 1:2 * K - 1] = prior["mc"][:, None, :] + 0.1 * prior["ms"][:, None, :] * randn(B, W, K)
    theta[..., 2 * K - 1:3 * K - 1] = (prior["sc"][:, None, :] + 0.1 * prior["ss"][:, None, :] * randn(B, W, K)).clamp_min(1e-3)

    lp = log_posterior(theta, ages, mask, prior)                       # (B,W)
    half = W // 2
    halves = ((torch.arange(0, half, device=device), torch.arange(half, W, device=device)),
              (torch.arange(half, W, device=device), torch.arange(0, half, device=device)))

    n_kept = len(range(burn_in, n_steps, thin))
    Wsamp = torch.empty(B, n_kept, W, K, device=device, dtype=torch.float32)  # weights kept for quantiles (f32)
    ki = 0
    acc_sum = torch.zeros(B, device=device, dtype=dtype); acc_cnt = 0
    # mus/sigmas only need mean+std (no quantiles in the schema) -> stream sum/sumsq, no storage
    mu_sum = torch.zeros(B, K, device=device, dtype=dtype); mu_sq = torch.zeros(B, K, device=device, dtype=dtype)
    sg_sum = torch.zeros(B, K, device=device, dtype=dtype); sg_sq = torch.zeros(B, K, device=device, dtype=dtype)
    logp_max = torch.full((B,), float("-inf"), device=device, dtype=dtype)
    n_samp = 0

    for step in range(n_steps):
        for S, C in halves:
            ns = S.numel()
            jj = C[torch.randint(C.numel(), (B, ns), device=device, generator=g)]   # complementary pick
            Xc = torch.gather(theta, 1, jj[..., None].expand(B, ns, D))
            Xs = theta.index_select(1, S)
            z = ((a - 1.0) * rand(B, ns) + 1.0) ** 2 / a                            # g(z)
            Y = Xc + z[..., None] * (Xs - Xc)
            lpY = log_posterior(Y, ages, mask, prior)                               # (B,ns)
            logacc = (D - 1) * torch.log(z) + lpY - lp.index_select(1, S)
            accept = torch.log(rand(B, ns).clamp_min(1e-300)) < logacc              # (B,ns)
            theta[:, S, :] = torch.where(accept[..., None], Y, Xs)
            lp[:, S] = torch.where(accept, lpY, lp.index_select(1, S))
            acc_sum += accept.to(dtype).sum(1); acc_cnt += ns
        if step >= burn_in and (step - burn_in) % thin == 0:
            w_all, mu_all, sg_all = theta_to_params(theta)                         # (B,W,K)
            Wsamp[:, ki] = w_all.to(torch.float32); ki += 1
            mu_sum += mu_all.sum(1); mu_sq += (mu_all * mu_all).sum(1)
            sg_sum += sg_all.sum(1); sg_sq += (sg_all * sg_all).sum(1)
            logp_max = torch.maximum(logp_max, lp.max(1).values)
            n_samp += W

    Wf = Wsamp.reshape(B, n_kept * W, K).to(dtype)                                  # (B, n_samp, K)
    mu_mean = mu_sum / n_samp; sg_mean = sg_sum / n_samp
    return {
        "weights_mean": Wf.mean(1),
        "weights_std": Wf.std(1),
        "weights_q05": torch.quantile(Wf, 0.05, dim=1),
        "weights_q95": torch.quantile(Wf, 0.95, dim=1),
        "mus_mean": mu_mean,
        "mus_std": (mu_sq / n_samp - mu_mean * mu_mean).clamp_min(0).sqrt(),
        "sigmas_mean": sg_mean,
        "sigmas_std": (sg_sq / n_samp - sg_mean * sg_mean).clamp_min(0).sqrt(),
        "log_evidence_proxy": logp_max,
        "acceptance_fraction": acc_sum / acc_cnt,
        "n_samples_per_fit": n_kept * W,
    }
