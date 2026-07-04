# -*- coding: utf-8 -*-
"""
第二篇方法论文 · 真实数据 Bayesian 混合模型
=============================================
对 dz-unmix-cg case_study_data/ 4 个 NE 亚 K1 sample 做完整 bayesian inversion。

Model：
  - K 个高斯端元 (mean μ_k, std σ_k)，prior 来自第一篇 DEFAULT_SOURCES + 用户指定不确定度
  - 混合权重 w ~ Dirichlet(α=1)，uniform on simplex
  - Marginalized over latent source assignments per zircon:
      log p(a | w, μ, σ) = log Σ_k w_k · N(a | μ_k, σ_k)
  - 输入 ages 可以是 raw 或 bootstrap resample（外层不确定性）

Sampler：emcee 3.x EnsembleSampler（Goodman-Weare affine-invariant）
  - 8-D 后验 (K-1=2 weights + K=3 means + K=3 sigmas)
  - 64 walkers × 5000 steps (可调) - 用 smoke 测时再 tune

这是 paper2 的 *补充模块*，不修改 applicability_scan.py。供 realdata_scan.py 调用。

作者：潘路加 — 2026-05-27
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

import emcee


# 把第一篇 core 挂上 path，复用 DEFAULT_SOURCES 作 prior center
def _ensure_core_on_path():
    try:
        import dz_unmixing_experiment  # noqa: F401
        return
    except ImportError:
        pass
    here = os.path.dirname(os.path.abspath(__file__))
    # 尝试候选路径（D107 + 本地两种布局都覆盖）
    candidates = []
    if os.environ.get("DZ_CORE_DIR"):
        candidates.append(os.environ["DZ_CORE_DIR"])
    # D107: paper2/bayesian_realdata.py → ../ = dz-unmix/
    candidates.append(os.path.abspath(os.path.join(here, "..")))
    # 本地: dz-unmix-bounds/d107_paper2_addon/ → ../../dz-unmix-cg/
    candidates.append(os.path.abspath(os.path.join(here, "..", "..", "dz-unmix-cg")))
    # 本地另一种布局（如果 dz-unmix-cg 在 sibling）
    candidates.append(os.path.abspath(os.path.join(here, "..", "..", "..", "dz-unmix-cg")))

    for cand in candidates:
        if os.path.isfile(os.path.join(cand, "dz_unmixing_experiment.py")):
            if cand not in sys.path:
                sys.path.insert(0, cand)
            return
    # 找不到——给出明确错误（与 D107 SLURM 默认 export DZ_CORE_DIR 一致）
    raise ImportError(
        "dz_unmixing_experiment 不在 path 上。设 DZ_CORE_DIR 环境变量或确认\n"
        "  D107: paper2 与 dz_unmixing_experiment.py 是 sibling 关系\n"
        "  本地: dz-unmix-bounds/d107_paper2_addon/ 与 dz-unmix-cg/ 是 sibling 关系\n"
        f"  已试: {candidates}"
    )


_ensure_core_on_path()
from dz_unmixing_experiment import DEFAULT_SOURCES, T_GRID  # noqa: E402


# ----------------------------------------------------------------------------
# 1. Bayesian model
# ----------------------------------------------------------------------------

@dataclass
class PriorSpec:
    """端元参数的高斯先验 spec。"""
    means_center: np.ndarray   # (K,) prior mean for μ_k
    means_sigma: np.ndarray    # (K,) prior std for μ_k
    sigmas_center: np.ndarray  # (K,) prior mean for σ_k
    sigmas_sigma: np.ndarray   # (K,) prior std for σ_k

    @classmethod
    def from_default(cls, mean_unc_pct=0.1, sigma_unc_pct=0.2,
                     sources=None):
        """从 DEFAULT_SOURCES 构造 prior。
        mean_unc_pct: μ 的相对不确定度（0.10 → ±10%）
        sigma_unc_pct: σ 的相对不确定度（0.20 → ±20%）
        """
        srcs = sources if sources is not None else DEFAULT_SOURCES
        mc = np.array([s.mean_age for s in srcs], dtype=float)
        sc = np.array([s.sigma for s in srcs], dtype=float)
        return cls(
            means_center=mc,
            means_sigma=mc * float(mean_unc_pct),
            sigmas_center=sc,
            sigmas_sigma=sc * float(sigma_unc_pct),
        )

    def perturbed(self, mean_shift_frac: float, sigma_shift_frac: float):
        """系统 shift prior center（用于 perturbation grid 的 robustness 测试）。
        mean_shift_frac=+0.05 → 所有 endmember μ 上移 5%
        sigma_shift_frac=-0.10 → 所有 endmember σ 下移 10%
        """
        return PriorSpec(
            means_center=self.means_center * (1 + mean_shift_frac),
            means_sigma=self.means_sigma,
            sigmas_center=self.sigmas_center * (1 + sigma_shift_frac),
            sigmas_sigma=self.sigmas_sigma,
        )


def _theta_to_params(theta: np.ndarray, K: int):
    """theta = [w_1, ..., w_{K-1}, μ_1, ..., μ_K, σ_1, ..., σ_K]
    返回 (weights[K], mus[K], sigmas[K])。w_K = 1 - sum(w_{1..K-1})。
    """
    w_free = theta[:K-1]
    w_last = 1.0 - np.sum(w_free)
    weights = np.concatenate([w_free, [w_last]])
    mus = theta[K-1:2*K-1]
    sigmas = theta[2*K-1:3*K-1]
    return weights, mus, sigmas


def log_prior(theta: np.ndarray, prior: PriorSpec, K: int = 3) -> float:
    weights, mus, sigmas = _theta_to_params(theta, K)
    # Simplex 约束：所有 weights ≥ 0
    if np.any(weights < 0) or np.any(weights > 1):
        return -np.inf
    # σ > 0 约束
    if np.any(sigmas <= 0):
        return -np.inf
    # Dirichlet(1,1,1) 在 simplex 上是 constant；不加 log term（吸到归一化里）
    lp = 0.0
    # Gaussian prior on μ
    lp += -0.5 * np.sum(((mus - prior.means_center) / prior.means_sigma) ** 2)
    # Gaussian prior on σ (truncated implicitly by sigmas>0 check)
    lp += -0.5 * np.sum(((sigmas - prior.sigmas_center) / prior.sigmas_sigma) ** 2)
    return lp


def log_likelihood(theta: np.ndarray, ages: np.ndarray, K: int = 3) -> float:
    """Marginalized over latent assignments. ages shape (n,)。"""
    weights, mus, sigmas = _theta_to_params(theta, K)
    if np.any(weights < 0) or np.any(sigmas <= 0):
        return -np.inf
    # log component matrix (n, K)
    diff = (ages[:, None] - mus[None, :]) / sigmas[None, :]
    log_comp = (
        np.log(weights[None, :] + 1e-300)
        - 0.5 * diff * diff
        - np.log(sigmas[None, :])
        - 0.5 * np.log(2 * np.pi)
    )
    # logsumexp per-zircon → sum
    return float(np.sum(logsumexp(log_comp, axis=1)))


def log_posterior(theta: np.ndarray, ages: np.ndarray, prior: PriorSpec,
                  K: int = 3) -> float:
    lp = log_prior(theta, prior, K)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta, ages, K)
    return lp + ll


# ----------------------------------------------------------------------------
# 2. MCMC runner
# ----------------------------------------------------------------------------

def initial_positions(n_walkers: int, prior: PriorSpec, K: int, rng: np.random.Generator) -> np.ndarray:
    """围绕 prior center 初始化 walkers。"""
    ndim = 3 * K - 1
    p0 = np.zeros((n_walkers, ndim))
    # weights: Dirichlet(1,..) sample, take first K-1
    w = rng.dirichlet(np.ones(K), size=n_walkers)
    p0[:, :K-1] = w[:, :K-1]
    # means: prior center + small jitter (10% of prior sigma)
    for k in range(K):
        p0[:, K-1+k] = prior.means_center[k] + 0.1 * prior.means_sigma[k] * rng.standard_normal(n_walkers)
    # sigmas: prior center + small jitter (10% of prior sigma)
    for k in range(K):
        p0[:, 2*K-1+k] = prior.sigmas_center[k] + 0.1 * prior.sigmas_sigma[k] * rng.standard_normal(n_walkers)
        # 确保 σ > 0
        p0[:, 2*K-1+k] = np.maximum(p0[:, 2*K-1+k], 1e-3)
    return p0


def run_mcmc(ages: np.ndarray, prior: PriorSpec, *,
             K: int = 3,
             n_walkers: int = 64,
             n_steps: int = 5000,
             burn_in: int = 1000,
             seed: int = 42) -> dict:
    """跑一次 emcee MCMC，返回 posterior samples + summary。"""
    rng = np.random.default_rng(seed)
    ndim = 3 * K - 1
    p0 = initial_positions(n_walkers, prior, K, rng)

    sampler = emcee.EnsembleSampler(
        n_walkers, ndim, log_posterior,
        args=(ages, prior, K),
    )
    sampler.run_mcmc(p0, n_steps, progress=False)
    # acceptance fraction 应在 0.2-0.5 之间为健康
    accept_frac = float(np.mean(sampler.acceptance_fraction))
    samples = sampler.get_chain(discard=burn_in, flat=True)  # (n_kept * n_walkers, ndim)
    log_probs = sampler.get_log_prob(discard=burn_in, flat=True)

    # 解出 weights samples
    w_free = samples[:, :K-1]
    w_last = 1.0 - w_free.sum(axis=1)
    weights = np.concatenate([w_free, w_last[:, None]], axis=1)  # (n_samples, K)

    return {
        "samples": samples,
        "log_probs": log_probs,
        "weights": weights,
        "weights_mean": weights.mean(axis=0).tolist(),
        "weights_std": weights.std(axis=0).tolist(),
        "weights_q05": np.quantile(weights, 0.05, axis=0).tolist(),
        "weights_q95": np.quantile(weights, 0.95, axis=0).tolist(),
        "mus_mean": samples[:, K-1:2*K-1].mean(axis=0).tolist(),
        "mus_std": samples[:, K-1:2*K-1].std(axis=0).tolist(),
        "sigmas_mean": samples[:, 2*K-1:3*K-1].mean(axis=0).tolist(),
        "sigmas_std": samples[:, 2*K-1:3*K-1].std(axis=0).tolist(),
        "log_evidence_proxy": float(np.max(log_probs)),  # 近似 BIC/Bayes factor 用
        "acceptance_fraction": accept_frac,
        "n_walkers": n_walkers,
        "n_steps": n_steps,
        "burn_in": burn_in,
        "K": K,
    }


# ----------------------------------------------------------------------------
# 3. Bootstrap helper
# ----------------------------------------------------------------------------

def bootstrap_ages(ages: np.ndarray, seed: int) -> np.ndarray:
    """对 zircon ages 数组做 with-replacement bootstrap（保留长度）。"""
    rng = np.random.default_rng(seed)
    return rng.choice(ages, size=len(ages), replace=True)


# ----------------------------------------------------------------------------
# 4. Local smoke
# ----------------------------------------------------------------------------

def _smoke():
    """1 sample × 1 bootstrap × 1 MCMC fit，测真实 timing。"""
    import time
    from pathlib import Path

    # 寻找 sample CSV
    cand_paths = [
        Path("../case_study_data"),  # paper2 → dz-unmix
        Path("../../dz-unmix-cg/data/case_study_data"),  # bounds → cg
        Path("/public/home/panlujia/dz-unmix/case_study_data"),
    ]
    csv_path = None
    for cand in cand_paths:
        if cand.is_dir():
            csvs = list(cand.glob("*.csv"))
            if csvs:
                csv_path = csvs[0]
                break
    if csv_path is None:
        print("[smoke] 找不到 sample CSV，请在 dz-unmix-cg/case_study_data 准备")
        return

    print(f"[smoke] load {csv_path.name}")
    # 跳过 # 注释行
    ages = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or not line[0].isdigit():
                continue
            try:
                ages.append(float(line.split(",")[0]))
            except (ValueError, IndexError):
                pass
    ages = np.array(ages)
    print(f"[smoke] {len(ages)} zircons, range=[{ages.min():.0f}, {ages.max():.0f}] Ma")

    prior = PriorSpec.from_default(mean_unc_pct=0.10, sigma_unc_pct=0.20)
    print(f"[smoke] prior μ center = {prior.means_center.tolist()}")
    print(f"[smoke] prior σ center = {prior.sigmas_center.tolist()}")

    print("[smoke] MCMC: 64 walkers × 5000 steps ...")
    t0 = time.time()
    res = run_mcmc(ages, prior, n_walkers=64, n_steps=5000, burn_in=1000, seed=42)
    dt = time.time() - t0
    print(f"[smoke] {dt:.1f} s elapsed")
    print(f"[smoke] acceptance fraction = {res['acceptance_fraction']:.3f}")
    print(f"[smoke] weights posterior mean = {[f'{w:.3f}' for w in res['weights_mean']]}")
    print(f"[smoke]                  ± std = {[f'{w:.3f}' for w in res['weights_std']]}")
    print(f"[smoke] weights 5-95% CI:")
    for k in range(res['K']):
        print(f"           w_{k}: [{res['weights_q05'][k]:.3f}, {res['weights_q95'][k]:.3f}]")
    print(f"[smoke] endmember μ posterior mean = {[f'{m:.1f}' for m in res['mus_mean']]}")


if __name__ == "__main__":
    _smoke()
