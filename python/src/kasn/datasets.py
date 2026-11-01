"""Simulation designs from Section 4 of the paper.

The covariates are a stationary Gaussian VAR(1) with diagonal coefficient
``rho`` across time, mapped to ``Uniform(support)`` marginals through the
Gaussian copula by default. Only the first ``s`` columns enter ``f_0``.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy.stats import norm
from sklearn.utils import Bunch

__all__ = ["make_kasn_regression", "target_function", "TARGETS"]


def _f_additive(Xs: np.ndarray) -> np.ndarray:
    return np.sin(Xs).sum(axis=1)


def _f_compositional(Xs: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-np.sin(Xs).sum(axis=1)))
            + 0.1 * np.abs(Xs).sum(axis=1))


def _f_deep(Xs: np.ndarray) -> np.ndarray:
    inner = np.sin(Xs).sum(axis=1) / np.sqrt(max(Xs.shape[1], 1))
    psi2 = np.tanh(0.8 * inner)
    return np.sin(2.0 * psi2) + 0.5 * psi2 ** 2


TARGETS = {"additive": _f_additive, "compositional": _f_compositional,
           "deep": _f_deep}

_SCALE_CACHE: dict = {}


def _to_marginal(Z: np.ndarray, marginal: str,
                 support: Sequence[float]) -> np.ndarray:
    if marginal == "uniform":
        lo, hi = support
        return lo + (hi - lo) * norm.cdf(Z)
    if marginal == "normal":
        return Z
    raise ValueError(f"unknown marginal {marginal!r}, use 'uniform' or 'normal'")


def _target_scale(target: str, s: int, marginal: str,
                  support: Sequence[float]) -> tuple[float, float, float]:
    """Monte Carlo mean and sd of ``target`` and sd of the additive target."""
    key = (target, s, marginal, tuple(support))
    if key not in _SCALE_CACHE:
        rng = np.random.default_rng(20240817)
        Xr = _to_marginal(rng.standard_normal((200_000, s)), marginal, support)
        raw = TARGETS[target](Xr)
        _SCALE_CACHE[key] = (float(raw.mean()), float(raw.std()),
                             float(_f_additive(Xr).std()))
    return _SCALE_CACHE[key]


def target_function(X: np.ndarray, s: int, target: str = "compositional", *,
                    marginal: str = "uniform",
                    support: Sequence[float] = (-3.0, 3.0),
                    standardise: bool = True) -> np.ndarray:
    """Evaluate the true regression function ``f_0`` at ``X``.

    With ``standardise=True`` the non-additive targets are centred and
    rescaled to the standard deviation of the additive target, so every
    design has the same signal-to-noise ratio.
    """
    if target not in TARGETS:
        raise ValueError(f"unknown target {target!r}, expected one of "
                         f"{sorted(TARGETS)}")
    X = np.asarray(X, dtype=np.float64)
    out = TARGETS[target](X[:, :s])
    if standardise and target != "additive":
        mu, sd, ref_sd = _target_scale(target, s, marginal, support)
        if sd > 1e-12:
            out = (out - mu) / sd * ref_sd
    return out


def _stationary_ar1(n: int, d: int, rho: float, sd: float,
                    rng: np.random.Generator, burn_in: int) -> np.ndarray:
    """Gaussian AR(1) columns with an ``N(0, sd^2)`` marginal for every ``rho``."""
    if rho == 0.0:
        return rng.normal(0.0, sd, size=(n, d))
    total = n + burn_in
    innov = rng.normal(0.0, sd * np.sqrt(1.0 - rho ** 2), size=(total, d))
    out = np.empty((total, d))
    out[0] = rng.normal(0.0, sd, size=d)
    for t in range(1, total):
        out[t] = rho * out[t - 1] + innov[t]
    return out[burn_in:]


def make_kasn_regression(n: int = 2000, d: int = 500, s: int = 50, *,
                         target: str = "compositional", rho: float = 0.7,
                         marginal: str = "uniform",
                         support: Sequence[float] = (-3.0, 3.0),
                         noise_sd: float = 0.5, noise_rho: float = 0.0,
                         standardise: bool = True, burn_in: int = 500,
                         random_state: Optional[int] = None) -> Bunch:
    """Simulate ``Y_t = f_0(X_t) + eps_t`` as in the paper's Monte Carlo designs.

    Parameters
    ----------
    n, d, s : int
        Sample size, covariate dimension and number of relevant covariates.
    target : {'additive', 'compositional', 'deep'}
        ``'additive'`` is ``sum_j sin(X_j)``. ``'compositional'`` is
        ``sigmoid(sum_j sin X_j) + 0.1 sum_j |X_j|``.
    rho : float
        AR(1) coefficient of every covariate across time (``0`` gives i.i.d.).
    marginal, support
        Covariate marginal: ``'uniform'`` on ``support`` (via the Gaussian
        copula) or ``'normal'``.
    noise_sd, noise_rho : float
        Standard deviation and AR(1) coefficient of the error.
    standardise : bool
        Rescale non-additive targets to the additive target's sd.
    random_state : int, optional

    Returns
    -------
    Bunch with ``X`` (n, d), ``y`` (n,), ``f0`` (n,), ``eps`` (n,) and the
    design settings.
    """
    if s > d:
        raise ValueError("s cannot exceed d")
    rng = np.random.default_rng(random_state)
    Z = _stationary_ar1(n, d, rho, 1.0, rng, burn_in)
    X = _to_marginal(Z, marginal, support)
    f0 = target_function(X, s, target, marginal=marginal, support=support,
                         standardise=standardise)
    eps = _stationary_ar1(n, 1, noise_rho, noise_sd, rng, burn_in).ravel()
    return Bunch(X=X, y=f0 + eps, f0=f0, eps=eps, s=s, target=target,
                 rho=rho, marginal=marginal, support=tuple(support),
                 noise_sd=noise_sd, standardise=standardise)
