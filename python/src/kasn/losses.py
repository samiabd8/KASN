"""Criterion functions ``-l(Z, theta)`` for the KASN sieve extremum estimator.

Each loss returns per-observation values so that sample weights and
validation criteria share one implementation. ``affine`` losses are fitted on
the standardised response and are equivariant to that location-scale map.
The others act on the raw response through a link and a fixed offset.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

__all__ = ["Loss", "SquaredLoss", "HuberLoss", "QuantileLoss", "PoissonLoss",
           "LogisticLoss", "make_loss"]


class Loss:
    """Base class. ``f`` is the network output on the link scale."""

    name = "loss"
    affine = True

    def per_sample(self, f: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def inverse_link(self, eta: np.ndarray) -> np.ndarray:
        return eta


class SquaredLoss(Loss):
    """``(y - f)^2``. Targets the conditional mean."""

    name = "squared"

    def per_sample(self, f, y):
        return (f - y) ** 2


class HuberLoss(Loss):
    """Huber loss with threshold ``delta`` on the standardised response."""

    name = "huber"

    def __init__(self, delta: float = 1.0) -> None:
        if delta <= 0:
            raise ValueError("huber_delta must be positive")
        self.delta = float(delta)

    def per_sample(self, f, y):
        return F.huber_loss(f, y, reduction="none", delta=self.delta)


class QuantileLoss(Loss):
    """Check loss ``rho_tau(y - f)``. Targets the conditional ``tau``-quantile."""

    name = "quantile"

    def __init__(self, tau: float = 0.5) -> None:
        if not 0 < tau < 1:
            raise ValueError("quantile must lie in (0, 1)")
        self.tau = float(tau)

    def per_sample(self, f, y):
        u = y - f
        return torch.maximum(self.tau * u, (self.tau - 1.0) * u)


class PoissonLoss(Loss):
    """Poisson negative log-likelihood ``exp(f) - y f`` with log link."""

    name = "poisson"
    affine = False

    def per_sample(self, f, y):
        return torch.exp(f) - y * f

    def inverse_link(self, eta):
        return np.exp(eta)


class LogisticLoss(Loss):
    """Binary cross-entropy with logit link."""

    name = "logistic"
    affine = False

    def per_sample(self, f, y):
        return F.binary_cross_entropy_with_logits(f, y, reduction="none")

    def inverse_link(self, eta):
        return 1.0 / (1.0 + np.exp(-eta))


def make_loss(name: str, *, quantile: float = 0.5,
              huber_delta: float = 1.0) -> Loss:
    """Build a loss from its name."""
    if name == "squared":
        return SquaredLoss()
    if name == "huber":
        return HuberLoss(huber_delta)
    if name == "quantile":
        return QuantileLoss(quantile)
    if name == "poisson":
        return PoissonLoss()
    if name == "logistic":
        return LogisticLoss()
    raise ValueError(f"unknown loss {name!r}")
