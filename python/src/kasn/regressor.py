"""KASN regression: squared, Huber, quantile and Poisson criteria."""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.utils.validation import column_or_1d

from .base import COMMON_PARAMS_DOC, FITTED_DOC, BaseKASN
from .losses import Loss, make_loss

__all__ = ["KASNRegressor"]

_REG_LOSSES = ("squared", "huber", "quantile", "poisson")


class KASNRegressor(RegressorMixin, BaseKASN):
    __doc__ = """Kolmogorov-Arnold Sieve Network for regression.

    Fits ``theta_hat = argsup_{theta in K_n} L_n(theta)`` for the chosen
    criterion. Inputs are mapped to [0, 1] by their training EDFs (eq. 2.3)
    and a Group Lasso penalty acts on each edge's coefficient vector. After
    training, edges below ``Delta_n / r_n`` are pruned and the survivors are
    fine-tuned with the pruned edges frozen at zero.

    Parameters
    ----------
    loss : {'squared', 'huber', 'quantile', 'poisson'}, default='squared'
        ``'squared'``, ``'huber'`` and ``'quantile'`` act on the standardised
        response. ``'poisson'`` uses a log link on the raw (non-negative)
        response with offset ``log(mean(y))``.
    quantile : float, default=0.5
        ``tau`` for ``loss='quantile'``.
    huber_delta : float, default=1.0
        Huber threshold on the standardised response.
""" + COMMON_PARAMS_DOC + FITTED_DOC + """    y_mean_, y_scale_ : float
        Response standardisation (0 and 1 for ``'poisson'``).
"""

    def __init__(self, *, loss: str = "squared", quantile: float = 0.5,
                 huber_delta: float = 1.0, gamma: float = 0.4,
                 grid_size: Optional[int] = None, depth: Optional[int] = None,
                 width: Optional[int] = None, width_scale: int = 5,
                 spline_order: int = 4, grid_range: tuple = (-0.05, 1.05),
                 penalty: Optional[float] = None, penalty_scale: float = 0.05,
                 delta_mode: str = "penalty", zeta_delta: float = 0.4,
                 delta_scale: float = 1.0, delta_weight: float = 1.0,
                 prune: bool = True, prune_threshold: str = "delta_over_r",
                 prune_value: Optional[float] = None,
                 max_active_edges: Optional[int] = None,
                 finetune_epochs: int = 30, finetune_lr_scale: float = 0.5,
                 finetune_patience: int = 15, max_epochs: int = 500,
                 patience: Optional[int] = 50, learning_rate: float = 1e-3,
                 batch_size: Optional[int] = 32, weight_decay: float = 0.0,
                 lr_step_size: Optional[int] = None, lr_gamma: float = 0.5,
                 grad_clip: float = 500.0, validation_fraction: float = 0.2,
                 shuffle: bool = False, device: str = "auto", amp: bool = True,
                 random_state: Optional[int] = None, verbose: int = 0) -> None:
        self.loss = loss
        self.quantile = quantile
        self.huber_delta = huber_delta
        self.gamma = gamma
        self.grid_size = grid_size
        self.depth = depth
        self.width = width
        self.width_scale = width_scale
        self.spline_order = spline_order
        self.grid_range = grid_range
        self.penalty = penalty
        self.penalty_scale = penalty_scale
        self.delta_mode = delta_mode
        self.zeta_delta = zeta_delta
        self.delta_scale = delta_scale
        self.delta_weight = delta_weight
        self.prune = prune
        self.prune_threshold = prune_threshold
        self.prune_value = prune_value
        self.max_active_edges = max_active_edges
        self.finetune_epochs = finetune_epochs
        self.finetune_lr_scale = finetune_lr_scale
        self.finetune_patience = finetune_patience
        self.max_epochs = max_epochs
        self.patience = patience
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.grad_clip = grad_clip
        self.validation_fraction = validation_fraction
        self.shuffle = shuffle
        self.device = device
        self.amp = amp
        self.random_state = random_state
        self.verbose = verbose

    # ----------------------------------------------------------- loss hooks
    def _make_loss(self) -> Loss:
        if self.loss == "logistic":
            raise ValueError("use KASNClassifier for the logistic loss")
        if self.loss not in _REG_LOSSES:
            raise ValueError(f"loss must be one of {_REG_LOSSES}")
        return make_loss(self.loss, quantile=self.quantile,
                         huber_delta=self.huber_delta)

    def _check_y(self, y, n: int) -> np.ndarray:
        y = column_or_1d(np.asarray(y, dtype=np.float64), warn=True)
        if y.shape[0] != n:
            raise ValueError(f"X has {n} rows but y has {y.shape[0]}")
        if not np.all(np.isfinite(y)):
            raise ValueError("y contains NaN or infinite values")
        if self.loss == "poisson" and np.any(y < 0):
            raise ValueError("loss='poisson' needs a non-negative response")
        return y

    def _encode_target(self, y) -> tuple[np.ndarray, float]:
        if self.loss == "poisson":
            mu = float(y.mean())
            if mu <= 0:
                raise ValueError("loss='poisson' needs a positive mean response")
            self.y_mean_, self.y_scale_ = 0.0, 1.0
            return y, float(np.log(mu))
        self.y_mean_ = float(y.mean())
        sd = float(y.std())
        self.y_scale_ = sd if sd > 0 else 1.0
        return self._transform_target(y), 0.0

    def _transform_target(self, y) -> np.ndarray:
        return (y - self.y_mean_) / self.y_scale_

    def _extra_state(self) -> dict:
        return {"y_mean_": self.y_mean_, "y_scale_": self.y_scale_}

    def _set_extra_state(self, state: dict) -> None:
        self.y_mean_, self.y_scale_ = state["y_mean_"], state["y_scale_"]

    # ------------------------------------------------------------ prediction
    def predict(self, X) -> np.ndarray:
        """Conditional mean (squared, Huber, Poisson) or ``tau``-quantile."""
        eta = self._decision(X)
        if self.loss == "poisson":
            return self._loss.inverse_link(eta)
        return eta * self.y_scale_ + self.y_mean_
