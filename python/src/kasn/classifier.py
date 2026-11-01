"""KASN binary classification with the logistic criterion."""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.utils.validation import check_is_fitted, column_or_1d

from .base import COMMON_PARAMS_DOC, FITTED_DOC, BaseKASN
from .losses import LogisticLoss, Loss

__all__ = ["KASNClassifier"]


class KASNClassifier(ClassifierMixin, BaseKASN):
    __doc__ = """Kolmogorov-Arnold Sieve Network for binary classification.

    Fits the logit ``theta(x) = log(P(Y = 1 | x) / P(Y = 0 | x))`` by
    penalised binary cross-entropy, with offset ``logit(mean(y))``. The second
    entry of ``classes_`` is the positive class.

    Parameters
    ----------
""" + COMMON_PARAMS_DOC + FITTED_DOC + """    classes_ : ndarray of shape (2,)
"""

    def __init__(self, *, gamma: float = 0.4, grid_size: Optional[int] = None,
                 depth: Optional[int] = None, width: Optional[int] = None,
                 width_scale: int = 5, spline_order: int = 4,
                 grid_range: tuple = (-0.05, 1.05),
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

    def fit(self, X, y, X_val=None, y_val=None, sample_weight=None):
        classes = np.unique(column_or_1d(np.asarray(y), warn=True))
        if classes.shape[0] != 2:
            raise ValueError(f"KASNClassifier needs exactly two classes, got "
                             f"{classes.shape[0]}")
        self.classes_ = classes
        return super().fit(X, y, X_val=X_val, y_val=y_val,
                           sample_weight=sample_weight)

    # ----------------------------------------------------------- loss hooks
    def _make_loss(self) -> Loss:
        return LogisticLoss()

    def _check_y(self, y, n: int) -> np.ndarray:
        y = column_or_1d(np.asarray(y), warn=True)
        if y.shape[0] != n:
            raise ValueError(f"X has {n} rows but y has {y.shape[0]}")
        unseen = ~np.isin(y, self.classes_)
        if unseen.any():
            raise ValueError(f"labels {np.unique(y[unseen])} not among the "
                             f"training classes {self.classes_}")
        return (y == self.classes_[1]).astype(np.float64)

    def _encode_target(self, y) -> tuple[np.ndarray, float]:
        p = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
        return y, float(np.log(p / (1 - p)))

    def _transform_target(self, y) -> np.ndarray:
        return y

    def _extra_state(self) -> dict:
        return {"classes_": self.classes_.tolist()}

    def _set_extra_state(self, state: dict) -> None:
        self.classes_ = np.asarray(state["classes_"])

    # ------------------------------------------------------------ prediction
    def decision_function(self, X) -> np.ndarray:
        """Fitted logit of the positive class."""
        return self._decision(X)

    def predict_proba(self, X) -> np.ndarray:
        """Class probabilities, columns ordered as ``classes_``."""
        p = self._loss.inverse_link(self._decision(X))
        return np.column_stack([1.0 - p, p])

    def predict(self, X) -> np.ndarray:
        check_is_fitted(self, "classes_")
        return self.classes_[(self._decision(X) >= 0).astype(int)]
