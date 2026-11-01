"""Kolmogorov-Arnold Sieve Networks (KASN).

Sieve extremum estimation with deep Kolmogorov-Arnold networks, learned
B-spline edges and Group Lasso regularisation.

KASNRegressor
    Squared, Huber, quantile and Poisson criteria.
KASNClassifier
    Binary logistic criterion.
tune_kasn
    Prune-aware greedy hyperparameter search on a validation fold.
make_kasn_regression
    Monte Carlo designs of Section 4.
load
    Restore any estimator written by ``.save()``.
"""

from typing import Optional

from .base import _read_payload
from .classifier import KASNClassifier
from .datasets import make_kasn_regression, target_function
from .losses import make_loss
from .network import KASNLayer, KASNNetwork
from .preprocessing import EmpiricalCDFTransformer
from .regressor import KASNRegressor
from .splines import BSplineBasis
from .theory import architecture, check_conditions
from .tuning import TuningResult, tune_kasn

__version__ = "0.1.0"


def load(path, device: Optional[str] = None):
    """Load a ``KASNRegressor`` or ``KASNClassifier`` written by ``.save()``."""
    payload = _read_payload(path)
    classes = {"KASNRegressor": KASNRegressor,
               "KASNClassifier": KASNClassifier}
    if payload["class"] not in classes:
        raise ValueError(f"unknown estimator class {payload['class']!r}")
    return classes[payload["class"]]._from_payload(payload, device)


__all__ = [
    "KASNRegressor", "KASNClassifier", "load", "tune_kasn", "TuningResult",
    "make_kasn_regression", "target_function", "make_loss",
    "EmpiricalCDFTransformer", "BSplineBasis", "KASNLayer", "KASNNetwork",
    "architecture", "check_conditions", "__version__",
]
