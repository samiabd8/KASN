"""Input normalisation for the KASN estimator.

The KASN acts on inputs mapped to [0, 1] by their marginal empirical
distribution functions (eq. 2.3 of the paper). This induces compact support,
which is what allows the spline grid to stay fixed, and contributes an
O_P(n^{-1/2} log n) approximation error (Lemma 1) that is negligible relative
to the sieve approximation error of Theorem 1.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

__all__ = ["EmpiricalCDFTransformer"]


class EmpiricalCDFTransformer(TransformerMixin, BaseEstimator):
    """Marginal empirical-CDF map onto [0, 1], fitted on the training fold only.

    For each column j the transform is ``#{X_train[:, j] <= x} / (n + 1)``,
    so training values map into ``[1/(n+1), n/(n+1)]`` and new values beyond
    the training range map to the boundary values ``0`` or ``n/(n+1)``.

    Attributes
    ----------
    sorted_values_ : ndarray of shape (n_samples, n_features)
        Column-wise sorted training data.
    n_train_ : int
        Number of training observations used to build the EDFs.
    n_features_in_ : int
        Number of input columns seen during ``fit``.
    """

    def fit(self, X, y=None) -> "EmpiricalCDFTransformer":
        """Store the column-wise sorted training sample."""
        X = check_array(X, dtype=np.float64, ensure_2d=False)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.sorted_values_ = np.sort(X, axis=0)
        self.n_train_ = int(X.shape[0])
        self.n_features_in_ = int(X.shape[1])
        return self

    def transform(self, X) -> np.ndarray:
        """Map each column through its training EDF."""
        check_is_fitted(self, "sorted_values_")
        X = check_array(X, dtype=np.float64, ensure_2d=False)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} columns, the transformer was fitted on "
                f"{self.n_features_in_}")
        out = np.empty_like(X, dtype=np.float64)
        denom = self.n_train_ + 1.0
        for j in range(X.shape[1]):
            out[:, j] = np.searchsorted(self.sorted_values_[:, j], X[:, j],
                                        side="right") / denom
        return np.clip(out, 0.0, 1.0)
