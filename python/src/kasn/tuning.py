"""Prune-aware hyperparameter selection on a validation fold.

The search is greedy and coordinate-wise, as in the research scripts:
``gamma``, then ``learning_rate``, then ``batch_size``, then ``penalty_scale``
and ``width_scale`` jointly. Each candidate is scored on the estimator that will
actually be reported (pruned and fine-tuned) rather than on the raw fit,
since a weak penalty leaves the network dense and threshold pruning then
removes signal-carrying edges.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Optional, Sequence

from .regressor import KASNRegressor

__all__ = ["tune_kasn", "TuningResult"]


@dataclass
class TuningResult:
    """Outcome of :func:`tune_kasn`.

    Attributes
    ----------
    best_params : dict
        Parameters for the estimator class (base parameters merged with the
        selected values).
    log : list of dict
        One row per candidate fit.
    selection_time : float
        Wall time in seconds.
    """

    best_params: dict
    log: list = field(default_factory=list)
    selection_time: float = 0.0

    def to_frame(self):
        """The candidate log as a pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame(self.log)


def tune_kasn(X, y, X_val=None, y_val=None, *, estimator=KASNRegressor,
              base_params: Optional[dict] = None,
              gamma_grid: Sequence[float] = (0.4,),
              learning_rate_grid: Sequence[float] = (1e-3,),
              batch_size_grid: Sequence[int] = (16, 32, 64),
              penalty_scale_grid: Sequence[float] = (0.05, 0.1, 1.0, 5.0),
              width_scale_grid: Sequence[int] = (5, 15),
              tuning_epochs: int = 50, tuning_patience: Optional[int] = 10,
              prune_aware: bool = True, selection: str = "validation",
              random_state: Optional[int] = 0,
              verbose: int = 1) -> TuningResult:
    """Greedy, prune-aware search over the KASN hyperparameters.

    Parameters
    ----------
    X, y, X_val, y_val
        As in ``fit``. Without ``X_val`` the chronological tail of size
        ``validation_fraction`` is used for every candidate.
    estimator : class, default=KASNRegressor
        ``KASNRegressor`` or ``KASNClassifier``.
    base_params : dict, optional
        Fixed estimator parameters shared by all candidates (for example
        ``loss``). Candidates are scored on the validation value of that loss.
    *_grid : sequences
        Candidate values. A grid of length one is held fixed.
    tuning_epochs, tuning_patience : int
        Training length and early-stopping patience per candidate.
    prune_aware : bool
        Score the pruned and fine-tuned fit (recommended).
    selection : {'validation', 'delta_constraint'}
        ``'validation'`` picks the joint (penalty scale, width) minimising the
        validation loss. ``'delta_constraint'`` implements eq. (2.5): the
        smallest penalty scale whose unpruned fit has ``||c||_{1,2} <=
        Delta_n``. It needs an explicit radius (``delta_mode`` other than
        ``'dual'``).
    random_state : int, optional
        Shared seed, so every candidate starts from the same initialisation.

    Returns
    -------
    TuningResult
    """
    if selection not in ("validation", "delta_constraint"):
        raise ValueError("selection must be 'validation' or 'delta_constraint'")
    t0 = time.time()
    base = dict(base_params or {})
    if selection == "delta_constraint" and base.get("delta_mode") == "dual":
        raise ValueError("selection='delta_constraint' needs an explicit Delta_n")
    cur = {"gamma": gamma_grid[0], "learning_rate": learning_rate_grid[0],
           "batch_size": batch_size_grid[0],
           "penalty_scale": penalty_scale_grid[0],
           "width_scale": width_scale_grid[0]}
    log: list[dict] = []

    def run(cand: dict, phase: str):
        params = {**base, **cand, "max_epochs": tuning_epochs,
                  "patience": tuning_patience, "prune": prune_aware,
                  "random_state": random_state, "verbose": 0}
        est = estimator(**params).fit(X, y, X_val, y_val)
        score = est.val_loss_["final"]
        log.append({"phase": phase, **cand, "val_loss": score,
                    "val_loss_unpruned": est.val_loss_["fit"],
                    "lambda_n": est.lambda_n_,
                    "l12_norm_pre_prune": est.l12_norm_pre_prune_,
                    "active_edges": est.n_active_edges_,
                    "sparsity_ratio": est.sparsity_ratio_,
                    "G_n": est.architecture_["G"], "W_n": est.architecture_["W"],
                    "delta_target": est.architecture_["delta_target"]})
        return score, est

    for name, grid in (("gamma", gamma_grid),
                       ("learning_rate", learning_rate_grid),
                       ("batch_size", batch_size_grid)):
        if len(grid) <= 1:
            continue
        best = float("inf")
        for v in grid:
            score, _ = run({**cur, name: v}, phase=name)
            if score < best:
                best, cur[name] = score, v
        if verbose:
            print(f"  {name} -> {cur[name]}  (val {best:.6f})")

    best, best_pw, feasible = float("inf"), None, []
    for pen in sorted(penalty_scale_grid):
        for cw in width_scale_grid:
            score, est = run({**cur, "penalty_scale": pen, "width_scale": cw},
                             phase="penalty_width")
            if score < best:
                best, best_pw = score, (pen, cw)
            if est.l12_norm_pre_prune_ <= est.architecture_["delta_target"]:
                feasible.append((pen, cw, score))
    if selection == "delta_constraint":
        if not feasible:
            raise RuntimeError(
                f"no penalty scale in {sorted(penalty_scale_grid)} brings "
                f"||c||_1,2 inside "
                f"Delta_n. Extend penalty_scale_grid upward or raise delta_scale.")
        pen, cw, _ = min(feasible, key=lambda t: (t[0], t[2]))
        cur["penalty_scale"], cur["width_scale"] = pen, cw
    else:
        cur["penalty_scale"], cur["width_scale"] = best_pw
    if verbose:
        print(f"  (penalty_scale, width_scale) -> ({cur['penalty_scale']}, "
              f"{cur['width_scale']})  [{selection}]")

    for name, grid in (("gamma", gamma_grid), ("learning_rate", learning_rate_grid),
                       ("penalty_scale", penalty_scale_grid)):
        g = sorted(grid)
        if len(g) > 1 and cur[name] in (g[0], g[-1]):
            edge = "lower" if cur[name] == g[0] else "upper"
            warnings.warn(f"selected {name}={cur[name]} sits at the {edge} edge "
                          f"of {g}. Consider extending the grid.", stacklevel=2)

    return TuningResult(best_params={**base, **cur}, log=log,
                        selection_time=time.time() - t0)
