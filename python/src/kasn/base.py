"""Training machinery shared by :class:`KASNRegressor` and :class:`KASNClassifier`."""

from __future__ import annotations

import copy
import math
import time
import warnings
from typing import Optional

import numpy as np
import torch
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted

from ._utils import plain, resolve_device, seeded
from .losses import Loss
from .network import KASNNetwork
from .preprocessing import EmpiricalCDFTransformer
from .theory import architecture

_DELTA_MODES = ("penalty", "project", "dual")
_THRESHOLDS = ("delta_over_r", "relative", "fixed")
_FORMAT_VERSION = 2
_CHUNK = 8192

# Parameters shared by every KASN estimator, with their defaults.
COMMON_PARAMS = dict(
    gamma=0.4, grid_size=None, depth=None, width=None, width_scale=5,
    spline_order=4, grid_range=(-0.05, 1.05), penalty=None,
    penalty_scale=0.05, delta_mode="penalty", zeta_delta=0.4,
    delta_scale=1.0, delta_weight=1.0, prune=True,
    prune_threshold="delta_over_r", prune_value=None, max_active_edges=None,
    finetune_epochs=30, finetune_lr_scale=0.5, finetune_patience=15,
    max_epochs=500, patience=50, learning_rate=1e-3, batch_size=32,
    weight_decay=0.0, lr_step_size=None, lr_gamma=0.5, grad_clip=500.0,
    validation_fraction=0.2, shuffle=False, device="auto", amp=True,
    random_state=None, verbose=0,
)

COMMON_PARAMS_DOC = """
    gamma : float, default=0.4
        Grid exponent, ``G_n = max(5, floor(n^gamma))``.
    grid_size : int, optional
        Fix ``G_n`` directly (overrides ``gamma``).
    depth : int, optional
        ``L_n``, the number of hidden layers. The network has ``L_n + 1``
        layers of edges. Defaults to ``max(3, floor(log n))``.
    width : int, optional
        Hidden width ``W_n``. Defaults to ``width_scale * floor(log n)``.
    width_scale : int, default=5
        ``C_W`` in ``W_n = C_W floor(log n)``.
    spline_order : int, default=4
        De Boor order ``k`` of the edge B-splines (4 is cubic).
    grid_range : tuple of float, default=(-0.05, 1.05)
        Padded spline domain ``I_eps = [-eps, 1 + eps]``.
    penalty : float, optional
        Group Lasso weight ``lambda_n`` on ``||c||_{1,2}``. When given it is
        used as is and ``penalty_scale`` is ignored.
    penalty_scale : float, default=0.05
        Scale ``C_lambda`` in ``lambda_n = C_lambda sqrt(log(L_n W_n^2) / n)``.
    delta_mode : {'penalty', 'project', 'dual'}, default='penalty'
        Treatment of the radius ``Delta_n = max(5, delta_scale n^zeta_delta)``.
        ``'penalty'`` adds ``delta_weight * max(||c||_{1,2} - Delta_n, 0)^2``
        and projects onto the ball after training. ``'project'`` projects
        after every step. ``'dual'`` imposes no radius and reports the
        realised ``||c||_{1,2}`` as ``Delta_n``.
    zeta_delta, delta_scale : float, default=0.4, 1.0
    delta_weight : float, default=1.0
        Weight of the quadratic hinge in ``'penalty'`` mode.
    prune : bool, default=True
        Prune and fine-tune after the main fit.
    prune_threshold : {'delta_over_r', 'relative', 'fixed'}
        ``Delta_n / r_n``, ``prune_value`` (default 0.01) times the largest
        edge norm, or the absolute ``prune_value`` (default 1e-4).
    prune_value : float, optional
    max_active_edges : int, optional
        Cap ``s_n`` on ``||c||_{0,2}`` at pruning (largest edges kept).
    finetune_epochs, finetune_lr_scale, finetune_patience : default 30, 0.5, 15
        Fine-tuning with pruned edges frozen at zero. Discarded if it
        validates worse than the pruned fit.
    max_epochs, patience : int, default=500, 50
        Training length and early-stopping patience on the validation loss.
    learning_rate, batch_size, weight_decay : default 1e-3, 32, 0.0
        AdamW settings. ``batch_size=None`` is full batch.
    lr_step_size, lr_gamma : default None, 0.5
        Step decay of the learning rate every ``lr_step_size`` epochs
        (``None`` keeps it constant).
    grad_clip : float, default=500.0
    validation_fraction : float, default=0.2
        Chronological tail held out when ``X_val`` is not passed (0 disables
        validation).
    shuffle : bool, default=False
    device : str, default='auto'
    amp : bool, default=True
        Mixed precision on CUDA.
    random_state : int, optional
        Seeds initialisation and shuffling in a forked RNG.
    verbose : int, default=0
"""

FITTED_DOC = """
    Attributes
    ----------
    network_ : KASNNetwork
    architecture_ : dict
        ``n, d, G, L, W, r_n, delta_target, lambda_rate``.
    lambda_n_ : float
        Group Lasso weight used in training.
    delta_n_ : float
        Radius of the class containing the reported estimator.
    l12_norm_, l12_norm_pre_prune_ : float
        ``||c_hat||_{1,2}`` of the reported and of the unpruned fit.
    n_active_edges_ : int
        ``s_hat_n = ||c_hat||_{0,2}``.
    sparsity_ratio_ : float
        ``1 - s_hat_n / r_n``.
    val_loss_ : dict
        Validation loss after the fit, after pruning and of the final fit.
    history_ : list of dict
        Per-epoch log.
"""


class BaseKASN(BaseEstimator):
    """Shared fit, prune, persistence and diagnostics. Not for direct use."""

    # ------------------------------------------------------- subclass hooks
    def _make_loss(self) -> Loss:
        raise NotImplementedError

    def _encode_target(self, y) -> tuple[np.ndarray, float]:
        """Fit the response transform on the training fold.

        Returns the training target on the scale the loss sees and the
        constant offset added to the network output.
        """
        raise NotImplementedError

    def _transform_target(self, y) -> np.ndarray:
        raise NotImplementedError

    def _check_y(self, y, n: int) -> np.ndarray:
        raise NotImplementedError

    # ================================================================ checks
    def _check_params(self) -> None:
        if self.delta_mode not in _DELTA_MODES:
            raise ValueError(f"delta_mode must be one of {_DELTA_MODES}")
        if self.prune_threshold not in _THRESHOLDS:
            raise ValueError(f"prune_threshold must be one of {_THRESHOLDS}")
        if self.penalty is not None and self.penalty < 0:
            raise ValueError("penalty must be non-negative")
        if self.penalty_scale < 0:
            raise ValueError("penalty_scale must be non-negative")
        if not 0 <= self.validation_fraction < 1:
            raise ValueError("validation_fraction must lie in [0, 1)")
        if self.max_active_edges is not None and int(self.max_active_edges) < 1:
            raise ValueError("max_active_edges must be a positive integer")
        if self.grid_size is None and not 0 < self.gamma < 0.5:
            warnings.warn(f"gamma={self.gamma} lies outside (0, 1/2), where "
                          f"Theorem 3 does not apply", stacklevel=3)

    def _check_X(self, X, reset: bool) -> np.ndarray:
        names = None
        if hasattr(X, "columns"):
            names = np.asarray([str(c) for c in X.columns], dtype=object)
        X = check_array(X, dtype=np.float64)
        if reset:
            self.n_features_in_ = int(X.shape[1])
            if names is not None:
                self.feature_names_in_ = names
            elif hasattr(self, "feature_names_in_"):
                del self.feature_names_in_
        elif X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, the model was "
                             f"fitted on {self.n_features_in_}")
        return X

    # =================================================================== fit
    def fit(self, X, y, X_val=None, y_val=None, sample_weight=None):
        """Fit the KASN.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        X_val, y_val : array-like, optional
            Validation fold for early stopping, pruning decisions and
            fine-tune reversion. Defaults to the chronological tail of size
            ``validation_fraction``.
        sample_weight : array-like of shape (n_samples,), optional
            Weights on the training loss.
        """
        t0 = time.time()
        self._check_params()
        X = self._check_X(X, reset=True)
        y = self._check_y(y, X.shape[0])
        sw = None
        if sample_weight is not None:
            sw = np.asarray(sample_weight, dtype=np.float64).ravel()
            if sw.shape[0] != X.shape[0]:
                raise ValueError("sample_weight length does not match X")

        if X_val is not None or y_val is not None:
            if X_val is None or y_val is None:
                raise ValueError("pass both X_val and y_val, or neither")
            X_tr, y_tr, sw_tr = X, y, sw
            X_va = self._check_X(X_val, reset=False)
            y_va = self._check_y(y_val, X_va.shape[0])
        elif self.validation_fraction > 0:
            n_val = int(round(X.shape[0] * self.validation_fraction))
            n_val = min(max(n_val, 1), X.shape[0] - 3)
            X_tr, y_tr, X_va, y_va = X[:-n_val], y[:-n_val], X[-n_val:], y[-n_val:]
            sw_tr = None if sw is None else sw[:-n_val]
        else:
            X_tr, y_tr, sw_tr, X_va, y_va = X, y, sw, None, None

        arch = architecture(X_tr.shape[0], X_tr.shape[1], gamma=self.gamma,
                            grid_size=self.grid_size, depth=self.depth,
                            width=self.width, width_scale=self.width_scale,
                            zeta_delta=self.zeta_delta,
                            delta_scale=self.delta_scale)
        self.architecture_ = arch
        self.lambda_n_ = float(self.penalty if self.penalty is not None
                               else self.penalty_scale * arch["lambda_rate"])
        self.n_train_ = int(X_tr.shape[0])
        self.n_val_ = 0 if X_va is None else int(X_va.shape[0])
        dev = resolve_device(self.device)
        self.device_ = str(dev)
        self._loss = self._make_loss()

        self.x_transformer_ = EmpiricalCDFTransformer().fit(X_tr)
        t_tr, self.offset_ = self._encode_target(y_tr)

        def to_t(a, col=False):
            t = torch.as_tensor(a, dtype=torch.float32, device=dev)
            return t.reshape(-1, 1) if col else t

        Xtr_t = to_t(self.x_transformer_.transform(X_tr))
        ytr_t = to_t(t_tr, col=True)
        wtr_t = None if sw_tr is None else to_t(sw_tr, col=True)
        if X_va is not None:
            Xva_t = to_t(self.x_transformer_.transform(X_va))
            yva_t = to_t(self._transform_target(y_va), col=True)
        else:
            Xva_t = yva_t = None

        self.history_ = []
        with seeded(self.random_state, dev):
            net = KASNNetwork(X_tr.shape[1], arch["W"], arch["L"], arch["G"],
                              spline_order=self.spline_order,
                              grid_range=self.grid_range).to(dev)
            self.network_ = net
            if self.verbose:
                print(f"KASN[{self._loss.name}]: n={arch['n']} d={arch['d']} "
                      f"L_n={arch['L']} W_n={arch['W']} G_n={arch['G']} "
                      f"r_n={arch['r_n']:,} lambda_n={self.lambda_n_:.2e} "
                      f"device={dev}")
            val_fit = self._train(net, Xtr_t, ytr_t, wtr_t, Xva_t, yva_t,
                                  epochs=int(self.max_epochs),
                                  lr=float(self.learning_rate),
                                  patience=self.patience, phase="fit")
            self.l12_norm_pre_prune_ = net.l12_value()
            val_pruned = val_final = val_fit
            self.prune_threshold_ = None
            if self.prune:
                val_pruned, val_final = self._prune_and_finetune(
                    net, Xtr_t, ytr_t, wtr_t, Xva_t, yva_t, val_fit)
        self._finalise_delta(net)
        if Xva_t is not None:
            val_final = self._loss_value(net, Xva_t, yva_t)

        self.val_loss_ = {"fit": val_fit, "pruned": val_pruned,
                          "final": val_final}
        self.l12_norm_ = net.l12_value()
        self.n_active_edges_ = net.n_nonzero_edges()
        self.n_output_connected_edges_ = net.n_output_connected_edges()
        self.sparsity_ratio_ = 1.0 - self.n_active_edges_ / net.r_n
        self.fit_time_ = time.time() - t0
        if self.verbose:
            print(f"KASN: active edges {self.n_active_edges_:,}/{net.r_n:,} "
                  f"(sparsity {self.sparsity_ratio_:.4f}), "
                  f"||c||_1,2={self.l12_norm_:.3f}, Delta_n={self.delta_n_:.3f}, "
                  f"{self.fit_time_:.1f}s")
        return self

    # ------------------------------------------------------------- training
    def _penalty_terms(self, net: KASNNetwork) -> torch.Tensor:
        R = net.l12_norm()
        pen = self.lambda_n_ * R
        if self.delta_mode == "penalty":
            excess = torch.clamp(R - self.architecture_["delta_target"], min=0.0)
            pen = pen + float(self.delta_weight) * excess ** 2
        return pen

    def _forward(self, net: KASNNetwork, X: torch.Tensor) -> torch.Tensor:
        amp = bool(self.amp) and X.device.type == "cuda"
        parts = []
        for i in range(0, X.shape[0], _CHUNK):
            with torch.autocast(device_type=X.device.type, enabled=amp):
                parts.append(net(X[i:i + _CHUNK]).float())
        return torch.cat(parts) + self.offset_

    @torch.no_grad()
    def _loss_value(self, net: KASNNetwork, X: torch.Tensor,
                    y: torch.Tensor) -> float:
        net.eval()
        return float(self._loss.per_sample(self._forward(net, X), y).mean())

    def _train(self, net, Xtr, ytr, wtr, Xva, yva, *, epochs: int, lr: float,
               patience: Optional[int], phase: str) -> Optional[float]:
        """AdamW with early stopping on the validation loss. Restores the best."""
        dev = Xtr.device
        amp = bool(self.amp) and dev.type == "cuda"
        opt = torch.optim.AdamW(net.parameters(), lr=lr,
                                weight_decay=float(self.weight_decay))
        sched = (None if self.lr_step_size is None else
                 torch.optim.lr_scheduler.StepLR(
                     opt, step_size=int(self.lr_step_size),
                     gamma=float(self.lr_gamma)))
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        n = Xtr.shape[0]
        bs = n if self.batch_size is None else min(int(self.batch_size), n)
        has_val = Xva is not None
        best, best_state, best_epoch, wait = math.inf, None, -1, 0
        log_every = max(1, epochs // 10)
        r_n = net.r_n
        offset = self.offset_

        for epoch in range(epochs):
            net.train()
            order = (torch.randperm(n, device=dev) if self.shuffle
                     else torch.arange(n, device=dev))
            total, seen = 0.0, 0
            for start in range(0, n, bs):
                idx = order[start:start + bs]
                xb, yb = Xtr[idx], ytr[idx]
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=dev.type, enabled=amp):
                    f = net(xb).float() + offset
                    per = self._loss.per_sample(f, yb)
                    fit_loss = (per.mean() if wtr is None
                                else (per * wtr[idx]).mean())
                    loss = fit_loss + self._penalty_terms(net)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(),
                                               float(self.grad_clip))
                scaler.step(opt)
                scaler.update()
                if self.delta_mode == "project":
                    net.project_to_delta_ball(self.architecture_["delta_target"])
                net.apply_mask()
                total += float(fit_loss.detach()) * xb.shape[0]
                seen += xb.shape[0]
            if sched is not None:
                sched.step()

            train_loss = total / max(seen, 1)
            val = self._loss_value(net, Xva, yva) if has_val else math.nan
            R = net.l12_value()
            delta_eff = (R if self.delta_mode == "dual"
                         else self.architecture_["delta_target"])
            with torch.no_grad():
                above = int(sum((nm > delta_eff / r_n).sum()
                                for nm in net.edge_norms()))
            self.history_.append({
                "phase": phase, "epoch": epoch, "train_loss": train_loss,
                "val_loss": val, "l12_norm": R, "active_above_threshold": above,
                "lr": float(opt.param_groups[0]["lr"])})

            score = val if has_val else train_loss
            if score < best:
                best, best_epoch, wait = score, epoch, 0
                best_state = copy.deepcopy(net.state_dict())
            else:
                wait += 1
            if self.verbose and (epoch % log_every == 0 or epoch == epochs - 1):
                print(f"  [{phase}] ep {epoch:4d}/{epochs} train={train_loss:.5f}"
                      f" val={val:.5f} best={best:.5f} (ep {best_epoch})"
                      f" R_n={R:.2f} above_thr={above}/{r_n}")
            if has_val and patience is not None and wait >= int(patience):
                if self.verbose:
                    print(f"  [{phase}] early stop at ep {epoch} "
                          f"(best ep {best_epoch})")
                break

        if best_state is not None and has_val:
            net.load_state_dict(best_state)
            net.apply_mask()
        return best if has_val else None

    def _prune_and_finetune(self, net, Xtr, ytr, wtr, Xva, yva,
                            val_fit: Optional[float]):
        """Prune at the chosen threshold, then fine-tune the survivors."""
        has_val = Xva is not None
        with torch.no_grad():
            if self.prune_threshold == "delta_over_r":
                delta_eff = (net.l12_value() if self.delta_mode == "dual"
                             else self.architecture_["delta_target"])
                thr = delta_eff / net.r_n
            elif self.prune_threshold == "relative":
                frac = 0.01 if self.prune_value is None else float(self.prune_value)
                thr = frac * max(float(n.max()) for n in net.edge_norms())
            else:
                thr = 1e-4 if self.prune_value is None else float(self.prune_value)
        self.prune_threshold_ = float(thr)
        n_zero = net.prune(thr, self.max_active_edges)
        val_pruned = self._loss_value(net, Xva, yva) if has_val else None
        val_final = val_pruned
        if self.verbose:
            print(f"KASN: pruned at {thr:.3e}, {n_zero:,}/{net.r_n:,} edges zero"
                  + (f", val {val_fit:.5f} -> {val_pruned:.5f}" if has_val else ""))

        if int(self.finetune_epochs) > 0:
            net.freeze_mask()
            pruned_state = copy.deepcopy(net.state_dict())
            self._train(net, Xtr, ytr, wtr, Xva, yva,
                        epochs=int(self.finetune_epochs),
                        lr=float(self.learning_rate) * float(self.finetune_lr_scale),
                        patience=self.finetune_patience, phase="finetune")
            net.apply_mask()
            if has_val:
                val_final = self._loss_value(net, Xva, yva)
                if val_final > val_pruned:
                    net.load_state_dict(pruned_state)
                    val_final = self._loss_value(net, Xva, yva)
                    if self.verbose:
                        print("KASN: fine-tuning validated worse, reverted to "
                              "the pruned fit")
            assert net.r_n - net.n_nonzero_edges() >= n_zero, \
                "prune mask was not respected during fine-tuning"
            net.clear_mask()
        return val_pruned, val_final

    def _finalise_delta(self, net: KASNNetwork) -> None:
        """Make the reported estimator a member of K_n (Lemma 5 (i))."""
        R = net.l12_value()
        if self.delta_mode == "dual":
            self.delta_n_ = R
            return
        delta = self.architecture_["delta_target"]
        if R > delta:
            net.project_to_delta_ball(delta)
            shrink = net.l12_value() / R
            if shrink < 0.9:
                warnings.warn(
                    f"the feasibility projection rescaled ||c||_1,2 to "
                    f"{shrink:.2f} of its fitted value (Delta_n={delta:.2f}). "
                    f"Train longer, raise delta_scale or use "
                    f"delta_mode='project'.",
                    stacklevel=3)
        self.delta_n_ = delta

    # ============================================================ prediction
    def _decision(self, X) -> np.ndarray:
        """Network output plus offset (link scale), shape (n,)."""
        check_is_fitted(self, "network_")
        X = self._check_X(X, reset=False)
        dev = next(self.network_.parameters()).device
        Xt = torch.as_tensor(self.x_transformer_.transform(X),
                             dtype=torch.float32, device=dev)
        with torch.inference_mode():
            self.network_.eval()
            out = self._forward(self.network_, Xt).cpu().numpy().ravel()
        return out.astype(np.float64)

    # =========================================================== diagnostics
    def sparsity_report(self) -> dict:
        """Edge counts, thresholds and norms of the fitted network."""
        check_is_fitted(self, "network_")
        net = self.network_
        thr = self.delta_n_ / net.r_n
        with torch.no_grad():
            above = int(sum((n > thr).sum() for n in net.edge_norms()))
        return plain({
            "r_n": net.r_n, "p_n": net.n_coefficients,
            "active_edges": self.n_active_edges_,
            "output_connected_edges": self.n_output_connected_edges_,
            "active_above_delta_over_r": above,
            "sparsity_ratio": self.sparsity_ratio_,
            "prune_threshold": self.prune_threshold_,
            "delta_n": self.delta_n_, "delta_mode": self.delta_mode,
            "l12_norm": self.l12_norm_,
            "l12_norm_pre_prune": self.l12_norm_pre_prune_,
            "lambda_n": self.lambda_n_,
            "G_n": net.grid_size, "L_n": net.depth, "W_n": net.width,
            "spline_order": net.spline_order,
        })

    def edge_norms(self) -> list[np.ndarray]:
        """Per-layer ``||c_j||_2`` matrices, shape ``(out, in)`` each."""
        check_is_fitted(self, "network_")
        with torch.no_grad():
            return [n.detach().cpu().numpy() for n in self.network_.edge_norms()]

    def grid_coverage(self, X) -> list[dict]:
        """Share of each layer's inputs outside ``grid_range`` and outside the
        knot support (where every basis function is zero)."""
        check_is_fitted(self, "network_")
        X = self._check_X(X, reset=False)
        dev = next(self.network_.parameters()).device
        Xt = torch.as_tensor(self.x_transformer_.transform(X),
                             dtype=torch.float32, device=dev)
        out = []
        for i, (layer, z) in enumerate(zip(self.network_.layers,
                                           self.network_.layer_inputs(Xt))):
            lo, hi = layer.basis.grid_range
            slo, shi = layer.basis.support
            out.append({
                "layer": i, "min": float(z.min()), "max": float(z.max()),
                "frac_outside_grid_range": float(((z < lo) | (z > hi)).float().mean()),
                "frac_outside_support": float(((z < slo) | (z > shi)).float().mean()),
            })
        return out

    # ========================================================== persistence
    def _extra_state(self) -> dict:
        return {}

    def _set_extra_state(self, state: dict) -> None:
        pass

    def save(self, path) -> None:
        """Save the fitted estimator (readable with ``kasn.load``)."""
        check_is_fitted(self, "network_")
        from . import __version__
        params = self.get_params()
        params["grid_range"] = list(params["grid_range"])
        fitted = {k: getattr(self, k) for k in (
            "n_features_in_", "n_train_", "n_val_", "offset_", "lambda_n_",
            "delta_n_", "l12_norm_", "l12_norm_pre_prune_", "n_active_edges_",
            "n_output_connected_edges_", "sparsity_ratio_", "prune_threshold_",
            "val_loss_", "fit_time_", "device_")}
        payload = {
            "format_version": _FORMAT_VERSION, "kasn_version": __version__,
            "class": type(self).__name__,
            "params": plain(params), "architecture": plain(self.architecture_),
            "fitted": plain(fitted), "extra": plain(self._extra_state()),
            "history": plain(self.history_),
            "feature_names": (None if not hasattr(self, "feature_names_in_")
                              else [str(c) for c in self.feature_names_in_]),
            "x_sorted": torch.from_numpy(
                np.ascontiguousarray(self.x_transformer_.sorted_values_)),
            "state_dict": {k: v.detach().cpu()
                           for k, v in self.network_.state_dict().items()},
        }
        torch.save(payload, path)

    @classmethod
    def _from_payload(cls, payload: dict, device: Optional[str]):
        params = dict(payload["params"])
        params["grid_range"] = tuple(params["grid_range"])
        est = cls(**params)
        for k, v in payload["fitted"].items():
            setattr(est, k, v)
        est._set_extra_state(payload["extra"])
        est.architecture_ = payload["architecture"]
        est.history_ = payload["history"]
        if payload["feature_names"] is not None:
            est.feature_names_in_ = np.asarray(payload["feature_names"],
                                               dtype=object)
        xs = payload["x_sorted"].numpy()
        tr = EmpiricalCDFTransformer()
        tr.sorted_values_, tr.n_train_, tr.n_features_in_ = (
            xs, int(xs.shape[0]), int(xs.shape[1]))
        est.x_transformer_ = tr
        est._loss = est._make_loss()
        a = est.architecture_
        dev = resolve_device(device if device is not None else est.device)
        net = KASNNetwork(a["d"], a["W"], a["L"], a["G"],
                          spline_order=params["spline_order"],
                          grid_range=params["grid_range"])
        net.load_state_dict(payload["state_dict"])
        est.network_ = net.to(dev)
        est.device_ = str(dev)
        return est

    @classmethod
    def load(cls, path, device: Optional[str] = None):
        """Load an estimator written by :meth:`save`."""
        payload = _read_payload(path)
        if payload["class"] != cls.__name__:
            raise ValueError(f"{path} holds a {payload['class']}, not a "
                             f"{cls.__name__}. Use kasn.load().")
        return cls._from_payload(payload, device)


def _read_payload(path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != _FORMAT_VERSION:
        raise ValueError(f"unsupported file format "
                         f"{payload.get('format_version')!r}")
    return payload
