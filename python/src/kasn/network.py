"""KASN architecture: stacked layers of univariate B-spline edges.

The network realises the class in eq. (2.4) of the paper,

    f(x; c) = Phi_{L_n} o (Phi_{L_n-1} + h) o ... o (Phi_1 + h) o Phi_0 (x),

so ``L_n`` hidden layers give ``L_n + 1`` layers of edges. Each ``Phi_l`` is a
matrix of univariate B-spline edges, nodes perform summation, and
``h(x) = x`` is the parameter-free identity residual on the ``L_n - 1``
width-to-width layers. The coefficient vector ``c_j`` of each edge is a
group for the l_{1,2} (Group Lasso) penalty and the Delta_n constraint.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .splines import BSplineBasis

__all__ = ["KASNLayer", "KASNNetwork", "project_l1_ball"]


def project_l1_ball(v: torch.Tensor, radius: float) -> torch.Tensor:
    """Euclidean projection of a non-negative vector onto ``{v : sum(v) <= R}``.

    Sort-and-threshold algorithm: sort in descending order, find the largest
    ``rho`` with ``u_rho - (cumsum_rho - R) / rho > 0`` and soft-threshold at
    ``theta = (cumsum_rho - R) / rho``. Returns ``v`` unchanged when feasible.
    """
    if radius < 0:
        raise ValueError("radius must be non-negative")
    if float(v.sum()) <= radius:
        return v
    u, _ = torch.sort(v, descending=True)
    css = torch.cumsum(u, dim=0)
    rho_idx = torch.arange(1, v.numel() + 1, device=v.device, dtype=v.dtype)
    cond = u - (css - radius) / rho_idx > 0
    rho = int(torch.nonzero(cond).max().item()) + 1
    theta = (css[rho - 1] - radius) / rho
    return torch.clamp(v - theta, min=0.0)


class KASNLayer(nn.Module):
    """One layer ``Phi_l``: an ``out x in`` matrix of univariate B-spline edges.

    Parameters
    ----------
    in_features, out_features : int
        Layer fan-in and fan-out. The layer holds ``in * out`` edges.
    grid_size : int
        ``G_n``.
    spline_order : int
        De Boor order ``k`` of the edge splines.
    grid_range : sequence of two floats
        Padded spline domain ``I_eps``.
    residual : bool
        Add the identity residual ``h(x) = x``. Only active when
        ``in_features == out_features``.
    """

    def __init__(self, in_features: int, out_features: int, grid_size: int,
                 spline_order: int, grid_range: Sequence[float],
                 residual: bool) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.residual = bool(residual) and self.in_features == self.out_features
        self.basis = BSplineBasis(self.in_features, grid_size, spline_order,
                                  grid_range)
        self.grid_size = int(grid_size)
        self.num_basis = self.basis.num_basis
        self.spline_weight = nn.Parameter(
            torch.empty(self.out_features, self.in_features, self.num_basis))
        self._mask: Optional[torch.Tensor] = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.spline_weight,
                         -0.5 / self.grid_size, 0.5 / self.grid_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = self.basis(x).reshape(x.shape[0], -1)
        out = F.linear(b, self.spline_weight.reshape(self.out_features, -1))
        return out + x if self.residual else out

    def edge_norms(self) -> torch.Tensor:
        """``||c_j||_2`` for every edge, shape ``(out_features, in_features)``."""
        return torch.linalg.vector_norm(self.spline_weight, ord=2, dim=2)

    @property
    def n_edges(self) -> int:
        return self.out_features * self.in_features


class KASNNetwork(nn.Module):
    """Deep KASN with ``depth = L_n`` hidden layers of constant ``width``.

    ``Phi_0`` maps ``input_dim -> width``, the ``L_n - 1`` residual layers
    ``Phi_1, ..., Phi_{L_n - 1}`` map ``width -> width`` and ``Phi_{L_n}`` maps
    ``width -> 1``. The number of potential edges is

        r_n = (d + 1) W + (L_n - 1) W^2,

    which :meth:`_check_r_n` verifies against the realised layers.
    """

    def __init__(self, input_dim: int, width: int, depth: int, grid_size: int,
                 spline_order: int = 4,
                 grid_range: Sequence[float] = (-0.05, 1.05)) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth (L_n) must be >= 1, got {depth}")
        if width < 1 or input_dim < 1:
            raise ValueError("width and input_dim must be positive")
        self.input_dim = int(input_dim)
        self.width = int(width)
        self.depth = int(depth)
        self.grid_size = int(grid_size)
        self.spline_order = int(spline_order)
        self.grid_range = (float(grid_range[0]), float(grid_range[1]))
        args = dict(grid_size=self.grid_size, spline_order=self.spline_order,
                    grid_range=self.grid_range)
        layers = [KASNLayer(self.input_dim, self.width, residual=False, **args)]
        layers += [KASNLayer(self.width, self.width, residual=True, **args)
                   for _ in range(self.depth - 1)]
        layers.append(KASNLayer(self.width, 1, residual=False, **args))
        self.layers = nn.ModuleList(layers)
        self._check_r_n()

    # ------------------------------------------------------------ structure
    @property
    def r_n(self) -> int:
        """Total number of potential edges."""
        return sum(layer.n_edges for layer in self.layers)

    @property
    def n_coefficients(self) -> int:
        """``p_n``: total number of spline coefficients."""
        return sum(layer.spline_weight.numel() for layer in self.layers)

    def _check_r_n(self) -> None:
        d, W, L = self.input_dim, self.width, self.depth
        theory = (d + 1) * W + (L - 1) * W ** 2
        assert theory == self.r_n, (
            f"r_n mismatch: (d+1)W + (L-1)W^2 = {theory}, layers hold {self.r_n}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    @torch.no_grad()
    def layer_inputs(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Inputs reaching each layer, used for the grid-coverage diagnostic."""
        out = []
        for layer in self.layers:
            out.append(x)
            x = layer(x)
        return out

    # ------------------------------------------------------- regularisation
    def edge_norms(self) -> list[torch.Tensor]:
        return [layer.edge_norms() for layer in self.layers]

    def l12_norm(self) -> torch.Tensor:
        """``R = ||c||_{1,2} = sum_j ||c_j||_2`` as a differentiable tensor."""
        return sum(layer.edge_norms().sum() for layer in self.layers)

    @torch.no_grad()
    def l12_value(self) -> float:
        """``||c||_{1,2}`` as a Python float, outside the autograd graph."""
        return float(self.l12_norm())

    @torch.no_grad()
    def project_to_delta_ball(self, delta: float) -> int:
        """Project the coefficients onto ``{c : ||c||_{1,2} <= delta}``.

        The l_{1,2} ball projection factorises: project the vector of group
        norms onto the l_1 ball of radius ``delta`` and rescale each group to
        its projected norm. Groups thresholded to zero are removed, so the
        projection doubles as a selection step. Returns the number of edges
        the projection set to zero.
        """
        norms = torch.cat([n.reshape(-1) for n in self.edge_norms()])
        if float(norms.sum()) <= delta:
            return 0
        projected = project_l1_ball(norms, float(delta))
        scale = torch.where(norms > 1e-12, projected / norms.clamp_min(1e-12),
                            torch.zeros_like(norms))
        offset, zeroed = 0, 0
        for layer in self.layers:
            k = layer.n_edges
            s = scale[offset:offset + k]
            layer.spline_weight.mul_(s.view(layer.out_features,
                                            layer.in_features, 1))
            zeroed += int(((s == 0) & (norms[offset:offset + k] > 0)).sum())
            offset += k
        assert offset == norms.numel()
        return zeroed

    # --------------------------------------------------------------- pruning
    @torch.no_grad()
    def prune(self, threshold: float,
              max_active: Optional[int] = None) -> int:
        """Zero every edge with ``||c_j||_2 <= threshold``.

        If ``max_active`` is given, additionally keep only the ``max_active``
        largest surviving edges, which enforces ``||c||_{0,2} <= s_n``.
        Returns the number of edges that are zero after pruning.
        """
        for layer in self.layers:
            layer.spline_weight[layer.edge_norms() <= threshold] = 0.0
        if max_active is not None:
            norms = torch.cat([n.reshape(-1) for n in self.edge_norms()])
            n_active = int((norms > 0).sum())
            if n_active > max_active:
                cut = torch.topk(norms, int(max_active)).values.min()
                for layer in self.layers:
                    layer.spline_weight[layer.edge_norms() < cut] = 0.0
        return self.r_n - self.n_nonzero_edges()

    @torch.no_grad()
    def freeze_mask(self) -> int:
        """Freeze the current zero pattern so later training cannot revive it."""
        n_masked = 0
        for layer in self.layers:
            layer._mask = layer.edge_norms() == 0
            n_masked += int(layer._mask.sum())
        return n_masked

    @torch.no_grad()
    def apply_mask(self) -> None:
        """Re-zero masked edges. Called after every optimiser step."""
        for layer in self.layers:
            if layer._mask is not None:
                layer.spline_weight[layer._mask] = 0.0

    def clear_mask(self) -> None:
        for layer in self.layers:
            layer._mask = None

    # ----------------------------------------------------------- diagnostics
    @torch.no_grad()
    def n_nonzero_edges(self) -> int:
        """Realised ``||c||_{0,2}``: edges with a non-zero coefficient vector."""
        return int(sum((n > 0).sum() for n in self.edge_norms()))

    @torch.no_grad()
    def n_output_connected_edges(self) -> int:
        """Non-zero edges lying on a path to the output.

        Traced backwards from the output node. A hidden node stays alive if it
        feeds an alive node through a non-zero edge or through the identity
        residual. Non-zero edges into dead nodes do not affect the output
        (the H1-N1 case in Figure 1 of the paper).
        """
        alive_out = torch.ones(1, dtype=torch.bool,
                               device=self.layers[-1].spline_weight.device)
        count = 0
        for layer in reversed(self.layers):
            active = layer.edge_norms() > 0
            used = active & alive_out.unsqueeze(1)
            count += int(used.sum())
            alive_in = used.any(dim=0)
            if layer.residual:
                alive_in = alive_in | alive_out
            alive_out = alive_in
        return count
