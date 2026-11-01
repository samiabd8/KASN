"""Uniform B-spline basis used for every KASN edge."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

__all__ = ["BSplineBasis"]


class BSplineBasis(nn.Module):
    """B-spline basis of de Boor order ``k`` (degree ``k - 1``) on a uniform grid.

    The grid has ``grid_size`` equal intervals on ``grid_range`` and is
    extended by ``k - 1`` knots on each side, so that the ``grid_size + k - 1``
    basis functions form a partition of unity on ``grid_range``. Evaluation
    uses the Cox-de Boor recursion.

    Parameters
    ----------
    in_features : int
        Number of input coordinates (one basis per coordinate, shared grid).
    grid_size : int
        Number of intervals ``G_n`` on ``grid_range``.
    spline_order : int
        De Boor order ``k`` (``k = 4`` is cubic). The theory requires
        ``k >= m + 1`` for edge smoothness ``m``.
    grid_range : sequence of two floats
        The padded domain ``I_eps = [-eps, 1 + eps]``.

    Notes
    -----
    The research scripts used the efficient-KAN convention (``spline_order``
    is the degree), so their ``KASN_SPLINE_ORDER = 4`` is ``spline_order = 5``
    here.
    """

    def __init__(self, in_features: int, grid_size: int, spline_order: int,
                 grid_range: Sequence[float] = (-0.05, 1.05)) -> None:
        super().__init__()
        if spline_order < 1:
            raise ValueError(f"spline_order must be >= 1, got {spline_order}")
        if grid_size < 1:
            raise ValueError(f"grid_size must be >= 1, got {grid_size}")
        lo, hi = float(grid_range[0]), float(grid_range[1])
        if not hi > lo:
            raise ValueError(f"grid_range must be increasing, got {grid_range}")
        self.in_features = int(in_features)
        self.grid_size = int(grid_size)
        self.spline_order = int(spline_order)
        self.degree = self.spline_order - 1
        self.grid_range = (lo, hi)
        self.num_basis = self.grid_size + self.degree
        h = (hi - lo) / self.grid_size
        knots = (torch.arange(-self.degree, self.grid_size + self.degree + 1,
                              dtype=torch.float32) * h + lo)
        self.register_buffer("grid",
                             knots.expand(self.in_features, -1).contiguous())

    @property
    def mesh(self) -> float:
        """Knot spacing ``h = (hi - lo) / G``."""
        lo, hi = self.grid_range
        return (hi - lo) / self.grid_size

    @property
    def support(self) -> tuple[float, float]:
        """Interval outside which every basis function is exactly zero."""
        lo, hi = self.grid_range
        return lo - self.degree * self.mesh, hi + self.degree * self.mesh

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the basis.

        Parameters
        ----------
        x : Tensor of shape (batch, in_features)

        Returns
        -------
        Tensor of shape (batch, in_features, num_basis)
        """
        grid = self.grid.to(x.dtype)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for p in range(1, self.degree + 1):
            left = (x - grid[:, :-(p + 1)]) / (
                grid[:, p:-1] - grid[:, :-(p + 1)]).clamp_min(1e-8)
            right = (grid[:, p + 1:] - x) / (
                grid[:, p + 1:] - grid[:, 1:(-p)]).clamp_min(1e-8)
            bases = left * bases[:, :, :-1] + right * bases[:, :, 1:]
        return bases.contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        out = self.b_splines(x.reshape(-1, self.in_features))
        return out.reshape(*shape[:-1], self.in_features, self.num_basis)
