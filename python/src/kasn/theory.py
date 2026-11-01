"""Architecture rates and parameter restrictions from Section 2.4 of the paper.

The defaults follow eq. (2.6)-(2.8):

    L_n ~ log n,   W_n ~ s_0 ~ log n,   G_n ~ n^gamma,
    Delta_n ~ n^zeta_delta,   zeta_delta > gamma / 2,
    1 / (4(m + 1)) < gamma < 1 / 2,   k >= m + 1.
"""

from __future__ import annotations

import math
import warnings
from typing import Optional

__all__ = ["architecture", "check_conditions"]


def architecture(n: int, d: int, *, gamma: float = 0.4,
                 grid_size: Optional[int] = None, depth: Optional[int] = None,
                 width: Optional[int] = None, width_scale: int = 5,
                 zeta_delta: float = 0.4, delta_scale: float = 1.0) -> dict:
    """Resolve the KASN architecture for a training sample of size ``n``.

    Explicit ``grid_size``, ``depth`` or ``width`` override the rate-based
    rules ``G_n = max(5, floor(n^gamma))``, ``L_n = max(3, floor(log n))``
    hidden layers and ``W_n = width_scale * floor(log n)``. The network has
    ``L_n + 1`` layers of edges and ``r_n = (d + 1) W + (L_n - 1) W^2``.

    Returns
    -------
    dict with keys ``n, d, G, L, W, r_n, delta_target, lambda_rate``, where
    ``lambda_rate = sqrt(log(L_n W_n^2) / n)``, so that the Group Lasso weight
    is ``lambda_n = C_lambda * lambda_rate`` for penalty scale ``C_lambda``.
    """
    n, d = int(n), int(d)
    if n < 3:
        raise ValueError(f"need at least 3 training observations, got {n}")
    log_n = math.log(n)
    G = int(grid_size) if grid_size is not None else max(5, int(n ** gamma))
    L = int(depth) if depth is not None else max(3, int(log_n))
    W = (int(width) if width is not None
         else max(1, int(width_scale) * int(math.floor(log_n))))
    r_n = (d + 1) * W + (L - 1) * W ** 2
    return {
        "n": n, "d": d, "G": G, "L": L, "W": W, "r_n": int(r_n),
        "delta_target": float(max(5.0, delta_scale * n ** zeta_delta)),
        "lambda_rate": float(math.sqrt(math.log(L * W ** 2 + 1e-8) / n)),
    }


def check_conditions(gamma: float, *, zeta_delta: Optional[float] = None,
                     spline_order: int = 4, m: int = 2,
                     strict: bool = False) -> list[str]:
    """Check the restrictions the rate theorems are stated under.

    Parameters
    ----------
    gamma : float
        Grid exponent in ``G_n ~ n^gamma``.
    zeta_delta : float, optional
        Radius exponent. Only relevant when Delta_n is imposed explicitly.
    spline_order : int
        De Boor order ``k``.
    m : int
        Assumed edge smoothness (edges in ``W_2^{m+1}``).
    strict : bool
        Raise ``ValueError`` on a violation instead of warning.

    Returns
    -------
    list of str
        One message per violated condition (empty if all hold).
    """
    msgs = []
    lo = 1.0 / (4.0 * (m + 1))
    if not lo < gamma < 0.5:
        msgs.append(f"gamma={gamma} violates 1/(4(m+1))={lo:.4f} < gamma < 1/2 "
                    f"for m={m}")
    if zeta_delta is not None and not zeta_delta > gamma / 2:
        msgs.append(f"zeta_delta={zeta_delta} must exceed gamma/2={gamma / 2:.4f}")
    if spline_order < m + 1:
        msgs.append(f"spline order k={spline_order} must be >= m+1={m + 1}")
    for msg in msgs:
        if strict:
            raise ValueError(msg)
        warnings.warn(msg, stacklevel=2)
    return msgs
