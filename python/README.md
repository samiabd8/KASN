# kasn

Kolmogorov-Arnold Sieve Networks. PyTorch backend, scikit-learn API.

## Install

```bash
pip install kasn
pip install -e "python[test]" && pytest python/tests   # development
```

## Usage

```python
from kasn import KASNRegressor, KASNClassifier, make_kasn_regression, load

sim = make_kasn_regression(n=2000, d=50, s=10, target="compositional", random_state=0)
X, y = sim.X, sim.y
m = KASNRegressor(depth=3, random_state=0).fit(X[:1000], y[:1000], X[1000:1500], y[1000:1500])
m.predict(X[1500:]); m.sparsity_report(); m.grid_coverage(X[1500:])
m.save("fit.kasn"); m = load("fit.kasn")

KASNRegressor(loss="quantile", quantile=0.9)
KASNRegressor(loss="huber", huber_delta=1.0)
KASNRegressor(loss="poisson")
KASNClassifier()                       # binary logistic
```

`tune_kasn` runs the prune-aware greedy search. `examples/nonparametric_regression.py` reproduces the layout of Tables 1-4.

## Parameters

| Parameter | Definition | Default |
|---|---|---|
| `gamma`, `grid_size` | `G_n = max(5, floor(n^gamma))` | `gamma = 0.4` |
| `depth` | `L_n` hidden layers, `L_n + 1` edge layers | `max(3, floor(log n))` |
| `width`, `width_scale` | `W_n = C_W floor(log n)` | `C_W = 5` |
| | `r_n = (d + 1) W_n + (L_n - 1) W_n^2` | |
| `spline_order` | de Boor order `k` | 4 |
| `grid_range` | `I_eps` | `(-0.05, 1.05)` |
| `penalty` | `lambda_n` | `C_lambda sqrt(log(L_n W_n^2) / n)` |
| `penalty_scale` | `C_lambda` | 0.05 |
| `delta_mode` | `'penalty'`, `'project'`, `'dual'` | `'penalty'` |
| `zeta_delta`, `delta_scale` | `Delta_n = max(5, delta_scale n^zeta_delta)` | 0.4, 1.0 |
| `max_active_edges` | `s_n` cap | none |
| `max_epochs`, `patience` | | 500, 50 |

`n` is the training fold size.
