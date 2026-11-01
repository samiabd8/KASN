"""Local nonparametric regression example (Section 4.1 of the KASN paper, scaled down).

Simulates the additive or compositional design, splits it chronologically
into train / validation / test (0.5 / 0.25 / 0.25), fits the KASN and a
validation-tuned Lasso, and reports the columns of Tables 1-4: RMSE and R^2
on each fold, the held-out L2 error against f_0, and the active edge count.

Runs on a laptop CPU in under a minute. ``--tune`` runs the prune-aware
hyperparameter search. Paper scale: ``--n 5000 --d 100 --s 30``.

    python examples/nonparametric_regression.py --target compositional
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score

from kasn import (EmpiricalCDFTransformer, KASNRegressor, check_conditions,
                  make_kasn_regression, target_function, tune_kasn)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--target", default="compositional",
                   choices=["additive", "compositional", "deep"])
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--d", type=int, default=50)
    p.add_argument("--s", type=int, default=10)
    p.add_argument("--rho", type=float, default=0.7)
    p.add_argument("--gamma", type=float, default=0.4)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--tune", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    args, _ = p.parse_known_args()          # Jupyter-safe
    return args


def metrics(name, pred, y, f0, idx, extra):
    row = {"method": name}
    for fold in ("train", "val", "test"):
        i = idx[fold]
        row[f"{fold}_rmse"] = float(np.sqrt(mean_squared_error(y[i], pred[i])))
        row[f"{fold}_r2"] = float(r2_score(y[i], pred[i]))
    i = idx["test"]
    row["l2_error"] = float(np.sqrt(np.mean((pred[i] - f0[i]) ** 2)))
    row.update(extra)
    return row


def main() -> None:
    a = parse_args()
    check_conditions(a.gamma, m=2)
    data = make_kasn_regression(n=a.n, d=a.d, s=a.s, target=a.target,
                                rho=a.rho, random_state=a.seed)
    X, y, f0 = data.X, data.y, data.f0
    n_tr, n_va = int(0.5 * a.n), int(0.25 * a.n)
    idx = {"train": np.arange(n_tr), "val": np.arange(n_tr, n_tr + n_va),
           "test": np.arange(n_tr + n_va, a.n)}
    Xtr, ytr = X[idx["train"]], y[idx["train"]]
    Xva, yva = X[idx["val"]], y[idx["val"]]
    print(f"Design: target={a.target} n={a.n} d={a.d} S={a.s} rho_X={a.rho} "
          f"(chronological split {n_tr}/{n_va}/{a.n - n_tr - n_va})")

    params = dict(gamma=a.gamma, depth=a.depth, max_epochs=a.epochs,
                  patience=a.patience, device=a.device, random_state=a.seed)
    if a.tune:
        res = tune_kasn(Xtr, ytr, Xva, yva, base_params=params,
                        penalty_scale_grid=(0.05, 0.1, 1.0),
                        gamma_grid=(0.3, 0.4), width_scale_grid=(3, 5),
                        batch_size_grid=(32,), tuning_epochs=40,
                        random_state=a.seed)
        params = res.best_params
        print(f"Tuned in {res.selection_time:.0f}s: gamma={params['gamma']} "
              f"penalty_scale={params['penalty_scale']} "
              f"width_scale={params['width_scale']}")

    rows = []
    t0 = time.time()
    kasn = KASNRegressor(**params, verbose=1).fit(Xtr, ytr, Xva, yva)
    rep = kasn.sparsity_report()
    rows.append(metrics("KASN", kasn.predict(X), y, f0, idx, {
        "parameters": f"{rep['active_edges']:,}/{rep['r_n']:,}",
        "fit_s": time.time() - t0}))

    t0 = time.time()
    cdf = EmpiricalCDFTransformer().fit(Xtr)
    best = min((Lasso(alpha=al, max_iter=10_000).fit(cdf.transform(Xtr), ytr)
                for al in (1e-5, 1e-3, 1e-2, 5e-2, 1e-1, 1.0)),
               key=lambda m: mean_squared_error(yva, m.predict(cdf.transform(Xva))))
    nz = int(np.sum(np.abs(best.coef_) > 1e-6))
    rows.append(metrics("LASSO", best.predict(cdf.transform(X)), y, f0, idx,
                        {"parameters": f"{nz}/{a.d}", "fit_s": time.time() - t0}))

    oracle = target_function(X, a.s, a.target)
    rows.append(metrics("Oracle", oracle, y, f0, idx,
                        {"parameters": "-", "fit_s": 0.0}))

    print(f"\n{'method':8s} {'trRMSE':>7s} {'trR2':>6s} {'vaRMSE':>7s} "
          f"{'vaR2':>6s} {'teRMSE':>7s} {'teR2':>6s} {'||f-f0||':>8s} "
          f"{'parameters':>14s} {'fit(s)':>7s}")
    for r in rows:
        print(f"{r['method']:8s} {r['train_rmse']:7.4f} {r['train_r2']:6.4f} "
              f"{r['val_rmse']:7.4f} {r['val_r2']:6.4f} {r['test_rmse']:7.4f} "
              f"{r['test_r2']:6.4f} {r['l2_error']:8.4f} {r['parameters']:>14s} "
              f"{r['fit_s']:7.1f}")

    print(f"\nKASN: sparsity ratio {rep['sparsity_ratio']:.3f}, "
          f"output-connected edges {rep['output_connected_edges']:,}, "
          f"Delta_n {rep['delta_n']:.2f} ({rep['delta_mode']}), "
          f"lambda_n {rep['lambda_n']:.2e}, "
          f"L_n={rep['L_n']} W_n={rep['W_n']} G_n={rep['G_n']}")
    print("Grid coverage on the test fold (share of layer inputs outside I_eps):")
    for c in kasn.grid_coverage(X[idx["test"]]):
        print(f"  layer {c['layer']}: range [{c['min']:+.3f}, {c['max']:+.3f}]"
              f"  outside grid {c['frac_outside_grid_range']:.4f}"
              f"  outside support {c['frac_outside_support']:.4f}")


if __name__ == "__main__":
    main()
