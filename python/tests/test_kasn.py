"""Fast CPU tests for the kasn package."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sklearn.base import clone

import kasn
from kasn import (BSplineBasis, EmpiricalCDFTransformer, KASNClassifier,
                  KASNNetwork, KASNRegressor, architecture, check_conditions,
                  make_kasn_regression, tune_kasn)
from kasn.base import COMMON_PARAMS
from kasn.network import project_l1_ball

FAST = dict(depth=2, width=8, grid_size=6, max_epochs=25, patience=10,
            finetune_epochs=5, batch_size=64, device="cpu", random_state=0)


@pytest.fixture(scope="module")
def data():
    return make_kasn_regression(n=400, d=6, s=2, target="additive",
                                random_state=3)


# ------------------------------------------------------------- splines
def test_partition_of_unity_and_count():
    b = BSplineBasis(1, grid_size=10, spline_order=4, grid_range=(-0.5, 1.5))
    x = torch.linspace(-0.5, 1.5 - 1e-6, 501).unsqueeze(-1)
    B = b(x)[:, 0, :]
    assert B.shape[1] == 10 + 3
    assert torch.allclose(B.sum(dim=1), torch.ones(501), atol=1e-5)
    assert (B >= -1e-7).all()


@pytest.mark.parametrize("order", [2, 4, 5])
def test_spline_degree_is_order_minus_one(order):
    b = BSplineBasis(1, grid_size=8, spline_order=order, grid_range=(0.0, 1.0))
    lo = 3 / 8
    x = torch.linspace(lo + 1e-3, lo + 1 / 8 - 1e-3, 30,
                       dtype=torch.float64).unsqueeze(-1)
    B = b(x)[:, 0, :].numpy()
    xs = x.squeeze().numpy()
    exact = np.vander(xs, order)            # degree order - 1
    short = np.vander(xs, order - 1)        # degree order - 2
    res_exact = max(np.linalg.lstsq(exact, B[:, j], rcond=None)[1].sum()
                    for j in range(B.shape[1]) if B[:, j].any())
    res_short = max(np.linalg.lstsq(short, B[:, j], rcond=None)[1].sum()
                    for j in range(B.shape[1]) if B[:, j].any())
    assert res_exact < 1e-12 < res_short


# -------------------------------------------------------- preprocessing
def test_edf_transform():
    X = np.array([[3.0], [1.0], [2.0]])
    tr = EmpiricalCDFTransformer().fit(X)
    np.testing.assert_allclose(tr.transform(X).ravel(), [0.75, 0.25, 0.5])
    np.testing.assert_allclose(tr.transform([[0.0], [9.0]]).ravel(), [0.0, 0.75])


# -------------------------------------------------------------- network
def test_r_n_identity_and_residuals():
    net = KASNNetwork(input_dim=7, width=5, depth=3, grid_size=6)
    assert net.r_n == (7 + 1) * 5 + (3 - 1) * 5 ** 2
    assert [l.residual for l in net.layers] == [False, True, True, False]
    assert net(torch.rand(3, 7)).shape == (3, 1)


def test_l1_ball_projection_matches_definition():
    v = torch.tensor([3.0, 1.0, 0.5, 0.0])
    p = project_l1_ball(v, 2.0)
    assert abs(float(p.sum()) - 2.0) < 1e-6
    np.testing.assert_allclose(p.numpy(), [2.0, 0.0, 0.0, 0.0], atol=1e-6)
    assert torch.equal(project_l1_ball(v, 10.0), v)


def test_delta_ball_projection_feasible():
    net = KASNNetwork(4, 6, 3, 5)
    with torch.no_grad():
        net.layers[0].spline_weight.mul_(50)
    net.project_to_delta_ball(3.0)
    assert net.l12_value() <= 3.0 + 1e-4


def test_prune_cap_and_mask():
    net = KASNNetwork(4, 6, 3, 5)
    net.prune(0.0, max_active=10)
    assert net.n_nonzero_edges() == 10
    net.freeze_mask()
    with torch.no_grad():
        for l in net.layers:
            l.spline_weight.add_(1.0)
    net.apply_mask()
    assert net.n_nonzero_edges() == 10
    assert net.n_output_connected_edges() <= 10


def test_no_nan_gradient_at_pruned_edges():
    net = KASNNetwork(3, 4, 3, 5)
    net.prune(0.0, max_active=5)
    net.l12_norm().backward()
    assert all(not torch.isnan(l.spline_weight.grad).any() for l in net.layers)


# ----------------------------------------------------------- regressor
def test_fit_predict_learns(data):
    m = KASNRegressor(**{**FAST, "max_epochs": 150, "patience": 30,
                         "learning_rate": 5e-3, "delta_mode": "dual"}
                      ).fit(data.X[:300], data.y[:300])
    pred = m.predict(data.X[300:])
    assert pred.shape == (100,)
    assert m.score(data.X[300:], data.y[300:]) > 0.5
    rep = m.sparsity_report()
    assert rep["active_edges"] == m.n_active_edges_ <= rep["r_n"]
    assert m.delta_n_ == pytest.approx(m.l12_norm_)          # dual mode


def test_reproducible_and_global_rng_untouched(data):
    state = torch.get_rng_state()
    a = KASNRegressor(**FAST).fit(data.X, data.y).predict(data.X[:20])
    b = KASNRegressor(**FAST).fit(data.X, data.y).predict(data.X[:20])
    np.testing.assert_allclose(a, b)
    assert torch.equal(state, torch.get_rng_state())


def test_penalty_mode_is_default_and_feasible(data):
    m = KASNRegressor(**FAST).fit(data.X, data.y)
    assert m.delta_mode == "penalty"
    assert m.l12_norm_ <= m.delta_n_ + 1e-4
    assert m.lambda_n_ == pytest.approx(0.05 * m.architecture_["lambda_rate"])
    m2 = KASNRegressor(**{**FAST, "penalty": 0.3}).fit(data.X, data.y)
    assert m2.lambda_n_ == 0.3


def test_project_mode_respects_delta(data):
    m = KASNRegressor(**{**FAST, "delta_mode": "project", "delta_scale": 0.3,
                         "zeta_delta": 0.3}).fit(data.X, data.y)
    assert m.l12_norm_ <= m.delta_n_ + 1e-4


def test_max_active_edges(data):
    m = KASNRegressor(**{**FAST, "max_active_edges": 12}).fit(data.X, data.y)
    assert m.n_active_edges_ <= 12


def test_explicit_validation_and_weights(data):
    w = np.linspace(0.5, 1.5, 300)
    m = KASNRegressor(**FAST).fit(data.X[:300], data.y[:300],
                                  X_val=data.X[300:], y_val=data.y[300:],
                                  sample_weight=w)
    assert m.n_val_ == 100 and m.n_train_ == 300


def test_clone_and_save_load(tmp_path, data):
    m = KASNRegressor(**FAST).fit(data.X, data.y)
    assert clone(m).get_params() == m.get_params()
    path = tmp_path / "m.kasn"
    m.save(path)
    m2 = KASNRegressor.load(path, device="cpu")
    np.testing.assert_allclose(m.predict(data.X[:50]), m2.predict(data.X[:50]),
                               rtol=1e-6)
    assert m2.sparsity_report()["active_edges"] == m.n_active_edges_


def test_grid_coverage_shape(data):
    m = KASNRegressor(**FAST).fit(data.X, data.y)
    cov = m.grid_coverage(data.X)
    assert len(cov) == FAST["depth"] + 1
    assert cov[0]["frac_outside_grid_range"] == 0.0


def test_bad_params():
    with pytest.raises(ValueError):
        KASNRegressor(delta_mode="nope").fit(np.zeros((10, 2)), np.zeros(10))


# -------------------------------------------------- theory and tuning
def test_architecture_rates():
    a = architecture(2500, 100, gamma=0.35, depth=3, width_scale=5)
    assert (a["G"], a["L"], a["W"]) == (15, 3, 35)
    assert a["r_n"] == 101 * 35 + 2 * 35 ** 2


def test_check_conditions():
    assert check_conditions(0.4, zeta_delta=0.3, spline_order=4, m=2) == []
    with pytest.warns(UserWarning):
        check_conditions(0.6, m=2)


def test_tune_runs(data):
    res = tune_kasn(data.X, data.y, base_params={k: v for k, v in FAST.items()
                                                 if k not in ("max_epochs",
                                                              "patience")},
                    batch_size_grid=(64,), penalty_scale_grid=(0.05, 1.0),
                    width_scale_grid=(1,), tuning_epochs=10, verbose=0)
    assert res.best_params["penalty_scale"] in (0.05, 1.0)
    assert len(res.log) == 2


def test_dgp_dependence():
    d = make_kasn_regression(n=3000, d=3, s=2, rho=0.7, marginal="normal",
                             random_state=0)
    x = d.X[:, 0] - d.X[:, 0].mean()
    rho_hat = np.mean(x[1:] * x[:-1]) / np.mean(x ** 2)
    assert abs(rho_hat - 0.7) < 0.05
    assert d.X.shape == (3000, 3)


# ------------------------------------------------------- defaults and losses
def test_common_defaults_shared():
    for cls in (KASNRegressor, KASNClassifier):
        params = cls().get_params()
        for k, v in COMMON_PARAMS.items():
            assert params[k] == v, (cls.__name__, k)
    assert KASNRegressor().get_params()["spline_order"] == 4


@pytest.mark.parametrize("loss", ["huber", "quantile"])
def test_robust_losses(data, loss):
    m = KASNRegressor(**{**FAST, "loss": loss}).fit(data.X, data.y)
    assert np.all(np.isfinite(m.predict(data.X[:10])))


def test_quantile_ordering():
    d = make_kasn_regression(n=600, d=4, s=2, target="additive", rho=0.0,
                             noise_sd=1.0, random_state=1)
    kw = {**FAST, "max_epochs": 120, "patience": 30, "learning_rate": 5e-3,
          "loss": "quantile"}
    lo = KASNRegressor(**{**kw, "quantile": 0.1}).fit(d.X, d.y).predict(d.X)
    hi = KASNRegressor(**{**kw, "quantile": 0.9}).fit(d.X, d.y).predict(d.X)
    assert np.mean(hi > lo) > 0.95
    assert 0.0 < np.mean(d.y < lo) < 0.3 and 0.7 < np.mean(d.y < hi) < 1.0


def test_poisson():
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(600, 3))
    mu = np.exp(1.0 + np.sin(3 * X[:, 0]))
    y = rng.poisson(mu)
    m = KASNRegressor(**{**FAST, "loss": "poisson", "max_epochs": 120,
                         "patience": 30, "learning_rate": 5e-3}).fit(X, y)
    pred = m.predict(X)
    assert np.all(pred > 0)
    assert np.corrcoef(pred, mu)[0, 1] > 0.8
    with pytest.raises(ValueError):
        KASNRegressor(loss="poisson").fit(X, -np.ones(600))


def test_classifier_and_load(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(600, 3))
    p = 1 / (1 + np.exp(-4 * np.sin(3 * X[:, 0] - 1.5)))
    y = np.where(rng.uniform(size=600) < p, "yes", "no")
    c = KASNClassifier(**{**FAST, "max_epochs": 120, "patience": 30,
                          "learning_rate": 5e-3}).fit(X, y)
    assert list(c.classes_) == ["no", "yes"]
    proba = c.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert c.score(X, y) > 0.7
    path = tmp_path / "c.kasn"
    c.save(path)
    c2 = kasn.load(path, device="cpu")
    assert isinstance(c2, KASNClassifier)
    np.testing.assert_allclose(c2.predict_proba(X[:20]), proba[:20], rtol=1e-6)
    with pytest.raises(ValueError):
        KASNRegressor.load(path)
    with pytest.raises(ValueError):
        KASNClassifier(**FAST).fit(X, np.arange(600) % 3)
