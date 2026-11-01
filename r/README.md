# kasn (R)

reticulate interface to the `kasn` Python package. The Python dependency is declared with `reticulate::py_require()` (reticulate >= 1.41).

```r
remotes::install_github("samiabd8/kasn", subdir = "r")
library(kasn)

sim <- kasn_simulate(n = 2000, d = 50, s = 10, target = "compositional", seed = 0)
fit <- kasn(sim$x[1:1000, ], sim$y[1:1000], x_val = sim$x[1001:1500, ],
            y_val = sim$y[1001:1500], depth = 3, random_state = 0)
predict(fit, sim$x[1501:2000, ])
summary(fit); kasn_history(fit); kasn_grid_coverage(fit, sim$x)
kasn_save(fit, "fit.rds"); fit <- kasn_load("fit.rds")

kasn(x, y, loss = "quantile", quantile = 0.9)
kasn(x, y, loss = "poisson")
kasn(x, yfactor, loss = "logistic")      # predict(type = "response" | "class")
kasn(y ~ ., data = df, val_data = df_val)
kasn_tune(x, y, x_val, y_val, penalty_scale_grid = c(0.05, 0.1, 1))
```

Parameters pass through `...` to the Python estimator, see `kasn_defaults()`.

Own Python environment: set `RETICULATE_PYTHON` before the first call, or `install_kasn()`. Local checkout: `Sys.setenv(KASN_PY_REQUIREMENT = "kasn @ file:///path/to/kasn/python")`.

Example: `inst/examples/nonparametric_regression.R`.
