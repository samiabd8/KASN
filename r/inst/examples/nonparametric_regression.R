# Local nonparametric regression example (Section 4.1 of the KASN paper,
# scaled down). Mirrors python/examples/nonparametric_regression.py.
#
#   Rscript nonparametric_regression.R
#
# Paper scale: n = 5000, d = 100, s = 30.

library(kasn)

target <- "compositional"
n <- 2000; d <- 50; s <- 10
sim <- kasn_simulate(n = n, d = d, s = s, target = target, rho = 0.7, seed = 0)

# Chronological 0.5 / 0.25 / 0.25 split
tr <- seq_len(n / 2)
va <- (n / 2 + 1):(3 * n / 4)
te <- (3 * n / 4 + 1):n

fit <- kasn(sim$x[tr, ], sim$y[tr], x_val = sim$x[va, ], y_val = sim$y[va],
            depth = 3, random_state = 0, verbose = 1)
print(fit)

pred <- predict(fit, sim$x)
rmse <- function(i) sqrt(mean((sim$y[i] - pred[i])^2))
r2 <- function(i) 1 - mean((sim$y[i] - pred[i])^2) / mean((sim$y[i] - mean(sim$y[i]))^2)
cat(sprintf("\nKASN   test RMSE %.4f   test R2 %.4f   ||f_hat - f0|| %.4f\n",
            rmse(te), r2(te), sqrt(mean((pred[te] - sim$f0[te])^2))))
oracle_r2 <- 1 - mean((sim$y[te] - sim$f0[te])^2) /
  mean((sim$y[te] - mean(sim$y[te]))^2)
cat(sprintf("Oracle test R2 %.4f\n", oracle_r2))

cat("\nGrid coverage on the test fold:\n")
print(kasn_grid_coverage(fit, sim$x[te, ]))

# saveRDS() cannot hold the Python object
f <- tempfile(fileext = ".rds")
kasn_save(fit, f)
fit2 <- kasn_load(f)
stopifnot(isTRUE(all.equal(predict(fit2, sim$x[te, ]), pred[te],
                           tolerance = 1e-6)))

# Prune-aware tuning on the validation fold
# res <- kasn_tune(sim$x[tr, ], sim$y[tr], sim$x[va, ], sim$y[va],
#                  base_params = list(depth = 3), gamma_grid = c(0.3, 0.4),
#                  batch_size_grid = 32, penalty_scale_grid = c(0.05, 0.1, 1),
#                  width_scale_grid = c(3, 5), tuning_epochs = 40)
# fit <- do.call(kasn, c(list(sim$x[tr, ], sim$y[tr], x_val = sim$x[va, ],
#                             y_val = sim$y[va]), res$best_params))
