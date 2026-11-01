skip_if_no_kasn <- function() {
  skip_on_cran()
  if (!kasn_available()) skip("Python module 'kasn' not available")
}

fast <- list(depth = 2, width = 8, grid_size = 6, max_epochs = 25,
             patience = 10, finetune_epochs = 5, batch_size = 64,
             device = "cpu", random_state = 0)

sim_data <- function() {
  kasn_simulate(n = 400, d = 6, s = 2, target = "additive", seed = 3)
}

test_that("matrix interface fits and predicts", {
  skip_if_no_kasn()
  sim <- sim_data()
  fit <- do.call(kasn, c(list(sim$x[1:300, ], sim$y[1:300]), fast))
  expect_s3_class(fit, "kasn")
  p <- predict(fit, sim$x[301:400, ])
  expect_length(p, 100)
  expect_true(all(is.finite(p)))
  s <- summary(fit)
  expect_lte(s$sparsity$active_edges, s$sparsity$r_n)
  expect_lte(s$sparsity$l12_norm, s$sparsity$delta_n + 1e-4)
  expect_output(print(fit), "loss = squared")
  h <- kasn_history(fit)
  expect_true(all(c("phase", "epoch", "train_loss", "val_loss") %in% names(h)))
  expect_length(kasn_edge_norms(fit), fast$depth + 1)
  expect_equal(nrow(kasn_grid_coverage(fit, sim$x)), fast$depth + 1)
  expect_error(predict(fit, sim$x, type = "class"))
})

test_that("formula interface with validation data and factors", {
  skip_if_no_kasn()
  sim <- sim_data()
  df <- data.frame(y = sim$y, sim$x)
  df$g <- factor(rep(c("a", "b"), length.out = nrow(df)))
  fit <- do.call(kasn, c(list(y ~ ., data = df[1:250, ],
                              val_data = df[251:320, ]), fast))
  expect_length(predict(fit, df[321:400, ]), 80)
  expect_error(predict(fit, df[321:400, 1:3]))
})

test_that("quantile, poisson and logistic losses", {
  skip_if_no_kasn()
  sim <- sim_data()
  q <- do.call(kasn, c(list(sim$x, sim$y, loss = "quantile", quantile = 0.9),
                       fast))
  expect_true(all(is.finite(predict(q, sim$x[1:10, ]))))
  yc <- rpois(400, exp(1 + 0.3 * sim$f0))
  pf <- do.call(kasn, c(list(sim$x, yc, loss = "poisson"), fast))
  expect_true(all(predict(pf, sim$x[1:10, ]) > 0))
  yb <- factor(ifelse(sim$f0 > median(sim$f0), "hi", "lo"))
  cf <- do.call(kasn, c(list(sim$x, yb, loss = "logistic"), fast))
  pr <- predict(cf, sim$x)
  expect_true(all(pr >= 0 & pr <= 1))
  cl <- predict(cf, sim$x, type = "class")
  expect_s3_class(cl, "factor")
  expect_identical(levels(cl), c("hi", "lo"))
  expect_error(kasn(sim$x, sim$y, loss = "logistic"), "two levels")
})

test_that("integer coercion, bad names and reproducibility", {
  skip_if_no_kasn()
  sim <- sim_data()
  a <- do.call(kasn, c(list(sim$x, sim$y), fast))
  b <- do.call(kasn, c(list(sim$x, sim$y), fast))
  expect_equal(predict(a, sim$x[1:20, ]), predict(b, sim$x[1:20, ]))
  expect_error(kasn(sim$x, sim$y, not_a_param = 1), "unknown KASN parameter")
})

test_that("save and load round trip", {
  skip_if_no_kasn()
  sim <- sim_data()
  yb <- sim$y > 0
  for (args in list(list(sim$x, sim$y), list(sim$x, yb, loss = "logistic"))) {
    fit <- do.call(kasn, c(args, fast))
    f <- tempfile(fileext = ".rds")
    kasn_save(fit, f)
    fit2 <- kasn_load(f, device = "cpu")
    expect_equal(predict(fit, sim$x[1:30, ]), predict(fit2, sim$x[1:30, ]),
                 tolerance = 1e-6)
  }
})

test_that("tuning returns usable parameters", {
  skip_if_no_kasn()
  sim <- sim_data()
  base <- fast[setdiff(names(fast), c("max_epochs", "patience"))]
  res <- kasn_tune(sim$x, sim$y, base_params = base, batch_size_grid = 64,
                   penalty_scale_grid = c(0.05, 1), width_scale_grid = 1,
                   tuning_epochs = 10, verbose = FALSE)
  expect_true(res$best_params$penalty_scale %in% c(0.05, 1))
  expect_equal(nrow(res$log), 2)
  fit <- do.call(kasn, c(list(sim$x, sim$y), res$best_params,
                         list(max_epochs = 10)))
  expect_s3_class(fit, "kasn")
})
