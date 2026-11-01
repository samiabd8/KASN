#' Fit a Kolmogorov-Arnold Sieve Network
#'
#' Fits the KASN sieve extremum estimator for the chosen criterion. Inputs
#' are normalised by their training empirical CDFs and a Group Lasso penalty
#' acts on the B-spline coefficient vector of every edge. After training,
#' edges below \eqn{\Delta_n / r_n} are pruned and the survivors fine-tuned.
#'
#' Model parameters are passed through `...` to the Python estimator, see
#' [kasn_defaults()]. The main ones are:
#'
#' * `gamma` (0.4): \eqn{G_n = \max(5, \lfloor n^\gamma \rfloor)}.
#' * `depth` (`NULL`): \eqn{L_n} hidden layers, \eqn{L_n + 1} edge layers.
#'   Defaults to \eqn{\max(3, \lfloor \log n \rfloor)}.
#' * `width_scale` (5): \eqn{W_n = C_W \lfloor \log n \rfloor}.
#' * `spline_order` (4): de Boor order \eqn{k} (cubic).
#' * `grid_range` (`c(-0.05, 1.05)`): \eqn{I_\epsilon}.
#' * `penalty_scale` (0.05): \eqn{C_\lambda} in
#'   \eqn{\lambda_n = C_\lambda \sqrt{\log(L_n W_n^2)/n}}. `penalty` sets
#'   \eqn{\lambda_n} directly.
#' * `delta_mode` ("penalty"), `zeta_delta` (0.4): radius
#'   \eqn{\Delta_n = n^{\zeta_\delta}}.
#' * `max_active_edges` (`NULL`): cap \eqn{s_n}.
#' * `max_epochs` (500), `patience` (50), `learning_rate` (1e-3),
#'   `batch_size` (32), `validation_fraction` (0.2).
#' * `quantile` (0.5), `huber_delta` (1): loss parameters.
#' * `device` ("auto"), `random_state` (`NULL`), `verbose` (0).
#'
#' @param x Numeric matrix or data frame of covariates (rows in time order).
#' @param y Response. Numeric for regression losses, two-level factor,
#'   character, logical or 0/1 for `loss = "logistic"`.
#' @param x_val,y_val Optional validation fold. Defaults to the last
#'   `validation_fraction` of the rows.
#' @param weights Optional observation weights on the training loss.
#' @param loss "squared", "huber", "quantile", "poisson" or "logistic".
#' @param formula A model formula such as `y ~ .`.
#' @param data,val_data Data frames for the formula interface.
#' @param ... Estimator parameters (see Details).
#'
#' @return An object of class `kasn`.
#' @seealso [predict.kasn()], [kasn_tune()], [kasn_save()]
#' @examplesIf identical(Sys.getenv("NOT_CRAN"), "true") && kasn_available()
#' sim <- kasn_simulate(n = 600, d = 10, s = 3, target = "additive", seed = 1)
#' fit <- kasn(sim$x[1:450, ], sim$y[1:450], depth = 2, max_epochs = 60,
#'             random_state = 0)
#' fit
#' mean((predict(fit, sim$x[451:600, ]) - sim$f0[451:600])^2)
#' @export
kasn <- function(x, ...) {
  UseMethod("kasn")
}

#' @rdname kasn
#' @export
kasn.default <- function(x, y, x_val = NULL, y_val = NULL, weights = NULL,
                         loss = "squared", ...) {
  .check_kasn()
  loss <- match.arg(loss, .losses)
  x <- .as_matrix(x)
  enc <- .encode_response(y, nrow(x), loss)
  params <- .coerce_params(list(...))
  cls <- .estimator_class(loss)
  if (loss != "logistic") params$loss <- loss
  .check_param_names(params, cls)
  est <- do.call(cls, params)
  args <- list(x, enc$y)
  if (!is.null(x_val) || !is.null(y_val)) {
    if (is.null(x_val) || is.null(y_val)) {
      stop("supply both `x_val` and `y_val`, or neither", call. = FALSE)
    }
    x_val <- .as_matrix(x_val)
    args$X_val <- x_val
    args$y_val <- .encode_response(y_val, nrow(x_val), loss, enc$levels)$y
  }
  if (!is.null(weights)) {
    args$sample_weight <- .as_numpy(weights, nrow(x))
  }
  do.call(est$fit, args)
  .new_kasn(est, call = match.call(), loss = loss, levels = enc$levels,
            feature_names = colnames(x))
}

#' @rdname kasn
#' @export
kasn.formula <- function(formula, data, val_data = NULL, weights = NULL,
                         loss = "squared", ...) {
  mf <- stats::model.frame(formula, data)
  tt <- attr(mf, "terms")
  y <- stats::model.response(mf)
  x <- .model_matrix(tt, mf)
  xlevels <- stats::.getXlevels(tt, mf)
  x_val <- y_val <- NULL
  if (!is.null(val_data)) {
    mfv <- stats::model.frame(tt, val_data, xlev = xlevels)
    y_val <- stats::model.response(mfv)
    x_val <- .model_matrix(tt, mfv, attr(x, "contrasts"))
  }
  fit <- kasn.default(x, y, x_val = x_val, y_val = y_val, weights = weights,
                      loss = loss, ...)
  fit$call <- match.call()
  fit$terms <- tt
  fit$xlevels <- xlevels
  fit$contrasts <- attr(x, "contrasts")
  fit
}

#' Predict from a fitted KASN
#'
#' @param object A `kasn` fit.
#' @param newdata Matrix or data frame with the training covariates (or the
#'   variables in the formula).
#' @param type "response" gives the conditional mean, quantile or Poisson
#'   mean, and for `loss = "logistic"` the probability of the second level.
#'   "class" gives predicted labels (logistic only).
#' @param ... Unused.
#' @return Numeric vector, or a factor for `type = "class"`.
#' @export
predict.kasn <- function(object, newdata, type = c("response", "class"),
                         ...) {
  .check_alive(object)
  type <- match.arg(type)
  if (missing(newdata) || is.null(newdata)) {
    stop("`newdata` is required", call. = FALSE)
  }
  x <- .design_matrix(object, newdata)
  if (object$loss != "logistic") {
    if (type == "class") {
      stop('type = "class" needs loss = "logistic"', call. = FALSE)
    }
    return(as.numeric(reticulate::py_to_r(object$py$predict(x))))
  }
  p <- reticulate::py_to_r(object$py$predict_proba(x))[, 2]
  if (type == "response") return(as.numeric(p))
  factor(object$levels[(p >= 0.5) + 1L], levels = object$levels)
}

#' @export
print.kasn <- function(x, ...) {
  .check_alive(x)
  a <- .py_attr(x$py, "architecture_")
  rep <- reticulate::py_to_r(x$py$sparsity_report())
  v <- .py_attr(x$py, "val_loss_")
  cat("<kasn> Kolmogorov-Arnold Sieve Network, loss = ", x$loss, "\n", sep = "")
  cat(sprintf("  n = %d, d = %d, L_n = %d, W_n = %d, G_n = %d, k = %d\n",
              a$n, a$d, a$L, a$W, a$G, rep$spline_order))
  cat(sprintf("  active edges %s / %s (sparsity %.3f)\n",
              format(rep$active_edges, big.mark = ","),
              format(rep$r_n, big.mark = ","), rep$sparsity_ratio))
  cat(sprintf("  lambda_n = %.3g, ||c||_{1,2} = %.3f, Delta_n = %.3f (%s)\n",
              rep$lambda_n, rep$l12_norm, rep$delta_n, rep$delta_mode))
  if (!is.null(v$final)) {
    cat(sprintf("  validation loss: fit %.4f, pruned %.4f, final %.4f\n",
                v$fit, v$pruned, v$final))
  }
  invisible(x)
}

#' Summarise a fitted KASN
#'
#' @param object A `kasn` fit.
#' @param ... Unused.
#' @return A `summary.kasn` list: architecture, sparsity report, validation
#'   losses and fit time.
#' @export
summary.kasn <- function(object, ...) {
  .check_alive(object)
  out <- list(
    call = object$call,
    loss = object$loss,
    architecture = .py_attr(object$py, "architecture_"),
    sparsity = reticulate::py_to_r(object$py$sparsity_report()),
    val_loss = .py_attr(object$py, "val_loss_"),
    fit_time = .py_attr(object$py, "fit_time_")
  )
  class(out) <- "summary.kasn"
  out
}

#' @export
print.summary.kasn <- function(x, ...) {
  cat("KASN fit, loss = ", x$loss, "\n\nArchitecture\n", sep = "")
  str(x$architecture, give.head = FALSE, no.list = TRUE)
  cat("\nSparsity\n")
  str(x$sparsity, give.head = FALSE, no.list = TRUE)
  cat("\nValidation loss\n")
  str(x$val_loss, give.head = FALSE, no.list = TRUE)
  cat(sprintf("\nFit time: %.1f s\n", x$fit_time))
  invisible(x)
}

#' Diagnostics of a fitted KASN
#'
#' `kasn_history()`: per-epoch training log. `kasn_edge_norms()`: per-layer
#' matrices of \eqn{\|c_j\|_2}. `kasn_grid_coverage()`: share of each
#' layer's inputs outside \eqn{I_\epsilon} and outside the knot support.
#'
#' @param object A `kasn` fit.
#' @param newdata Covariates at which to evaluate the grid coverage.
#' @return A data frame, or a list of matrices.
#' @name kasn_diagnostics
NULL

#' @rdname kasn_diagnostics
#' @export
kasn_history <- function(object) {
  .check_alive(object)
  .rows_to_df(.py_attr(object$py, "history_"))
}

#' @rdname kasn_diagnostics
#' @export
kasn_edge_norms <- function(object) {
  .check_alive(object)
  lapply(reticulate::py_to_r(object$py$edge_norms()), as.matrix)
}

#' @rdname kasn_diagnostics
#' @export
kasn_grid_coverage <- function(object, newdata) {
  .check_alive(object)
  x <- .design_matrix(object, newdata)
  .rows_to_df(reticulate::py_to_r(object$py$grid_coverage(x)))
}

#' Default KASN parameters
#'
#' @param loss Loss whose estimator parameters to list.
#' @return Named list of parameters and defaults.
#' @export
kasn_defaults <- function(loss = "squared") {
  .check_kasn()
  loss <- match.arg(loss, .losses)
  reticulate::py_to_r(.estimator_class(loss)()$get_params())
}
