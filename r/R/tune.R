#' Prune-aware hyperparameter search
#'
#' Greedy coordinate search on a validation fold over `gamma`,
#' `learning_rate`, `batch_size`, then `penalty_scale` and `width_scale`
#' jointly. Candidates are scored on the validation loss of the pruned and
#' fine-tuned fit.
#'
#' @inheritParams kasn
#' @param base_params Named list of fixed estimator parameters.
#' @param gamma_grid,learning_rate_grid,batch_size_grid,penalty_scale_grid,width_scale_grid
#'   Candidate values. A grid of length one is held fixed.
#' @param tuning_epochs,tuning_patience Epochs and early-stopping patience
#'   per candidate.
#' @param selection "validation", or "delta_constraint" (smallest penalty
#'   scale with \eqn{\|c\|_{1,2} \le \Delta_n}, eq. 2.5).
#' @param seed Shared initialisation seed.
#' @param verbose Print the selected value of each coordinate.
#' @return A list with `best_params` (pass to [kasn()] with [do.call()]), the
#'   candidate `log` and the `selection_time` in seconds.
#' @export
kasn_tune <- function(x, y, x_val = NULL, y_val = NULL, loss = "squared",
                      base_params = list(), gamma_grid = 0.4,
                      learning_rate_grid = 1e-3, batch_size_grid = c(16, 32, 64),
                      penalty_scale_grid = c(0.05, 0.1, 1, 5),
                      width_scale_grid = c(5, 15), tuning_epochs = 50,
                      tuning_patience = 10, selection = "validation",
                      seed = 0, verbose = TRUE) {
  .check_kasn()
  loss <- match.arg(loss, .losses)
  x <- .as_matrix(x)
  enc <- .encode_response(y, nrow(x), loss)
  if (!is.null(x_val)) {
    x_val <- .as_matrix(x_val)
    y_val <- .encode_response(y_val, nrow(x_val), loss, enc$levels)$y
  }
  cls <- .estimator_class(loss)
  base_params <- .coerce_params(base_params)
  if (loss != "logistic") base_params$loss <- loss
  .check_param_names(base_params, cls)
  res <- kasn_py$tune_kasn(
    x, enc$y, x_val, y_val, estimator = cls,
    base_params = reticulate::dict(base_params),
    gamma_grid = as.list(as.numeric(gamma_grid)),
    learning_rate_grid = as.list(as.numeric(learning_rate_grid)),
    batch_size_grid = as.list(as.integer(batch_size_grid)),
    penalty_scale_grid = as.list(as.numeric(penalty_scale_grid)),
    width_scale_grid = as.list(as.integer(width_scale_grid)),
    tuning_epochs = as.integer(tuning_epochs),
    tuning_patience = if (is.null(tuning_patience)) NULL else as.integer(tuning_patience),
    selection = selection,
    random_state = if (is.null(seed)) NULL else as.integer(seed),
    verbose = as.integer(verbose)
  )
  best <- .py_attr(res, "best_params")
  best$loss <- NULL
  list(best_params = best, log = .rows_to_df(.py_attr(res, "log")),
       selection_time = .py_attr(res, "selection_time"))
}
