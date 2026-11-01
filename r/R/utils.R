.losses <- c("squared", "huber", "quantile", "poisson", "logistic")

# Parameters that must reach Python as integers. R numerics arrive as floats.
.int_params <- c("grid_size", "depth", "width", "width_scale", "spline_order",
                 "max_active_edges", "finetune_epochs", "finetune_patience",
                 "max_epochs", "patience", "batch_size", "lr_step_size",
                 "random_state", "verbose")

.estimator_class <- function(loss) {
  if (identical(loss, "logistic")) kasn_py$KASNClassifier else kasn_py$KASNRegressor
}

.coerce_params <- function(params) {
  for (nm in intersect(names(params), .int_params)) {
    if (!is.null(params[[nm]])) params[[nm]] <- as.integer(params[[nm]])
  }
  if (!is.null(params$grid_range)) {
    params$grid_range <- as.list(as.numeric(params$grid_range))
  }
  params
}

.check_param_names <- function(params, cls) {
  if (length(params) == 0) return(invisible(TRUE))
  if (is.null(names(params)) || any(!nzchar(names(params)))) {
    stop("all parameters in `...` must be named", call. = FALSE)
  }
  known <- names(reticulate::py_to_r(cls()$get_params()))
  bad <- setdiff(names(params), known)
  if (length(bad)) {
    stop("unknown KASN parameter(s): ", paste(bad, collapse = ", "),
         ". See kasn_defaults().", call. = FALSE)
  }
  invisible(TRUE)
}

.as_matrix <- function(x) {
  if (is.data.frame(x)) {
    ok <- vapply(x, is.numeric, logical(1))
    if (!all(ok)) {
      stop("non-numeric columns in `x`: ", paste(names(x)[!ok], collapse = ", "),
           ". Use the formula interface for factors.", call. = FALSE)
    }
  }
  x <- as.matrix(x)
  if (!is.numeric(x)) stop("`x` must be numeric", call. = FALSE)
  storage.mode(x) <- "double"
  x
}

# R vectors reach Python as lists (scalars at length one), so send numpy.
.as_numpy <- function(v, n, what = "y") {
  v <- as.numeric(v)
  if (length(v) != n) {
    stop(sprintf("`%s` has length %d but `x` has %d rows", what, length(v), n),
         call. = FALSE)
  }
  reticulate::np_array(v, dtype = "float64")
}

# Logistic responses are encoded 0/1 on the R side; the second level is the
# positive class. Returns the numpy response and the levels.
.encode_response <- function(y, n, loss, levels = NULL) {
  if (loss != "logistic") {
    return(list(y = .as_numpy(y, n), levels = NULL))
  }
  if (is.null(levels)) {
    levels <- if (is.factor(y)) levels(droplevels(y)) else sort(unique(as.character(y)))
    if (length(levels) != 2) {
      stop("loss = \"logistic\" needs a response with exactly two levels",
           call. = FALSE)
    }
  }
  ych <- as.character(y)
  if (anyNA(ych) || !all(ych %in% levels)) {
    stop("response contains values outside the training levels: ",
         paste(levels, collapse = ", "), call. = FALSE)
  }
  list(y = .as_numpy(as.numeric(ych == levels[2]), n), levels = levels)
}

.model_matrix <- function(tt, mf, contrasts = NULL) {
  x <- stats::model.matrix(tt, mf, contrasts.arg = contrasts)
  ctr <- attr(x, "contrasts")
  x <- x[, colnames(x) != "(Intercept)", drop = FALSE]
  attr(x, "contrasts") <- ctr
  x
}

.design_matrix <- function(object, newdata) {
  if (is.null(object$terms)) {
    x <- .as_matrix(newdata)
  } else {
    tt <- stats::delete.response(object$terms)
    mf <- stats::model.frame(tt, newdata, xlev = object$xlevels,
                             na.action = stats::na.pass)
    x <- .model_matrix(tt, mf, object$contrasts)
  }
  if (ncol(x) != object$n_features) {
    stop(sprintf("`newdata` has %d columns, the model was fitted on %d",
                 ncol(x), object$n_features), call. = FALSE)
  }
  x
}

.new_kasn <- function(est, call, loss, levels = NULL, feature_names = NULL) {
  structure(list(py = est, call = call, loss = loss, levels = levels,
                 feature_names = feature_names,
                 n_features = .py_attr(est, "n_features_in_"),
                 terms = NULL, xlevels = NULL, contrasts = NULL),
            class = "kasn")
}

.py_attr <- function(obj, name) {
  reticulate::py_to_r(reticulate::py_get_attr(obj, name))
}

.rows_to_df <- function(rows) {
  if (length(rows) == 0) return(data.frame())
  cols <- names(rows[[1]])
  out <- lapply(cols, function(nm) {
    vals <- lapply(rows, function(r) if (is.null(r[[nm]])) NA else r[[nm]])
    unlist(vals, use.names = FALSE)
  })
  names(out) <- cols
  as.data.frame(out, stringsAsFactors = FALSE)
}

.check_kasn <- function() {
  if (!kasn_available()) {
    stop("the Python module 'kasn' is not available. It is provisioned ",
         "through reticulate::py_require(), or install it into your own ",
         "environment with install_kasn().", call. = FALSE)
  }
  invisible(TRUE)
}

.check_alive <- function(object) {
  if (reticulate::py_is_null_xptr(object$py)) {
    stop("this fit points to a Python object from another R session. ",
         "Use kasn_save() and kasn_load().", call. = FALSE)
  }
  invisible(TRUE)
}
