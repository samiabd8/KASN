#' Save and restore a fitted KASN
#'
#' A `kasn` fit holds a pointer to a live Python object, so [saveRDS()] alone
#' cannot persist it. `kasn_save()` writes a single `.rds` file holding the R
#' metadata (call, formula terms, factor levels) and the bytes of the Python
#' model file, and `kasn_load()` rebuilds the fit in a new session.
#'
#' @param object A `kasn` fit.
#' @param file Path of the `.rds` file.
#' @param device Device for the restored network ("cpu", "cuda", ...).
#'   Defaults to the one the model was configured with.
#' @return `kasn_save()` returns `file` invisibly, `kasn_load()` a `kasn` fit.
#' @export
kasn_save <- function(object, file) {
  .check_alive(object)
  tmp <- tempfile(fileext = ".pt")
  on.exit(unlink(tmp), add = TRUE)
  object$py$save(tmp)
  payload <- list(
    format = "kasn-r-1",
    meta = object[setdiff(names(object), "py")],
    py_bytes = readBin(tmp, "raw", n = file.info(tmp)$size)
  )
  saveRDS(payload, file)
  invisible(file)
}

#' @rdname kasn_save
#' @export
kasn_load <- function(file, device = NULL) {
  .check_kasn()
  payload <- readRDS(file)
  if (!identical(payload$format, "kasn-r-1")) {
    stop("`file` was not written by kasn_save()", call. = FALSE)
  }
  tmp <- tempfile(fileext = ".pt")
  on.exit(unlink(tmp), add = TRUE)
  writeBin(payload$py_bytes, tmp)
  est <- kasn_py$load(tmp, device = device)
  structure(c(list(py = est), payload$meta), class = "kasn")
}
