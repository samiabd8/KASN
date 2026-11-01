#' Is the Python backend available?
#'
#' @return `TRUE` if the Python module `kasn` can be imported.
#' @export
kasn_available <- function() {
  reticulate::py_module_available("kasn")
}

#' Install the Python backend into a user-managed environment
#'
#' Not needed in the default setup, where [reticulate::py_require()]
#' provisions an isolated environment on first use. Use this when you manage
#' your own virtualenv or conda environment (for example to pin a CUDA build
#' of PyTorch), then point reticulate at it with
#' `Sys.setenv(RETICULATE_PYTHON = ...)` or [reticulate::use_virtualenv()]
#' before calling any `kasn` function.
#'
#' @param envname Name or path of the environment.
#' @param method,... Passed to [reticulate::py_install()].
#' @export
install_kasn <- function(envname = "r-kasn", method = "auto", ...) {
  reticulate::py_install(.kasn_requirement(), envname = envname,
                         method = method, pip = TRUE, ...)
}
