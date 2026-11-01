#' kasn: Kolmogorov-Arnold Sieve Networks
#'
#' R interface to the `kasn` Python package. The Python backend is declared
#' with [reticulate::py_require()] when the package loads, so reticulate
#' provisions it on first use unless you point reticulate at your own Python
#' environment (see [install_kasn()]). Set the environment variable
#' `KASN_PY_REQUIREMENT` to override the requirement string, for example to
#' a local checkout during development.
#'
#' Main functions: [kasn()], [predict.kasn()], [kasn_tune()],
#' [kasn_simulate()], [kasn_save()] and [kasn_load()].
#'
#' @keywords internal
#' @importFrom utils str
"_PACKAGE"
