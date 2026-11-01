#' Simulate the KASN Monte Carlo designs
#'
#' \eqn{Y_t = f_0(X_t) + \varepsilon_t}, where the covariates follow a
#' stationary Gaussian AR(1) across time with uniform marginals on `support`
#' (Gaussian copula) and only the first `s` columns enter \eqn{f_0}.
#' `"additive"` is \eqn{\sum_j \sin X_j} and `"compositional"` is
#' \eqn{\sigma(\sum_j \sin X_j) + 0.1 \sum_j |X_j|}, rescaled to the additive
#' target's standard deviation when `standardise = TRUE`.
#'
#' @param n,d,s Sample size, covariate dimension and relevant covariates.
#' @param target "additive", "compositional" or "deep".
#' @param rho AR(1) coefficient of the covariates (0 gives i.i.d. rows).
#' @param marginal "uniform" or "normal".
#' @param support Support of the uniform marginal.
#' @param noise_sd,noise_rho Error standard deviation and AR(1) coefficient.
#' @param standardise Rescale non-additive targets.
#' @param seed Random seed.
#' @return A list with `x` (n x d matrix), `y`, `f0` and `eps`.
#' @examplesIf identical(Sys.getenv("NOT_CRAN"), "true") && kasn_available()
#' sim <- kasn_simulate(n = 200, d = 5, s = 2)
#' str(sim)
#' @export
kasn_simulate <- function(n = 2000, d = 500, s = 50, target = "compositional",
                          rho = 0.7, marginal = "uniform", support = c(-3, 3),
                          noise_sd = 0.5, noise_rho = 0, standardise = TRUE,
                          seed = NULL) {
  .check_kasn()
  b <- kasn_py$make_kasn_regression(
    n = as.integer(n), d = as.integer(d), s = as.integer(s), target = target,
    rho = rho, marginal = marginal, support = as.list(as.numeric(support)),
    noise_sd = noise_sd, noise_rho = noise_rho, standardise = standardise,
    random_state = if (is.null(seed)) NULL else as.integer(seed))
  get <- function(nm) reticulate::py_to_r(b$get(nm))
  list(x = as.matrix(get("X")), y = as.numeric(get("y")),
       f0 = as.numeric(get("f0")), eps = as.numeric(get("eps")))
}
