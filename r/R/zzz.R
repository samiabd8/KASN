# Delay-loaded handle on the Python module. Populated in .onLoad().
kasn_py <- NULL

# The Python requirement declared to reticulate. Override with the
# KASN_PY_REQUIREMENT environment variable, for example
# "kasn @ file:///path/to/kasn/python" during development or "kasn==0.1.0"
# once the package is on PyPI.
.kasn_requirement <- function() {
  req <- Sys.getenv("KASN_PY_REQUIREMENT", unset = "")
  if (nzchar(req)) {
    return(req)
  }
  "kasn @ git+https://github.com/samiabd8/kasn#subdirectory=python"
}

.onLoad <- function(libname, pkgname) {
  reticulate::py_require(.kasn_requirement(), python_version = ">=3.9")
  kasn_py <<- reticulate::import("kasn", delay_load = TRUE, convert = FALSE)
}
