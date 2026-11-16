# Installs the R packages for the released gRodon and Phydon into the project library .Rlib, at the versions the paper used.
# Needs R (the paper used R 4.6.1; for example: brew install r). Run from the repository root: make r-packages
# Every installed package and its version is listed in scripts/r/R_packages.lock.tsv; the last lines compare against it.
lib <- normalizePath(".Rlib")
.libPaths(c(lib, .libPaths()))
options(repos = c(CRAN = "https://cloud.r-project.org"), Ncpus = 8)
install.packages(c("BiocManager", "remotes"), lib = lib)
lock <- read.delim("scripts/r/R_packages.lock.tsv", stringsAsFactors = FALSE)
ver <- setNames(lock$version, lock$package)
# direct CRAN dependencies at their exact versions
for (p in c("ape", "castor", "picante", "matrixStats", "dplyr", "magrittr")) {
  remotes::install_version(p, version = ver[[p]], lib = lib, upgrade = "never", dependencies = TRUE)
}
# Bioconductor 3.23 (Biostrings 2.80.2, coRdon 1.30.0)
BiocManager::install(version = "3.23", lib = lib, update = FALSE, ask = FALSE)
BiocManager::install(c("Biostrings", "coRdon"), lib = lib, update = FALSE, ask = FALSE)
# the released tools at the exact commits the paper ran (gRodon 2.7.3, Phydon 1.1.0)
remotes::install_github("jlw-ecoevo/gRodon2", ref = "4b13053e7f49ca99bceeee36940f5e820c5b0c32", lib = lib, upgrade = "never")
remotes::install_github("xl0418/Phydon", ref = "321bb76d24a2488532ecce4fa7ec2abfd724eba9", lib = lib, upgrade = "never")
# report any package whose installed version differs from the lock file
for (i in seq_len(nrow(lock))) {
  p <- lock$package[i]
  if (p == "R") next
  got <- tryCatch(as.character(packageVersion(p, lib.loc = lib)), error = function(e) "missing")
  if (got != lock$version[i]) cat("differs from the paper:", p, "installed", got, "paper", lock$version[i], "\n")
}
for (p in c("gRodon", "Phydon", "Biostrings", "coRdon", "picante", "castor")) cat(p, as.character(packageVersion(p, lib.loc = lib)), "\n")
