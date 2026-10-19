# Installs the R packages for the released gRodon and Phydon into the project library .Rlib.
# Needs R (for example: brew install r). Run from the repository root: make r-packages
lib <- normalizePath(".Rlib")
.libPaths(c(lib, .libPaths()))
options(repos = c(CRAN = "https://cloud.r-project.org"), Ncpus = 8)
install.packages(c("BiocManager", "remotes", "ape", "castor", "picante", "matrixStats", "dplyr", "magrittr"), lib = lib)
BiocManager::install(c("Biostrings", "coRdon"), lib = lib, update = FALSE, ask = FALSE)
remotes::install_github("jlw-ecoevo/gRodon2", lib = lib, upgrade = "never")
remotes::install_github("xl0418/Phydon", lib = lib, upgrade = "never")
for (p in c("gRodon", "Phydon", "Biostrings", "coRdon", "picante", "castor")) cat(p, as.character(packageVersion(p, lib.loc = lib)), "\n")
