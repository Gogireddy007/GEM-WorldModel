# Runs the released gRodon (predictGrowth, full mode) on every labeled genome, once with the
# measured growth temperature and once without it, and keeps its codon statistics.
# gRodon does not use relatives, so no folds are needed; its regression is the published fit.
# Input: data/processed/released_tools_folds.csv. Output: data/processed/released_grodon.csv.
.libPaths(c(normalizePath(".Rlib"), .libPaths()))
suppressPackageStartupMessages({library(gRodon); library(Biostrings)})
f <- read.csv("data/processed/released_tools_folds.csv", stringsAsFactors = FALSE)
rows <- list()
for (i in seq_len(nrow(f))) {
  genes <- readDNAStringSet(f$gene_location[i])
  he <- grepl("ribosomal protein", names(genes), ignore.case = TRUE)
  one <- function(temp) tryCatch(as.data.frame(predictGrowth(genes, he, mode = "full", temperature = temp)),
                                 error = function(e) NULL)
  with_t <- one(f$growth_temp_c[i]); no_t <- one("none")
  if (is.null(with_t) || is.null(no_t)) next
  rows[[length(rows) + 1]] <- data.frame(genome_name = f$genome_name[i], d_with_temperature = with_t$d,
    d_without_temperature = no_t$d, CUBHE = no_t$CUBHE, ConsistencyHE = no_t$ConsistencyHE, CPB = no_t$CPB,
    GC = no_t$GC, nHE = no_t$nHE)
  if (i %% 100 == 0) message(i, " of ", nrow(f))
}
out <- do.call(rbind, rows)
write.csv(out, "data/processed/released_grodon.csv", row.names = FALSE)
message("wrote ", nrow(out), " genomes")
