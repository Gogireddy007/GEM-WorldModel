# Released gRodon (predictGrowth, full mode, no temperature) on a list of gene files, in parallel.
# Usage: Rscript scripts/r/grodon_stats.R <input.csv with genome_name,gene_location> <output.csv> [cores]
# Gene files are nucleotide FASTA with the ribosomal proteins named "... ribosomal protein" (the
# highly expressed set). Writes d (gRodon's doubling time without temperature) and its statistics.
.libPaths(c(normalizePath(".Rlib"), .libPaths()))
suppressPackageStartupMessages({library(gRodon); library(Biostrings); library(parallel)})
args <- commandArgs(trailingOnly = TRUE)
inp <- read.csv(args[1], stringsAsFactors = FALSE)
inp <- inp[!is.na(inp$gene_location) & nzchar(inp$gene_location), ]
cores <- if (length(args) >= 3) as.integer(args[3]) else 8
one <- function(i) {
  genes <- readDNAStringSet(inp$gene_location[i])
  he <- grepl("ribosomal protein", names(genes), ignore.case = TRUE)
  r <- tryCatch(suppressWarnings(as.data.frame(predictGrowth(genes, he, mode = "full", temperature = "none"))),
                error = function(e) NULL)
  if (is.null(r)) return(data.frame(genome_name = inp$genome_name[i], grodon_d = NA, CUBHE = NA, ConsistencyHE = NA,
                                    CPB = NA, grodon_GC = NA, nHE = sum(he)))
  data.frame(genome_name = inp$genome_name[i], grodon_d = r$d, CUBHE = r$CUBHE, ConsistencyHE = r$ConsistencyHE,
             CPB = r$CPB, grodon_GC = r$GC, nHE = r$nHE)
}
out <- do.call(rbind, mclapply(seq_len(nrow(inp)), one, mc.cores = cores))
write.csv(out, args[2], row.names = FALSE)
message("wrote ", nrow(out), " genomes, ", sum(!is.na(out$CUBHE)), " with statistics")
