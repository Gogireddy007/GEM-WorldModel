# Runs the released Phydon (and through it the released gRodon) on the labeled genomes, one test
# fold at a time. Before each call, the species of the fold's genomes are removed from Phydon's own
# training table (its Madin growth rates on GTDB r220), so both its phylogenetic estimate and the
# distance to the nearest training relative that sets its weighting are computed without them, as
# if Phydon had been trained without the test species. Phydon's code itself is not changed.
# Input: data/processed/released_tools_folds.csv (compare_released_tools.py --prepare).
# Output: data/processed/released_tools_predictions.csv (appends; finished folds are skipped).
# Run from the repository root; packages are in the project library .Rlib.
.libPaths(c(normalizePath(".Rlib"), .libPaths()))
suppressPackageStartupMessages({library(Phydon)})
args <- commandArgs(trailingOnly = TRUE)
seeds <- if (length(args)) as.integer(strsplit(args[1], ",")[[1]]) else 0:2
folds <- read.csv("data/processed/released_tools_folds.csv", stringsAsFactors = FALSE)
out_file <- "data/processed/released_tools_predictions.csv"
ns <- asNamespace("Phydon")
train_name <- c(bacteria = "GTDB_tax_trait_repGenome_in_tree_expanded", archaea = "GTDB_tax_trait_repGenome_in_tree_expanded_archaea")
cluster_name <- c(bacteria = "sp_clusters_bac", archaea = "sp_clusters_arc")
original <- lapply(train_name, function(n) get(n, envir = ns))

# genome accession -> species representative, from Phydon's own GTDB clusters
rep_of <- list()
for (dom in names(cluster_name)) {
  cl <- get(cluster_name[[dom]], envir = ns)
  members <- strsplit(cl$Clustered.genomes, ",")
  rep_of[[dom]] <- setNames(rep(cl$Representative.genome, lengths(members)), trimws(unlist(members)))
}

set_training <- function(dom, table) {
  unlockBinding(train_name[[dom]], ns)
  assign(train_name[[dom]], table, envir = ns)
  lockBinding(train_name[[dom]], ns)
}

done <- if (file.exists(out_file)) read.csv(out_file, stringsAsFactors = FALSE) else NULL
for (s in seeds) {
  col <- paste0("fold_seed", s)
  for (f in sort(unique(folds[[col]]))) {
    for (dom in c("bacteria", "archaea")) {
      if (!is.null(done) && any(done$seed == s & done$fold == f & done$domain == dom)) next
      sub <- folds[folds[[col]] == f & folds$domain == dom, ]
      if (nrow(sub) == 0) next
      held_out <- unique(na.omit(rep_of[[dom]][sub$genome_name]))
      table <- original[[dom]]
      set_training(dom, table[!(table$Representative.genome %in% held_out), ])
      info <- data.frame(gene_location = sub$gene_location, genome_name = sub$genome_name,
                         temperature = sub$growth_temp_c, stringsAsFactors = FALSE)
      ok <- tryCatch({
        invisible(capture.output(r <- suppressWarnings(Phydon(info, tax = dom, regression_mode = "geometric_mean", gRodon_mode = "full"))))
        TRUE
      }, error = function(e) { message("seed ", s, " fold ", f, " ", dom, ": ", conditionMessage(e)); FALSE })
      set_training(dom, table)
      if (!ok) next
      r <- r[, c("genome", "rep_genome", "phy_distance", "gRodonpred", "phylopred", "combopred", "nHE")]
      r$seed <- s; r$fold <- f; r$domain <- dom; r$training_species_removed <- length(held_out)
      write.table(r, out_file, sep = ",", row.names = FALSE, append = file.exists(out_file),
                  col.names = !file.exists(out_file))
      message(format(Sys.time()), " seed ", s, " fold ", f, " ", dom, ": ", nrow(r), " genomes, ",
              length(held_out), " species removed from Phydon's training table")
    }
  }
}
