# Columns of gem_growth_predictions.csv

One row per high-quality GEM genome with a prediction (9,139 genomes). Ranges and errors were measured with whole species, genera, families or orders held out, for the genome's model and novelty level; for the full model the ranges also depend on the domain.

| Column | Meaning |
|---|---|
| `genome_id` | GEM genome identifier (Nayfach et al. 2021) |
| `gtdb_taxonomy` | GTDB taxonomy from the GEM metadata (an older GTDB release than the training data, so the novelty level is a worst case) |
| `nearest_labeled_relative` | closest rank shared with a labeled training species: same genus, same family, new genus, same order, new family, or new order |
| `doubling_time_h` | predicted maximal doubling time at the predicted growth temperature, hours |
| `doubling_time_low_h` | lower end of the 80% range, hours |
| `doubling_time_high_h` | upper end of the 80% range, hours |
| `doubling_time_fold_error` | typical fold error at this novelty level (median of measured/predicted or its inverse) |
| `doubling_time_20C_h` | predicted doubling time standardized to 20 C, hours |
| `doubling_time_20C_low_h` | lower end of the 80% range, hours |
| `doubling_time_20C_high_h` | upper end of the 80% range, hours |
| `doubling_time_20C_fold_error` | typical fold error at this novelty level |
| `growth_temp_c` | predicted optimal growth temperature, C |
| `growth_temp_low_c` | lower end of the 80% range, C |
| `growth_temp_high_c` | upper end of the 80% range, C |
| `growth_temp_error_c` | mean absolute error at this novelty level, C |
| `slower_than_a_day` | predicted doubling time above 24 hours |
| `prob_slower_than_a_day` | calibrated probability that the doubling time is above 24 hours (useful for ranking; the model rarely predicts above a day for a new lineage) |
| `model_used` | inputs available for this genome, and so the model used |
| `domain` | bacteria or archaea |
| `phylogeny_rank_used` | GTDB rank at which the genome was placed for the phylogeny input; empty when the genome could not be placed |
| `completeness` | CheckM completeness, % (GEM metadata) |
| `contamination` | CheckM contamination, % (GEM metadata) |
| `ecosystem_category` | GEM metadata |
| `ecosystem_type` | GEM metadata |
| `habitat` | GEM metadata |
