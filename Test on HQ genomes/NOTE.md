# Which file to use

Use `hq_genome_predictions_v2.csv`. The earlier files in this folder (`unlabeled_predictions_hq.csv`,
`hq_genome_predictions_full.csv`, the four charts, and the PDF report) came from a first version of the
prediction script that gave the model phylogeny and 16S inputs in a different coordinate system from the one it
was trained on, so those predictions are not reliable. Their median doubling time was 0.6 hours against 7 hours
for the training species, and none were slower than 24 hours.

The v2 table fixes that: the model now gets the same kind of inputs for every genome. Each row also says how close
its nearest labeled relative is and what typical error to expect at that level. Read the predictions as rough
estimates, typically within a factor of about 2, and better for ranking genomes than for exact values. They are
less reliable for genomes with no close labeled relative, which is most of them.

The charts and the PDF have not been regenerated yet.
