# Fixed inputs

These files are part of the method, not something a rerun should recompute, because the step that first made them
depended on a moving source (an NCBI search) or on a working state that no longer exists. `make fixed-inputs` copies
them into `data/processed/` before anything else runs.

| File | What it is | Why it is fixed |
|---|---|---|
| `hq_genome_ids_all.csv` | the 9,143 GEM high-quality genomes the paper predicts | the list the released predictions are made for |
| `mag_validation_ids.csv` | the 720 GEM genomes of labeled species used for the MAG check | defines the MAG validation set |
| `gem_used_genome_ids.csv` | union of the two lists above (9,608 genomes) | the GEM genomes whose GC content is measured |
| `gem_16s_genome_ids.csv` | the 9,419 of those genomes searched for a 16S gene | the 16S search was run on these genomes only; searching more would change which MAGs have a 16S input |
| `labeled_16s_sequences.csv` | one 16S sequence per labeled species (NCBI nucleotide search, top hit) | NCBI's top hit for a name changes as records are added |
| `labeled_16s_landmark_names.csv` | the 16 labeled species whose 16S genes serve as reference points | chosen once from an earlier, smaller 16S set; written to `labeled_16s_landmark_names.npy` |
