#!/bin/bash
/mnt/mnemo5/sum02dean/miniconda3/envs/string-score-3/bin/python ../pre_process.py \
--input_dir ../pre_processed_data/ \
--output_dir '../pre_processed_data/' \
--species_id '511145 4932 9606' \
--neg_ratio 4 \
--drop_homology True \
--test_ratio 0.10 \
--scale False

# ecoli: 511145
# human: 9606
# yeast: 4932