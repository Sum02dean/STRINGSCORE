#!/bin/bash
/mnt/mnemo5/sum02dean/miniconda3/envs/string-score-2.0/bin/python bambi_model.py \
--model_name bambi \
--output_dir models/test_env/ \
--input_dir pre_processed_data/scaled/ \
--species_id '511145 9606 4932' \
--cogs True \
--use_noise True \
--drop_homology True \
--n_runs 1 \
--n_chains 3 \
--n_draws 1000 \
--n_tune 3000 \
--family bernoulli

# ecoli: 511145
# human: 9606
# yeast: 4932 slac