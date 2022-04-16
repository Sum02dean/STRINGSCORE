#!/bin/bash
/mnt/mnemo5/sum02dean/miniconda3/envs/string-score-env/bin/python ../bambi_model.py \
--model_name bambi \
--input_dir ../pre_processed_data/ \
--output_dir ../models/test_env/ \
--species_id '511145 9606 4932' \
--cogs True \
--use_noise True \
--drop_homology True \
--n_sampling_runs 1 \
--n_chains 3 \
--n_draws 10 \
--n_tune 10 \
--family bernoulli \
--ensemble_report False

# ecoli: 511145
# human: 9606
# yeast: 4932 slac