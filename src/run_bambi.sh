#!/bin/bash
/mnt/mnemo5/sum02dean/miniconda3/envs/string_score/bin/python bambi_model.py \
--output_dir models/drop_zero_col/28_02_2022/ \
--model_name bambi \
--species_id '511145 9606 4932' \
--cogs True \
--use_noise True \
--drop_homology True \
--n_runs 3 \
--n_chains 2 \
--n_draws  1000 \
--n_tune 3000 \
--family bernoulli

# ecoli: 511145
# human: 9606
# yeast: 4932 slac