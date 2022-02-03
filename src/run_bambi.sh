#!/bin/bash
/mnt/mnemo5/sum02dean/miniconda3/envs/string_score/bin/python bambi_model.py \
--output_dir models/03_02_2022/ \
--model_name bambi \
--species_id '511145' \
--cogs True \
--use_noise True \
--class_weight 4 \
--neg_ratio 4 \
--drop_homology True \
--use_foi False \
--n_samples 1 \
--n_chains 4 \
--n_draws  1000 \
--n_tune 3000 \
--family bernoulli


# ecoli: 511145
# human: 9606
# yeast: 4932