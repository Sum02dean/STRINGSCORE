#!/bin/bash
/mnt/mnemo5/sum02dean/miniconda3/envs/string_score/bin/python net_model.py \
--output_dir models/ \
--model_name neural_net_unbalanced \
--species_id '511145 9606 4932' \
--cogs True \
--use_noise True \
--drop_homology True \
--n_runs 1 \


# ecoli: 511145
# human: 9606
# yeast: 4932