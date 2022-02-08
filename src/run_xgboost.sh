#!/bin/bash
/mnt/mnemo5/sum02dean/miniconda3/envs/string_score/bin/python xgboost_model.py \
--output_dir models/02_02_2022/ \
--model_name standard_pp_ecoli \
--species_id '511145' \
--cogs True \
--use_noise True \
--class_weight 4 \
--neg_ratio 4 \
--drop_homology True \
--use_foi False \
--n_samples 4 \
--pre_process True \

# ecoli: 511145
# human: 9606
# yeast: 4932