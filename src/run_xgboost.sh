#!/bin/bash
/mnt/mnemo5/sum02dean/miniconda3/envs/string_score/bin/python xgboost_model.py \
--output_dir models/noise_model/ \
--model_name code_test \
--species_id '9606' \
--cogs False \
--use_noise True \
--class_weight 4 \
--neg_ratio 4 \
--drop_homology True \
--use_foi False

# ecoli: 511145
# human: 9606
# yeast: 4932

# TODO:
# - add model selection, Bayes log-reg
#https://towardsdatascience.com/introduction-to-bayesian-logistic-regression-7e39a0bae691
# - add uncertainty predictions
# - return most uncertain predictions


