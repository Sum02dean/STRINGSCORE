#!/bin/bash
/mnt/mnemo5/sum02dean/miniconda3/envs/string_score/bin/python xgboost_model.py \
--model_name mixed_model \
--species_id '511145 9606 4932' \
--output_dir cog_test_dir/ \
--cogs False \
--use_noise False \
--class_weight 1 \
--neg_ratio 1 \
--drop_homology True \

# ecoli: 511145
# human: 9606
# yeast: 4932

# TODO:
# - add model selection, Bayes log-reg
#https://towardsdatascience.com/introduction-to-bayesian-logistic-regression-7e39a0bae691
# - add uncertainty predictions
# - return most uncertain predictions


