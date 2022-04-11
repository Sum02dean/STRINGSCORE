#!/bin/bash
/mnt/mnemo5/sum02dean/miniconda3/envs/string-score-2.0/bin/python xgboost_model.py \
--model_name xgboost \
--output_dir models/test_env/ \
--input_dir pre_processed_data/ \
--species_id '511145' \
--cogs True \
--use_noise True \
--class_weight 4 \
--neg_ratio 4 \
--drop_homology True \
--use_foi False \
--n_samples 4 \
--pre_process False \

# ecoli: 511145
# human: 9606
# yeast: 4932