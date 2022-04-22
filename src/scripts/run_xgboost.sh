#!/bin/bash
/mnt/mnemo5/sum02dean/miniconda3/envs/string-score-env/bin/python ../xgboost_model.py \
--model_name xgboost \
--input_dir ../pre_processed_data/ \
--output_dir ../models/test_env/ \
--species_id '511145 9606 4932' \
--cogs True \
--use_noise True \
--class_weight 4 \
--neg_ratio 4 \
--drop_homology True \
--n_sampling_runs 1 \


# ecoli: 511145
# human: 9606
# yeast: 4932
# --class_weight and --neg_ratio should have the same value. To use a balanced dataset, set the value to 1
