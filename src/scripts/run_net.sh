#!/bin/bash
/mnt/mnemo5/sum02dean/miniconda3/envs/string-score-env/bin/python ../net_model.py \
--model_name neural_net \
--input_dir ../pre_processed_data/ \
--output_dir ../models/test_env \
--species_id '511145 9606 4932' \
--cogs True \
--use_noise True \
--drop_homology True \
--n_sampling_runs 1 \
--batch_size 50 \
--epochs 50 \
--hidden_size 100 \
--learning_rate 0.002 \


# ecoli: 511145
# human: 9606
# yeast: 4932