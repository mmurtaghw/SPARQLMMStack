#!/bin/bash

# Train on Qald9
base_data="qald9"
for setting in f1_col_cuttoff f1_hybrid_col_cuttoff is_execution_valid
do
    echo "running experiment $setting from data $base_data"
    datapath="data/processed/$base_data/$setting/"
    savepath="output/$base_data/$setting/"
    python train_roberta.py $datapath $savepath
done

# Train on vQuanda
base_data="vquanda"
for setting in f1_col_cuttoff f1_hybrid_col_cuttoff is_execution_valid
do
    echo "running experiment $setting from data $base_data"
    datapath="data/processed/$base_data/$setting/"
    savepath="output/$base_data/$setting/"
    python train_roberta.py $datapath $savepath
done