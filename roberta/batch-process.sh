#!/bin/bash

# Preprocess qald9
data="qald9"
for setting in f1_col_cuttoff f1_hybrid_col_cuttoff is_execution_valid
do
    echo "preprocessing $data for $setting"
    savepath="data/processed/$data/$setting/"
    python process_data.py $setting $data $savepath
done

# Preprocess vquanda
data="vquanda"
for setting in f1_col_cuttoff f1_hybrid_col_cuttoff is_execution_valid
do
    echo "preprocessing $data for $setting"
    savepath="data/processed/$data/$setting/"
    python process_data.py $setting $data $savepath
done