#!/bin/bash

for setting in bleu_cuttoff #bleu_hybrid_cuttoff f1_col_cuttoff_1stdev f1_hybrid_col_cuttoff_1stdev is_execution_valid
do
    echo "running experiment $setting"
    datapath="data/processed/$setting/"
    savepath="output/$setting/"
    python train_roberta.py $datapath $savepath
done
