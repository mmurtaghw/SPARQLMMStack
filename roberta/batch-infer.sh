#!/bin/bash

# run inference for all combos
# note that when "base_data" is qald9, a ROBERTA model TRAINEDD ON QALD9 will be loaded
# this will RUN INFERENCE ON VQUANDA (to avoid data leakage)
# and vice versa

base_data="qald9"
inf_on="vquanda"
for setting in f1_col_cuttoff f1_hybrid_col_cuttoff is_execution_valid
do
    out_file="output/cache/inf_on_${inf_on}-base_data_${base_data}-${setting}.tsv"
    python roberta_inference.py $base_data $inf_on $setting $out_file
done

base_data="vquanda"
inf_on="qald9"
for setting in f1_col_cuttoff f1_hybrid_col_cuttoff is_execution_valid
do
    out_file="output/cache/inf_on_${inf_on}-base_data_${base_data}-${setting}.tsv"
    python roberta_inference.py $base_data $inf_on $setting $out_file
done
