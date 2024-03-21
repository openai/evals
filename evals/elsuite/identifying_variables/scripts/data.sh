#!/bin/bash

# generate datasets of size 500 and 5000
echo "Generating default dataset: 500 samples"
python gen_data.py --n_samples 500 --jsonl_dir ../../../registry/data/identifying_variables/
echo "Generating large dataset: 5000 samples"
python gen_data.py --n_samples 5000 --jsonl_dir ../../../registry/data/identifying_variables/
echo "Generating default dataset: 500 samples (balanced ctrl vars)"
python gen_data.py --balanced_ctrl_vars --n_samples 500 --jsonl_dir ../../../registry/data/identifying_variables/
echo "Generating large dataset: 5000 samples (balanced ctrl vars)"
python gen_data.py --balanced_ctrl_vars --n_samples 5000 --jsonl_dir ../../../registry/data/identifying_variables/

echo "Done."
