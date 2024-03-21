#!/bin/bash
echo "Generating word samples..."
python evals/elsuite/already_said_that/scripts/gen_data.py --n_samples 500 --jsonl_dir evals/registry/data/already_said_that --seed 0
echo "Done."
