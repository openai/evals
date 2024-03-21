#!/bin/bash
logdir=./logs
outputdir=./outputs

timestamp=$(date +%Y%m%d_%H%M%S)
logpathbase=$logdir/$timestamp/

mkdir -p ${logpathbase}

echo Running experiments and logging to $logpathbase
read -p "Enter the number of runs: " num_runs

set -x # Enable printing of each command before it's executed
# Random baselines
oaieval incontext_rl/random incontext_rl.v0 --record_path ${logpathbase}explanations/random.log
oaieval incontext_rl/random incontext_rl.raw.v0 --record_path ${logpathbase}raw/random.log

for (( run=1; run<=num_runs; run++ ))
do
    echo "Run #$run"
    # Use explanations variant
    # Direct
    oaieval generation/direct/gpt-4-turbo-preview incontext_rl.v0 --record_path ${logpathbase}explanations/gpt-4-turbo-preview_${run}.log
    oaieval generation/direct/gpt-3.5-turbo incontext_rl.v0 --record_path ${logpathbase}explanations/gpt-3.5-turbo_${run}.log

    # Raw variant
    # Direct
    oaieval generation/direct/gpt-4-turbo-preview incontext_rl.raw.v0 --record_path ${logpathbase}raw/gpt-4-turbo-preview_${run}.log
    oaieval generation/direct/gpt-3.5-turbo incontext_rl.raw.v0 --record_path ${logpathbase}raw/gpt-3.5-turbo_${run}.log

done

echo Done running experiments, all logs in $logpathbase

echo Producing plots for use_explanations variant, outputs to $outputdir
python plot_experiments.py --log_dir $logpathbase/explanations --out_dir $outputdir/explanations
echo Producing plots for raw variant, outputs to $outputdir
python plot_experiments.py --log_dir $logpathbase/raw --out_dir $outputdir/raw
set +x # Disable printing of each command after they've been executed
