#!/bin/bash

logdir=./logs
outputdir=./outputs
export EVALS_THREADS=3

timestamp=$(date +%Y%m%d_%H%M%S)
logpathbase=$logdir/$timestamp/

# NOTE: for the experiments in the report, they always use gpt-4 as the voter
voter_model=gpt-4
echo Running experiments and logging to $logpathbase
for influencer_model in gpt-3.5-turbo-16k gpt-4-base gpt-4
do
    for interaction_length in 3 5
    do
        # TODO: switch .testing.v0 to just .v0
        oaieval $voter_model,$influencer_model ballots.${interaction_length}.testing.v0 --record_path $logpathbase${influencer_model}_${voter_model}_${interaction_length}-interactions.log
    done
done
echo Done running experiments, all logs in $logpathbase

echo Producing plots, outputs to $outputdir
python make_plots.py --log_dir $logpathbase --out_dir $outputdir