#!/bin/bash
logdir=./logs
outputdir=./outputs

timestamp=$(date +%Y%m%d_%H%M%S)
logpathbase=$logdir/$timestamp/

mkdir -p ${logpathbase}

echo Running experiments and logging to $logpathbase

oaieval generation/direct/gpt-3.5-turbo bugged_tools.all_log --record_path ${logpathbase}gpt-3.5-turbo.log
oaieval generation/direct/gpt-4 bugged_tools.all_log --record_path ${logpathbase}gpt-4.log

echo Done running experiments, all logs in $logpathbase

echo Producing plots, outputs to $outputdir
python plot_experiments.py --log_dir $logpathbase --out_dir $outputdir
