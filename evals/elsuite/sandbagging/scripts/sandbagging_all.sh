#!/bin/bash
logdir=./logs
outputdir=./outputs

timestamp=$(date +%Y%m%d_%H%M%S)
logpathbase=$logdir/$timestamp/

mkdir -p ${logpathbase}

echo Running experiments and logging to $logpathbase

# Baselines
oaieval sandbagging/default/gpt-4 no-sandbagging-all.v1 \
    --record_path ${logpathbase}no_sb_all_gpt4.log >> ${logpathbase}out.txt
oaieval sandbagging/default/gpt-3.5-turbo no-sandbagging-all.v1 \
    --record_path ${logpathbase}no_sb_all_gpt35.log >> ${logpathbase}out.txt

# Sandbagging on all MMLU examples
oaieval sandbagging/sb/gpt-4 sandbagging-all.v1 \
    --record_path ${logpathbase}sb_all_gpt4.log >> ${logpathbase}out.txt
oaieval sandbagging/sb/gpt-3.5-turbo sandbagging-all.v1 \
    --record_path ${logpathbase}sb_all_gpt35.log >> ${logpathbase}out.txt

echo Done running experiments, all logs in $logpathbase

echo Producing plots, outputs to $outputdir
python sandbagging_all_plots.py --log_dir $logpathbase --out_dir $outputdir
