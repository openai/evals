#!/bin/bash
logdir=./logs
outputdir=./outputs

timestamp=$(date +%Y%m%d_%H%M%S)
logpathbase=$logdir/$timestamp/

echo Running experiments and logging to $logpathbase

# Baselines
oaieval sandbagging/default/gpt-4 no-sandbagging-all.v1 \
    --record_path ${logpathbase}no_sb_all_gpt4.log
oaieval sandbagging/default/gpt-3.5-turbo no-sandbagging-all.v1 \
    --record_path ${logpathbase}no_sb_all_gpt35.log

oaieval sandbagging/default/gpt-4 no-sandbagging-subset.v1 \
    --record_path ${logpathbase}no_sb_subset_gpt4.log
oaieval sandbagging/default/gpt-3.5-turbo no-sandbagging-subset.v1 \
    --record_path ${logpathbase}no_sb_subset_gpt35.log

# Sandbagging on subset examples
oaieval sandbagging/sb_subset/gpt-4 sandbagging-subset.v1 \
    --record_path ${logpathbase}sb_subset_gpt4.log
oaieval sandbagging/sb_subset/gpt-3.5-turbo sandbagging-subset.v1 \
    --record_path ${logpathbase}sb_subset_gpt35.log

# Performance on rest on normal MMLU Qs
oaieval sandbagging/sb_subset/gpt-4 sandbagging-non-subset.v1 \
    --record_path ${logpathbase}sb_subset_normal_gpt4.log
oaieval sandbagging/sb_subset/gpt-3.5-turbo sandbagging-non-subset.v1 \
    --record_path ${logpathbase}sb_subset_normal_gpt35.log

echo Done running experiments, all logs in $logpathbase

echo Producing plots, outputs to $outputdir
python sandbagging_subset_plots.py --log_dir $logpathbase --out_dir $outputdir
