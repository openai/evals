#!/bin/bash
logdir=./logs
outputdir=./outputs
# export EVALS_THREADS=3

timestamp=$(date +%Y%m%d_%H%M%S)
logpathbase=$logdir/$timestamp/

echo Running experiments and logging to $logpathbase

oaieval gpt-3.5-turbo text_compression.gzip --record_path ${logpathbase}/gpt-3.5-turbo_gzip.log # CompletionFn doesn't matter for the gzip baseline

oaieval gpt-3.5-turbo text_compression.copytext --record_path ${logpathbase}/gpt-3.5-turbo_copytext.log
oaieval gpt-3.5-turbo text_compression.abbreviate --record_path ${logpathbase}/gpt-3.5-turbo_abbreviate.log
oaieval gpt-3.5-turbo text_compression.simple --record_path ${logpathbase}/gpt-3.5-turbo_simple.log
oaieval gpt-3.5-turbo text_compression.instructions --record_path ${logpathbase}/gpt-3.5-turbo_instructions.log
oaieval gpt-3.5-turbo text_compression.scratch --record_path ${logpathbase}/gpt-3.5-turbo_scratch.log

oaieval gpt-4 text_compression.copytext --record_path ${logpathbase}/gpt-4_copytext.log
oaieval gpt-4 text_compression.abbreviate --record_path ${logpathbase}/gpt-4_abbreviate.log
oaieval gpt-4 text_compression.simple --record_path ${logpathbase}/gpt-4_simple.log
oaieval gpt-4 text_compression.instructions --record_path ${logpathbase}/gpt-4_instructions.log
oaieval gpt-4 text_compression.scratch --record_path ${logpathbase}/gpt-4_scratch.log

echo Done running experiments, all logs in $logpathbase

echo Producing plots, outputs to $outputdir
python make_plots.py --log_dir $logpathbase --out_dir $outputdir