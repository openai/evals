#!/bin/bash

logdir=./logs
outputdir=./outputs

timestamp=$(date +%Y%m%d_%H%M%S)
logpathbase=$logdir/$timestamp/
mkdir -p ${logpathbase}

echo "Running core experiments (balanced prompt, five-, ten-, fifteen-turn conversations) and logging to $logpathbase"
for turn_cap in 5 10 15
do
    for con_artist_model in gpt-3.5-turbo-16k gpt-4
    do
        oaieval make-me-pay/${con_artist_model} make-me-pay \
            --extra_eval_params turn_cap=${turn_cap},duration_cap_minutes=0 \
            --record_path $logpathbase${turn_cap}_${con_artist_model}.log
    done
done

echo Done running experiments, all logs in $logpathbase

echo Producing plots, outputs to $outputdir
python make_plots.py --log_dir $logpathbase --out_dir $outputdir
