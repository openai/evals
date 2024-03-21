#!/bin/bash
logdir=./logs
outdir=./outputs

timestamp=$(date +%Y%m%d_%H%M%S)
logpathbase=$logdir/$timestamp
outpathbase=$outdir/$timestamp
SPLIT=main

mkdir -p ${logpathbase}

export EVALS_THREADS=250
echo Running full experiments and logging to $logpathbase

declare -a SOLVERS=(
    error_recovery/gpt-3.5-turbo-0613
    error_recovery/gpt-4-0613
    generation/hhh/gpt-4-base
)

# OWN REASONING VARIANT
for solver in "${SOLVERS[@]}"
do
    log_name=${SPLIT}_${solver//\//-}_own-reasoning

    oaieval $solver error-recovery.$SPLIT \
    --extra_eval_params final_answer_prompt_role=system \
    --record_path "$logpathbase/$log_name.log"
done

# OTHER REASONING VARIANT
for solver in "${SOLVERS[@]}"
do
    log_name=${SPLIT}_${solver//\//-}_other-reasoning

    oaieval $solver error-recovery.$SPLIT.other-reasoning \
    --extra_eval_params final_answer_prompt_role=system \
    --record_path "$logpathbase/$log_name.log"
done

echo Producing plots, outputs to $outpathbase

mkdir -p ${outpathbase}
python make_plots.py --log_dir ${logpathbase} --out_dir $outpathbase
