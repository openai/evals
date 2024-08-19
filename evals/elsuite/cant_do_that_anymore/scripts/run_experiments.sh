#!/bin/bash
logdir=./logs
outputdir=./outputs

timestamp=$(date +%Y%m%d_%H%M%S)
logpathbase=$logdir/$timestamp/

mkdir -p ${logpathbase}

declare -a SOLVERS_ZEROSHOT=(
    "generation/direct/gpt-3.5-turbo"
    "chess/generation/direct/gpt-3.5-turbo-instruct"
    "generation/direct/gpt-4-turbo-preview"
    "chess/generation/direct/gpt-4-base"
)

# See if variant was indicated
run_diagonal_variant=1
for arg in "$@"
do
    if [[ $arg == "--no_diagonal_variant" ]]; then
        run_diagonal_variant=0
        break
    fi
done

# TODO CoT solvers

echo Running experiments and logging to $logpathbase

for run_idx in {0..2}
do
    for solver in "${SOLVERS_ZEROSHOT[@]}"
    do
        log_name=${solver//\//-}
        oaieval $solver cant_do_that_anymore \
            --record_path ${logpathbase}run_${run_idx}_${log_name}.log \
            --extra_eval_params n_samples=1000 \
            --seed ${run_idx}
    done
done

echo Done running experiments, all logs in $logpathbase

echo Producing plots, outputs to $outputdir
python make_plots.py --log_dir $logpathbase --out_dir $outputdir

if [[ $run_diagonal_variant -eq 1 ]]; then
    echo Running diagonal experiment and logging to $logpathbase

    for run_idx in {0..2}
    do
        for solver in "${SOLVERS_ZEROSHOT[@]}"
        do
            log_name=${solver//\//-}
            oaieval $solver cant_do_that_anymore.all_diagonal \
                --record_path ${logpathbase}run_${run_idx}_${log_name}.log \
                --extra_eval_params n_samples=1000 \
                --seed ${run_idx}
        done
    done

    echo Done running experiments, all logs in $logpathbase

    echo Producing plots, outputs to $outputdir
    python make_plots.py --log_dir $logpathbase --out_dir $outputdir --diagonal_variant
fi