logdir=./logs
outputdir=./outputs
export EVALS_THREADS=50

timestamp=$(date +%Y%m%d_%H%M%S)
logpathbase=$logdir/$timestamp/

echo Running experiments and logging to $logpathbase

declare -a SOLVERS=(
    # Solvers for gpt-4-base
    "self_prompting/completion_hhh/gpt-4-base"
    # Solvers for code-davinici-002
    "self_prompting/completion_hhh/code-davinci-002"
    # Solvers for gpt-3.5-turbo-16k
    "self_prompting/direct/gpt-3.5-turbo-16k"
    "self_prompting/cot/gpt-3.5-turbo-16k"
    "self_prompting/cotexpert/gpt-3.5-turbo-16k"
    # Solvers for gpt-4-32k
    "self_prompting/direct/gpt-4-32k"
    "self_prompting/cot/gpt-4-32k"
    "self_prompting/cotexpert/gpt-4-32k"
    # Baseline solvers
    "self_prompting/oriprompt/baseline"
    "self_prompting/noprompt/baseline"
    "self_prompting/fewshot/baseline"
)

for solver in "${SOLVERS[@]}"
do
    oaieval $solver self_prompting --record_path "$logpathbase/$solver.log"
done

echo Done running experiments, all logs in $logpathbase

echo Producing plots, outputs to $outputdir

# Produce results
python make_plots.py --log_dir $logpathbase --out_dir $outputdir