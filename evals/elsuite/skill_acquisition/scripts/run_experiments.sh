logdir=./logs
outputdir=./outputs

timestamp=$(date +%Y%m%d_%H%M%S)
logpathbase=$logdir/$timestamp/

size=full
num_repeats=1
eval_variants_zero_shot=("skill_acquisition.miskito.zero_shot.$size")

# Check for --num_repeats argument
for arg in "$@"
do
    if [[ $arg == --num_repeats=* ]]; then
        num_repeats="${arg#*=}"
    fi
done


echo Running experiments and logging to $logpathbase

declare -a ZEROSHOT_SOLVERS=(
    # Solvers for gpt-3.5-turbo
    "generation/direct/gpt-3.5-turbo"
    "skill_acquisition/cot/gpt-3.5-turbo"


    # Solvers for gpt-4-turbo-preview
    "generation/direct/gpt-4-turbo-preview"
    "skill_acquisition/cot/gpt-4-turbo-preview"
)

declare -a FEWSHOT_SOLVERS=(
    "miskito_all/fewshot_direct/gpt-3.5-turbo"
    "miskito_all/fewshot_direct/gpt-4-turbo-preview"
)

if [ ! -d "$logpathbase/miskito" ]; then
    mkdir -p "$logpathbase/miskito" 
fi


# Run zero-shot experiments.
for eval_variant in "${eval_variants_zero_shot[@]}"
do
    if [[ $eval_variant == *"miskito"* ]]; then
        record_path="$logpathbase/miskito"
    fi

    for solver in "${ZEROSHOT_SOLVERS[@]}"
    do
        for ((i=1;i<=num_repeats;i++)); do
            echo "Running $solver, iteration $i"
            oaieval $solver $eval_variant --record_path "$record_path/$solver-$i.log"
        done
    done
done

# Run few-shot experiments.
# Miskito
for solver in "${FEWSHOT_SOLVERS[@]}"
do
    if [[ $solver == *"miskito"* ]]; then
        for ((i=1;i<=num_repeats;i++)); do
            echo "Running $solver, iteration $i"
            oaieval $solver skill_acquisition.miskito.few_shot.$size --record_path "$logpathbase/miskito/$solver-$i.log"
        done
    fi
done

echo Done running experiments, all logs in $logpathbase

echo Producing plots, outputs to $outputdir

# Produce results
python make_plots.py --log-dir $logpathbase --out-dir $outputdir