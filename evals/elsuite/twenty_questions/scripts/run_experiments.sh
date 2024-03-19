logdir=./logs
outputdir=./outputs

timestamp=$(date +%Y%m%d_%H%M%S)
logpathbase=$logdir/$timestamp

num_repeats=1

# Check for --num_repeats argument
for arg in "$@"
do
    if [[ $arg == --num_repeats=* ]]; then
        num_repeats="${arg#*=}"
    fi
done

echo Num repeats is: $num_repeats
echo Running experiments and logging to $logpathbase

declare -a SOLVERS=(
    # Solvers for gpt-3.5-turbo
    "generation/direct/gpt-3.5-turbo"
    "twenty_questions/cot/gpt-3.5-turbo"

    # # Solvers for gpt-4-turbo-preview
    "generation/direct/gpt-4-turbo-preview"
    "twenty_questions/cot/gpt-4-turbo-preview"

    # # Solvers for gpt-4-base
    "generation/hhh/gpt-4-base"
    "twenty_questions/cot_hhh/gpt-4-base"
)

if [ ! -d "$logpathbase/standard" ]; then
    mkdir -p "$logpathbase/standard" 
fi

if [ ! -d "$logpathbase/standard" ]; then
    mkdir -p "$logpathbase/shortlist" 
fi

    for solver in "${SOLVERS[@]}"
    do
        for ((i=1;i<=num_repeats;i++))
        do
            echo "Running $solver, iteration $i, standard variant."
            oaieval $solver twenty_questions.full --record_path "$logpathbase/standard/$solver-$i.log"

            echo "Running $solver, iteration $i, shortlist variant."
            oaieval $solver twenty_questions.shortlist.full --record_path "$logpathbase/shortlist/$solver-$i.log"
        done
    done
done

echo Done running experiments, all logs in $logpathbase

echo Producing plots, outputs to $outputdir

# Produce results
python scripts/make_plots.py --log-dir $logpathbase --out-dir $outputdir