logdir=./logs
outputdir=./outputs
timestamp=$(date +%Y%m%d_%H%M%S)
logpathbase="$logdir/$timestamp/"

echo Running experiments and logging to $logpathbase

DATASETS="tomi socialiqa hitom"
MODELS="gpt-3.5-turbo gpt-4 gpt-4-base"
SOLVER_TYPES="simple_solver cot_solver"

for dataset in $DATASETS
do
    for model in $MODELS
    do
        for solver in $SOLVER_TYPES
        do
            oaieval $dataset/$solver/$model "theory_of_mind."$dataset --record_path "$logpathbase/$model-$variant.log"
        done
    done
done

echo Done running experiments, all logs in $logpathbase

echo Producing plots, outputs to $outputdir
python3 make_plots.py --log_dir $logpathbase --out_dir $outputdir