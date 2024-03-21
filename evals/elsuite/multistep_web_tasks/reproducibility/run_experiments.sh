logdir=./logs
outdir=./outputs
timestamp=$(date +%Y%m%d_%H%M%S)
logpathbase="$logdir/$timestamp"
outpathbase="$outdir/$timestamp"

echo Running experiments and logging to $logpathbase

MODELS="gpt-4-32k-0613 gpt-3.5-turbo-16k-0613"
DATASETS="task_1 task_2 task_3 task_4 task_5 task_6 task_7 task_8 task_9"
N_ATTEMPTS=5
for i in $(seq 0 $(($N_ATTEMPTS - 1)) )
do
    mkdir -p $logpathbase/attempt_${i}
    echo starting attempt ${i} at $(date +%Y%m%d_%H%M%S) > $logpathbase/attempt_${i}/start_time.txt
    for dataset in $DATASETS
    do
        for model in $MODELS
        do
            # echo "Running $model on $dataset for the ${i}th time to $logpathbase/attempt${i}/${model}__$dataset.log"
            base_file_name="$logpathbase/attempt_${i}/${model}__$dataset"
            EVALS_SEQUENTIAL=1 oaieval mwt/strong/$model multistep-web-tasks.$dataset --record_path $base_file_name.log --log_to_file $base_file_name.txt
        done
    done
done

echo Done running experiments, all logs in $logpathbase

echo Producing plots, outputs to $outpathbase
python make_plots.py --log_dir $logpathbase --out_dir $outpathbase