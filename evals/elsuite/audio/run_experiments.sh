logdir=./logs
outputdir=./outputs
# export EVALS_THREADS=3

timestamp=$(date +%Y%m%d_%H%M%S)
logpathbase=$logdir/$timestamp/

echo Running experiments and logging to $logpathbase
oaievalset generation/direct/ultravox-v0.2 audio --record_path ${logpathbase}/payload.log --max_samples=100

echo Done running experiments, all logs in $logpathbase

echo Producing plots, outputs to $outputdir
#python make_plots.py --log_dir $logpathbase --out_dir $outputdir