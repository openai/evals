logdir=./logs

timestamp=$(date +%Y%m%d_%H%M%S)
logpathbase=$logdir/$timestamp
maxsamples=1000

echo Running experiments and logging to $logpathbase
oaievalset generation/direct/ultravox-v0.2 audio --record_dir ${logpathbase} --max_samples=${maxsamples}

echo Done running experiments, all logs in $logpathbase
python evals/elsuite/audio/make_table.py --log_dir $logpathbase 
