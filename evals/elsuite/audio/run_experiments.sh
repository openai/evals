#!/bin/bash
logdir=./logs

timestamp=$(date +%Y%m%d_%H%M%S)
logpathbase=$logdir/$timestamp

echo Running experiments and logging to $logpathbase
oaievalset --record_dir ${logpathbase} $@

echo Done running experiments, all logs in $logpathbase
python -m evals.elsuite.audio.make_table --log_dir $logpathbase 
