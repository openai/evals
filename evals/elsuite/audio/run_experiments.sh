#!/bin/bash
logdir=./logs

completion=$1
maxsamples=${2:-1000}
if [ -z "$completion" ]; then
  echo "Error: No completion specified."
  echo "Usage: $0 <completion>"
  exit 1
fi

timestamp=$(date +%Y%m%d_%H%M%S)
logpathbase=$logdir/$timestamp

echo Running experiments and logging to $logpathbase
oaievalset $completion audio --record_dir ${logpathbase} --max_samples=${maxsamples}

echo Done running experiments, all logs in $logpathbase
python -m evals.elsuite.audio.make_table --log_dir $logpathbase 
