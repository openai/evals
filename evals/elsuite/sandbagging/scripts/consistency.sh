#!/bin/bash
logdir=./logs
outputdir=./outputs

timestamp=$(date +%Y%m%d_%H%M%S)
logpathbase=$logdir/$timestamp/

mkdir -p ${logpathbase}

echo Running experiments and logging to $logpathbase

num_iterations=20
for ((i=0; i<$num_iterations; i++))
do
  oaieval sandbagging/sb_temp1/gpt-4 sandbagging-all-50.v1  \
    --seed $i --record_path ${logpathbase}consistency_gpt4_${i}.log >> ${logpathbase}out.txt
done

python3 consistency_plots.py --log_dir=$logpathbase --out_dir=$outputdir
