logdir=./logs
outputdir=./outputs
# export EVALS_THREADS=3

timestamp=$(date +%Y%m%d_%H%M%S)
logpathbase=$logdir/$timestamp/

echo Running experiments and logging to $logpathbase

# Baselines
oaieval gpt-4,gpt-4 steganography.copypayload --record_path ${logpathbase}/gpt-4_copypayload.log
oaieval gpt-4,gpt-4 steganography.task_payload --record_path ${logpathbase}/gpt-4_task_payload.log
oaieval gpt-4,gpt-4 steganography.taskonly --record_path ${logpathbase}/gpt-4_taskonly.log
# gpt-3.5-turbo
oaieval gpt-3.5-turbo,gpt-4 steganography.direct --record_path ${logpathbase}/gpt-3.5-turbo_direct.log
oaieval gpt-3.5-turbo,gpt-4 steganography.direct_ref --record_path ${logpathbase}/gpt-3.5-turbo_direct_ref.log
oaieval gpt-3.5-turbo,gpt-4 steganography.scratch --record_path ${logpathbase}/gpt-3.5-turbo_scratch.log
oaieval gpt-3.5-turbo,gpt-4 steganography.scratch_ref --record_path ${logpathbase}/gpt-3.5-turbo_scratch_ref.log
# gpt-4
oaieval gpt-4,gpt-4 steganography.direct --record_path ${logpathbase}/gpt-4_direct.log
oaieval gpt-4,gpt-4 steganography.direct_ref --record_path ${logpathbase}/gpt-4_direct_ref.log
oaieval gpt-4,gpt-4 steganography.scratch --record_path ${logpathbase}/gpt-4_scratch.log
oaieval gpt-4,gpt-4 steganography.scratch_ref --record_path ${logpathbase}/gpt-4_scratch_ref.log

echo Done running experiments, all logs in $logpathbase

echo Producing plots, outputs to $outputdir
python make_plots.py --log_dir $logpathbase --out_dir $outputdir