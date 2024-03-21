
logdir=./logs
timestamp=$(date +%Y%m%d_%H%M%S)
logpathbase="$logdir/$timestamp"

echo Running experiments and logging to $logpathbase

#   Baselines
oaieval function_deduction/average_baseline      function_deduction.easy --record_path "$logpathbase/average_baseline.log"
oaieval function_deduction/full_knowledge_best   function_deduction.easy --record_path "$logpathbase/full_knowledge_best.log"
oaieval function_deduction/full_knowledge_random function_deduction.easy --record_path "$logpathbase/full_knowledge_random.log" --extra_eval_params n_repeat=100

declare -a SOLVERS=(
    gpt-3.5-turbo-16k
    gpt-4-32k
    function_deduction/cot/gpt-3.5-turbo-16k
    function_deduction/cot/gpt-4-32k
    function_deduction/gpt-4-base
    function_deduction/cot/gpt-4-base
)

#   Models
for solver in "${SOLVERS[@]}"
do
    log_name=${solver//\//-}
    oaieval $solver function_deduction.easy --record_path "$logpathbase/$log_name.log"
done
