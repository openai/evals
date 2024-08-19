#!/bin/bash

# Function to display usage
usage() {
  echo "Usage: $0 -s size -l logdir"
  echo "  -s size    Specify the size of the experiments (options: 'balanced-hypotheses', 'balanced-ctrl', 'balanced-hypotheses-large', 'balanced-ctrl-large')"
  echo "  -l logdir  Specify the directory for log files"
  exit 1
}

# Check if no arguments were provided
if [ $# -eq 0 ]; then
  usage
  exit 1
fi

# Parse command-line options
while getopts 's:l:' flag; do
  case "${flag}" in
    s) size=${OPTARG} ;;
    l) logdir=${OPTARG} ;;
    *) usage ;;
  esac
done

# Check if mandatory arguments were provided
if [ -z "$size" ] || [ -z "$logdir" ]; then
  usage
  exit 1
fi

logdirbase=$logdir
NUM_REPEATS=3

# Function to run experiments
run_experiments() {
  local size=$1
  local logpathbase="${logdirbase}/${size}"
  local start_time=$SECONDS

  # Define RENDERERS and SOLVERS array based on size
  declare -a RENDERERS
  declare -a SOLVERS
  if [ "$size" == "balanced-hypotheses" ]; then
    RENDERERS=("markdown" "csv" "json" "language-tabular" "language-corrset" "corrset")
    SOLVERS=("generation/direct/gpt-3.5-turbo"
      "generation/cot/gpt-3.5-turbo"
      "generation/hhh/gpt-4-base"
      "generation/cot_hhh/gpt-4-base"
      "generation/direct/gpt-4-1106-preview"
      "generation/cot/gpt-4-1106-preview")
  elif [ "$size" == "balanced-ctrl" ]; then
    RENDERERS=("csv" "language-corrset")
    SOLVERS=("generation/direct/gpt-3.5-turbo"
      "generation/cot/gpt-3.5-turbo"
      "generation/hhh/gpt-4-base"
      "generation/cot_hhh/gpt-4-base"
      "generation/direct/gpt-4-1106-preview"
      "generation/cot/gpt-4-1106-preview")
  else
    RENDERERS=("csv" "language-corrset")
    SOLVERS=("generation/direct/gpt-4-1106-preview")
  fi

  # Main loop
  for ((i = 1; i <= NUM_REPEATS; i++)); do
    for solver in "${SOLVERS[@]}"; do
      for renderer in "${RENDERERS[@]}"; do
        run_solver $solver $renderer $size $i "$logpathbase"
      done
    done
    run_solver "identifying_variables/random" "corrset" $size $i "$logpathbase"
    run_solver "identifying_variables/noctrl" "corrset" $size $i "$logpathbase"
  done

  local end_time=$SECONDS
  echo "Done running experiments for $size size, all logs in $logpathbase"
  echo "Total execution time: $((end_time - start_time)) seconds."
}

# Function to run a single solver
run_solver() {
  local solver=$1
  local renderer=$2
  local size=$3
  local seed=$4
  local logpathbase=$5
  local solver_dotted=${solver//\//.}

  local record_path="${logpathbase}/${solver_dotted}_${renderer}_${size}_${seed}"
  echo "Running $solver with $renderer renderer and $size data size; seed $seed"

  local sub_start_time=$(date +%s)
  oaieval "$solver" "identifying_variables.${renderer}.${size}" --record_path "$record_path.log" --seed $seed
  local sub_end_time=$(date +%s)
  echo "${solver_dotted}_${renderer}_${size} execution time: $((sub_end_time - sub_start_time)) seconds."

  skip_tree_solvers=("identifying_variables/random" "identifying_variables/noctrl")
  if [[ ! "${skip_tree_solvers[@]}" =~ "$solver" ]] && [ "$size" == "balanced-hypotheses" ]; then
    echo "Now repeating with show_tree=True"
    oaieval "$solver" "identifying_variables.${renderer}.${size}" --extra_eval_params show_tree=True --record_path "${record_path}_tree.log" --seed $seed
  fi
}

run_experiments "${size}"
