#!/bin/bash

usage() {
  echo "Usage: $0 -l logdir"
  echo "  -l logdir     Specify the directory for log files"
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
    l) logdir=${OPTARG} ;;
    *) usage ;;
  esac
done

# Check if mandatory arguments were provided
if [ -z "$logdir" ]; then
  usage
  exit 1
fi

NUM_REPEATS=3

export EVALS_THREADS=10
export EVALS_THREADS_TIMEOUT=5

declare -a SOLVERS=(
  # gpt-4-turbo-preview
  "generation/direct/gpt-4-turbo-preview"
  "already_said_that/cot/gpt-4-turbo-preview"
  # gpt-3.5-turbo
  "generation/direct/gpt-3.5-turbo"
  "already_said_that/cot/gpt-3.5-turbo"
  # gpt-4-base
  "generation/hhh/gpt-4-base"
  # mixtral-8x7b-instruct
  "generation/direct/mixtral-8x7b-instruct"
  # llama chat 70b
  "generation/direct/llama-2-70b-chat"
  # gemini-pro
  "generation/direct/gemini-pro"
  # random baseline
  "already_said_that/random_baseline"
)

declare -a DISTRACTORS=(
  "reverse-sort-words-eng"
  "first-letters"
  "ambiguous-sentences"
  "which-is-heavier"
  "distractorless"
)

# Check if GEMINI_API_KEY is set
if [ -z "$GEMINI_API_KEY" ]; then
  echo "Enter your Gemini API Key:"
  read -s GEMINI_API_KEY
  export GEMINI_API_KEY
fi

# Check if TOGETHER_API_KEY is set
if [ -z "$TOGETHER_API_KEY" ]; then
  echo "Enter your Together API Key:"
  read -s TOGETHER_API_KEY
  export TOGETHER_API_KEY
fi

start_time=$SECONDS
for solver in "${SOLVERS[@]}"; do

  if [[ $solver == *"gemini"* ]]; then
    export EVALS_SEQUENTIAL=1
  else
    export EVALS_SEQUENTIAL=0
  fi

  solver_dotted=${solver//\//.}

  for ((i = 1; i <= NUM_REPEATS; i++)); do
    for distractor in "${DISTRACTORS[@]}"; do
      record_path="${logdir}/${solver_dotted}_${distractor}_${i}"
      echo "Running $solver with $distractor, seed $i"
      if [[ $solver == *"cot"* ]]; then
        oaieval $solver "already_said_that.${distractor}" \
          --seed $i --record_path "$record_path.log" \
          --completion_args persistent_memory=False
      else
        oaieval $solver "already_said_that.${distractor}" \
          --record_path "$record_path.log" \
          --seed $i
      fi
    done
  done
done
echo "Total time: $((SECONDS - start_time)) seconds"
