logdir=./logs
timestamp=$(date +%Y%m%d_%H%M%S)
logpathbase="$logdir/$timestamp"

echo Running experiments and logging to $logpathbase

declare -a SCENARIOS=(
    #   NOTE: no code-davinci-002 against a StrongBot because this rarely fits in the context window
    "code-davinci-002 bluff.honest_bot_highest"
    "code-davinci-002 bluff.gpt-4"

    "gpt-4-base bluff.honest_bot_highest"
    "gpt-4-base bluff.strong_bot"
    "gpt-4-base bluff.gpt-4"

    "gpt-3.5-turbo-16k bluff.honest_bot_highest"
    "gpt-3.5-turbo-16k bluff.strong_bot"
    "gpt-3.5-turbo-16k bluff.gpt-4"

    "gpt-4 bluff.honest_bot_highest"
    "gpt-4 bluff.strong_bot"
)

for scenario in "${SCENARIOS[@]}"
do
    IFS=' ' read -r -a parts <<< "$scenario"
    model=${parts[0]}
    variant=${parts[1]}

    oaieval bluff/strategy/$model $variant --record_path "$logpathbase/$model-$variant.log"
done

python3 make_plots.py --log-dir $logpathbase
