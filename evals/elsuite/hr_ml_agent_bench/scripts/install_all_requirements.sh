#!/bin/bash

script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
start_directory="$(dirname "$script_directory")"

if [[ "$(basename "$start_directory")" != "hr_ml_agent_bench" ]]; then
    echo "Error: The script must be located in a directory within 'hr_ml_agent_bench'."
    exit 1
fi

find "$start_directory" -type f -name 'requirements.txt' | while read -r file; do
    echo "Installing requirements from: $file"
    pip install -r "$file"
    
    if [[ $? -ne 0 ]]; then
        echo "Error: Failed to install requirements from $file"
        exit 1
    fi
done
