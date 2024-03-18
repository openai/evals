#!/bin/bash

start_directory="$(pwd)"

find "$start_directory" -type f -name 'requirements.txt' | while read -r file; do
    echo "Installing requirements from: $file"
    pip install -r "$file"
done
