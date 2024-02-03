#!/bin/bash
for paper in /root/uni-finder/enzyme/paper/*.pdf
do
	file_name="${paper##*/}"
	name=${file_name%.*}
	key_word=""
	key_word=$(grep "${name}" samples.jsonl)
        if [[ ${key_word} == "" ]]
	then
		echo "add ${name} to jsonl"
		sed 's|target_mark|'"${name}"'|g' sample_file >> samples.jsonl
	else
		echo "${name}: was already in the jsonl"
	fi
done
