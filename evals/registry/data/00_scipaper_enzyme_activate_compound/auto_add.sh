#!/bin/bash
target_job=$1
if [[ ${target_job} == "" ]]
then
	echo ">>> Error: target_job is not define"
	exit
fi
if [[ ! -f samples.jsonl ]]
then
	touch samples.jsonl
fi
for paper in /root/uni-finder/enzyme/"${target_job}"/paper/*.pdf
do
	echo "find file ${paper}"
	file_name="${paper##*/}"
	name=${file_name%.*}
	key_word=""
	key_word=$(grep "${name}" samples.jsonl)
        if [[ ${key_word} == "" ]]
	then
		echo "add ${name} to jsonl"
		sed 's|target_mark|'"${name}"'|g' sample_file | sed 's|target_Job|'"${target_job}"'|g' >> samples.jsonl
	else
		echo "${name}: was already in the jsonl"
	fi
done
