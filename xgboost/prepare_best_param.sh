#!/bin/sh
# Usage:
#    prepare_best_param.sh <config_file>

helpstring="Usage:
prepare_best_param.sh [config_file] 
\n-config_file: Name of the json configuration file
"
J_NAME=$1

# Check inputs
if [ -z ${1+x} ]
then
echo -e ${helpstring}
return
fi

cmsenv

if ! command -v jq &> /dev/null; then
    echo "jq non è installato. Installalo prima di continuare."
    exit 1
fi

file_json=$J_NAME

key="date"
date_value=$(date "+%Y%m%d-%H%M%S")
output_path=$(jq -r ".output_path" "$file_json")
name=$(jq -r ".Name" "$file_json")
file_json_new="${output_path}/${date_value}/${name}_config.json"
real_out="${output_path}/${date_value}"
mkdir -p "$real_out"
mkdir "${real_out}/log"
cp "$file_json" "$file_json_new"
jq --arg key "$key" --arg date_value "$date_value" '.[$key] = $date_value' "$file_json_new" > temp.json
mv temp.json "$file_json_new"

cp ./templates/submit.condor ./$real_out
sed -i "s#launch_training.sh#launch_best_param.sh#g" ./${real_out}/submit.condor
sed -i "s#PATH#${real_out}#g" ./${real_out}/submit.condor
number_of_submit=50
echo "request_cpus=4" >> "./${real_out}/submit.condor"
echo "request_memory=4096" >> "./${real_out}/submit.condor"
echo "queue ${number_of_submit}" >> "./${real_out}/submit.condor"
chmod a+x ./${real_out}/submit.condor

cp ./templates/launch_training.sh ./${real_out}/launch_best_param.sh
sed -i "s#launch_training.sh#launch_best_param.sh#g" ./${real_out}/launch_best_param.sh
sed -i "s#train_BDT.py#find_best_param.py#g" ./${real_out}/launch_best_param.sh
sed -i "s#index#random#g" ./${real_out}/launch_best_param.sh
sed -i "s#CXN#$file_json_new#g" ./${real_out}/launch_best_param.sh
chmod a+x ./${real_out}/launch_best_param.sh
echo "Files saved in ${real_out}"


echo "Completed successfully!"
