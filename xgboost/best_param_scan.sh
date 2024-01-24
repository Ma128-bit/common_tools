#!/bin/sh
# Usage:
#    prepare_best_param.sh

declare -a max_depth=(4 5 7 6)
declare -a learning_rate=(0.05 0.075 0.1)
declare -a n_estimators=(500 700 900 1000)
declare -a subsample=(0.7 0.8 0.9)
declare -a colsample_bytree=(0.7 0.8 0.9)
declare -a min_child_weight=(6 8 10)
declare -a gamma=(0.01 0.1 0.5 1)
declare -a reg_alpha=(0.01 0.1 0.5 1)
declare -a reg_lambda=(0.01 0.1 0.5 1)

for i in {0..60}; do
  random_I_max_depth=$((RANDOM % 4))
  random_max_depth=${max_depth[$random_I_max_depth]}
  
  random_I_learning_rate=$((RANDOM % 4))
  random_learning_rate=${learning_rate[$random_I_learning_rate]}
  
  random_I_n_estimators=$((RANDOM % 5))
  random_n_estimators=${n_estimators[$random_I_n_estimators]}
  
  random_I_subsample=$((RANDOM % 3))
  random_subsample=${subsample[$random_I_subsample]}

  random_I_colsample_bytree=$((RANDOM % 3))
  random_colsample_bytree=${colsample_bytree[$random_I_colsample_bytree]}

  random_I_min_child_weight=$((RANDOM % 4))
  random_min_child_weight=${min_child_weight[$random_I_min_child_weight]}

  random_I_gamma=$((RANDOM % 4))
  random_gamma=${gamma[$random_I_gamma]}

  random_I_reg_alpha=$((RANDOM % 4))
  random_reg_alpha=${reg_alpha[$random_I_reg_alpha]}

  random_I_reg_lambda=$((RANDOM % 4))
  random_reg_lambda=${reg_lambda[$random_I_reg_lambda]}

  cp ./config/config_tau3mu_template.json ./config_tau3mu.json
  sed -i "s#max_depth_val#${random_max_depth}#g" ./config_tau3mu.json
  sed -i "s#learning_rate_val#${random_learning_rate}#g" ./config_tau3mu.json
  sed -i "s#n_estimators_val#${random_n_estimators}#g" ./config_tau3mu.json
  sed -i "s#subsample_val#${random_subsample}#g" ./config_tau3mu.json
  sed -i "s#colsample_bytree_val#${random_colsample_bytree}#g" ./config_tau3mu.json
  sed -i "s#min_child_weight_val#${random_min_child_weight}#g" ./config_tau3mu.json
  sed -i "s#gamma_val#${random_gamma}#g" ./config_tau3mu.json
  sed -i "s#reg_alpha_val#${random_reg_alpha}#g" ./config_tau3mu.json
  sed -i "s#reg_lambda_val#${random_reg_lambda}#g" ./config_tau3mu.json

  output=$(source prepare_condor.sh config_tau3mu.json "Cat_A Cat_B Cat_C")
  string=$(echo "$output" | grep "Completed successfully" -B 1 | head -n 1)
  path=$(echo $string | grep -o 'results/BDT/[^ ]*')

  echo "submit ${path}"
  
  echo "${path}, ${random_max_depth}, ${random_learning_rate}, ${random_n_estimators}, ${random_subsample}, ${random_colsample_bytree}, ${random_min_child_weight}, ${random_gamma}, ${random_reg_alpha}, ${random_reg_lambda}" > parametri.txt
  
  condor_submit -name ettore "${path}/submit_Cat_A.condor"
  condor_submit -name ettore "${path}/submit_Cat_B.condor"
  condor_submit -name ettore "${path}/submit_Cat_C.condor"

  rm config_tau3mu.json
  
  done
  
