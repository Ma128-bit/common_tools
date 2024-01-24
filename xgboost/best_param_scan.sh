#!/bin/sh
# Usage:
#    prepare_best_param.sh

declare -a max_depth=(3 5 7 10)
declare -a learning_rate=(0.01 0.05 0.1 0.3)
declare -a n_estimators=(300 600 100)
declare -a subsample=(0.7 0.8 0.9)
declare -a colsample_bytree=(0.7 0.8 0.9)
declare -a min_child_weight=(2 5 8 11)
declare -a gamma=(0.01 0.1 0.5 1)
declare -a reg_alpha=(0.01 0.1 0.5 1)
declare -a reg_lambda=(0.01 0.1 0.5 1)

for i in {0..7}
  random_I_max_depth=$((RANDOM % ${#max_depth[@]}))
  random_max_depth=${max_depth[$random_I_max_depth]}
  
  random_I_learning_rate=$((RANDOM % ${#learning_rate[@]}))
  random_learning_rate=${max_depth[$random_I_learning_rate]}
  
  random_I_n_estimators=$((RANDOM % ${#n_estimators[@]}))
  random_n_estimators=${max_depth[$random_I_n_estimators]}
  
  random_I_subsample=$((RANDOM % ${#subsample[@]}))
  random_subsample=${max_depth[$random_I_subsample]}

  random_I_colsample_bytree=$((RANDOM % ${#colsample_bytree[@]}))
  random_colsample_bytree=${max_depth[$random_I_colsample_bytree]}

  random_I_min_child_weight=$((RANDOM % ${#min_child_weight[@]}))
  random_min_child_weight=${max_depth[$random_I_min_child_weight]}

  random_I_gamma=$((RANDOM % ${#gamma[@]}))
  random_gamma=${max_depth[$random_I_gamma]}

  random_I_reg_alpha=$((RANDOM % ${#reg_alpha[@]}))
  random_reg_alpha=${max_depth[$random_I_reg_alpha]}

  random_I_reg_lambda=$((RANDOM % ${#reg_lambda[@]}))
  random_reg_lambda=${max_depth[$random_I_reg_lambda]}


  echo "Elemento casuale: ${random_max_depth}, ${random_learning_rate}, ${random_n_estimators}, ${random_subsample}, ${random_colsample_bytree}, ${random_min_child_weight}, ${random_gamma}, ${random_reg_alpha}, ${random_reg_lambda}"



  
