The code is a modified version of the one used in the Bmm5 analysis https://github.com/drkovalskyi/Bmm5/blob/master/MVA/ModelHandler.py 

ModelHandler.py is a wrapper around XGBoost. All you need to do is create a json configuration file that contains the following information:
* `feature_names` : name of the branches of the tree used for the BDT training
* `other_branches` : other branches that you keed (for Y set, selections, weight, etc...)
* `Y_column` : The name of the branch used to create the Y set
* `BDT_0` : is the value of Y_column associated to 0 in the Y set, values of Y_column different from BDT_0 are set to 1 in the Y set
* `selections_*` : List of pre-selections (* = 1,2,3,...)
* `xgb_parameters` : Parameter of the XGBoost model
* `tree_name` : Name of the input tree in the root files
* `output_path` : Output path for saving results
* `event_fraction` : Fraction of root files used (usefull in case of big datasets)
* `validation_plots` : Boolean option to save validation plots
* `number_of_splits` : Number of kfoler
* `do_weight` : Boolean option to use weights
* `weight_column` : Tree branch used as weight
* `Name` : Customizable name that will be inserted into the output files
* `data_path` : Path where the datasets are saved
* `files` : List of datasets in data_path

An example of configuration file is **`config.json`**
