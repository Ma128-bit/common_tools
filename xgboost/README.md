# XGBoost - Boosted Decision Trees
Note: The code is a modified version of the one used in the [Bmm5 analysis](https://github.com/drkovalskyi/Bmm5/blob/master/MVA/ModelHandler.py)

## Configuration file 
ModelHandler.py is a wrapper around XGBoost. All you need to do is create a json configuration file that contains the following information:
* `feature_names` : name of the branches of the tree used for the BDT training. No feature wit name that contains "fold" or "bdt" or "bdt_cv"
* `other_branches` : other branches that you keed (for Y set, selections, weight, etc...). No branch wit name that contains "fold" or "bdt" or "bdt_cv"
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

An example of configuration file is **`config/config.json`**

### Instructions for selections_ :
The code automatically run all the pre-selections `selections_*` listed in the configuration file.

Each selection have this shape:
```python=
"selections_1": [
    {"highPurity": [">", 0]},
    {"pt": [">", 2]},
    {"eta": ["<", 2.5]},
    {"eta": [">", -2.5]},
    {"isGlobal": ["==", true], "isTracker": ["!=", true]}
]
```
Notes: 
* At this moment, only >, <, == and != operators are supported. 
* Different rows (example `{"highPurity": [">", 0]}` and `{"pt": [">", 2]}`) are in AND
* Element on the same row (like: `{"isGlobal": ["==", true], "isTracker": ["!=", true]}`) are in OR.
    * **ATTENTION!** To use the OR **all** keys in the row (in the example `isGlobal` and `isTracker`) must be different otherwise the code won't work!
* It is also possible to implement the NOT of an entire selection writing "!" at the end of the name: `selections_1!`

### Instructions for splitting in categories :
Categories are treated as selections. For example, if you want a category called "Cat_A", you have to add selections like:
```python=
"Cat_A_1": [
    (like selections_1 in example before)
]
```
All the selections that starts with `Cat_A` will be applied.

There are also special selections: 
* If a selection, that starts with `Cat_A`, contains `_sig_` (like `Cat_A_sig_2`) it is applied only on signal data
* If a selection, that starts with `Cat_A`, contains `_bkg_` (like `Cat_A_bkg_3`) it is applied only on bkg data

NOTE: Please in category names do not use strange characters (especially '/' and '#') or spaces, possibly only use letters and '_'

## Training
The script **`train_BDT.py`** allows for the training of the model. It has 3 inputs:

`python3 train_BDT.py --config [config_file] --index [kfold_number] --category [category_name]`

* [config_file]: **Mandatory**  Name of the configuration file (for example **`config/config.json`**)
* [kfold_number]: **Optional**  ID of the fold (for example if `number_of_splits` in the configuration file is 5, kfold_number can be chosen between 0 and 4). If the parameter is not inserted the training will be performed on all folds.
* [category_name]: **Optional** Is used to train the mode on a category. When you pass this all the selections that starts with [category_name] will be applied.

### Output of **`train_BDT.py`** 
In the folder `output_path` it create a directory having the date in %Y%m%d-%H%M%S format as its name. In this directory there is: 
* A copy of the configuration file
* All the model trained (saved as .pkl)
* The validation plots

## Training on the batch system 
First use **`prepare_condor.sh`** to create the files for condor submission
```
source prepare_condor.sh [config_file] [categories_names]
```
* [config_file]: **Mandatory**  Name of the configuration file (for example **`config/config.json`**)
* [category_name]: **Mandatory** List of categories (like "Cat_A Cat_B Cat_C"). If there are no categories you have to pass ""

Example: `source prepare_condor.sh config/config_tau3mu.json "Cat_A Cat_B Cat_C"`

### Submission
**`prepare_condor.sh`** creates the directory **`output_path`/date** (in %Y%m%d-%H%M%S format) with a copy of the configuration file and files for submission (submit.condor, launch_training.sh one per categoy) 

To submit the jobs, form the main directory, run the submission of **`output_path`/date/submit.condor** (or submit_*.condor with * = categoy name)

