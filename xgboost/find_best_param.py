import argparse
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from ModelHandler import *

class MuonMVA(ModelHandler):
    pass
#warnings.filterwarnings("default", category=UserWarning, module="numpy")

def best_par(files_Run2022, name, config, date, random=0, condor=False):
    model = MuonMVA(config)
    model.load_datasets(files_Run2022, config)

    with open(config, 'r') as file:
        json_file = json.load(file)
    output_path = json_file['output_path']
    selections_keys = [key for key in json_file.keys() if key.startswith("selections_")]
    print("pre-selections: ",selections_keys)

    for key in selections_keys:
        model.apply_selection(config, key)

    output_path_new = output_path + "/" + date
    if not os.path.exists(output_path_new):
        subprocess.call("mkdir -p %s" % output_path_new, shell=True)

    if not condor:
        config_out = "%s/%s_config.json" % (output_path_new, name)
        subprocess.run(["cp", config, config_out])
        with open(config_out, 'r') as file:
            dati = json.load(file)
        new_key = 'date'
        dati[new_key] = date
        with open(config_out, 'w') as file:
            json.dump(dati, file, indent=2)


    print("prepare train and test datasets:")
    model.prepare_train_and_test_datasets(config, 10, 0)
    print("Done!")
    """
    fixed_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'early_stopping_rounds':20,
    }
    """
    param_dist = {
        'max_depth': range(3,10,1),
        'learning_rate': [i/50.0 for i in range(1,15)],
        'n_estimators': [i*100 for i in range(1,10)],
        'subsample': [i/10.0 for i in range(6,10)],
        'colsample_bytree': [i/10.0 for i in range(6,10)],
        'min_child_weight':  [1, 5, 10, 15, 20],
        'gamma': [i/10.0 for i in range(1,10)],
        'reg_alpha': [i/10.0 for i in range(0,50)],
        'reg_lambda': [i/10.0 for i in range(0,50)],
        'objective': ['binary:logistic'],
        'eval_metric': ['auc'],
        'early_stopping_rounds':[20],
    }
    
    #params = {**fixed_params, **param_dist}

    xgbR = xgb.XGBRegressor()
    N_jobs=-1
    if condor==True:
        N_jobs=-1
        
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=(42 + random * 15))
    
    random_search = RandomizedSearchCV(
        xgbR, param_distributions=param_dist, n_iter=10, verbose=2, scoring='roc_auc', cv=cv, random_state=(42 + random * 15), n_jobs=N_jobs
    )
    
    random_search.fit(model.x_train, model.y_train, verbose=False, sample_weight=model.train_weights, eval_set=[(model.x_train, model.y_train)])
    """    
    random_search = RandomizedSearchCV(
        xgbR, param_distributions=param_dist, n_iter=10, verbose=2, scoring='roc_auc', cv=2, random_state=(42+random*15), n_jobs=N_jobs
    )

    print("Start fit:")
    random_search.fit(model.x_train, model.y_train, verbose = False, sample_weight = model.train_weights, eval_set=[(model.x_train, model.y_train)])
    print("Done!")
    """
    print("Parametri ottimali:", random_search.best_params_)
    print("Miglior AUC:", random_search.best_score_)
    
    test_auc = random_search.score(model.x_test, model.y_test)
    print("AUC sui dati di test:", test_auc)

    file_out = "%s/%s_results_%d.txt" % (output_path_new, name, random)

    with open(file_out, "w") as file:
        file.write("Optimal parameters: {}\n".format(random_search.best_params_))
        file.write("Best AUC: {}\n".format(random_search.best_score_))
        
        test_auc = random_search.score(model.x_test, model.y_test)
        file.write("AUC on test data: {}\n".format(test_auc))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--config", type=str, help="Path to the JSON configuration file")
    parser.add_argument("--condor", action="store_true", help="Use of condor")
    parser.add_argument("--random", type=int, help="random_state")

    args = parser.parse_args()
    
    config = args.config
    condor = args.condor
    random = args.random

    date = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    with open(config, 'r') as file:
        json_file = json.load(file)
    data_path = json_file['data_path']
    files = json_file['files']
    name = json_file['Name']
    if condor:
        date = json_file['date']
        
    files_Run2022 = [data_path + i for i in files]
    
    best_par(files_Run2022, name, config, date, random, condor)
