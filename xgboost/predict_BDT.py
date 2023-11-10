import argparse
from ModelHandler import *

class MuonMVA(ModelHandler):
    pass

def predict_BDT_model(files, name, config, date, categories=None):
    model = MuonMVA(config)
    model.load_datasets(files, config)

    with open(config, 'r') as file:
        json_file = json.load(file)
    number_of_splits = json_file['number_of_splits']
    output_path = json_file['output_path']
    selections_keys = [key for key in json_file.keys() if key.startswith("selections_")]
    print("pre-selections: ",selections_keys)

    for key in selections_keys:
        model.apply_selection(config, key)

    if categories is None:
        models = model.load_models(name, date)
        model.predict_models(models, name)
        model.mk_bdt_score(models)
        print(model.data.columns.tolist())
        print(model.data)

    if categories is not None:
        models = model.load_models(name, date)
        category_list = categories.split(' ')
        models_per_cat = {k: {} for k in category_list}
        for key, value in models.items():
            for category in category_list:
                if category in key:
                    models_per_cat[category][key] = value
        




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the BDT training")

    parser.add_argument("--config", type=str, help="Path to the JSON configuration file")
    parser.add_argument("--categories", type=str, help="List of categories separated by ' ' ")

    args = parser.parse_args()
    
    config = args.config
    categories = args.categories
    
    with open(config, 'r') as file:
        json_file = json.load(file)
    data_path = json_file['data_path']
    files = json_file['files']
    name = json_file['Name']
    date = json_file['date']
        
    files_Run2022 = [data_path + i for i in files]
    
    predict_BDT_model(files_Run2022, name, config, date, categories)
    #Fa il train sui number_of_splits fold ma poi manca la parte dove vengono aggrgati (?)
    #Quindi bisogna creare un nuovo file .root con il nuovo branch
    #Implementare la divisione in categorie per tau3mu

