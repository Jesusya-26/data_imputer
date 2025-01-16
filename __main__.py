from data_imputer import DataImputer

dataset_path = "data_imputer/simulations/living_building_spb_4326.geojson"
imputer = DataImputer(dataset_path)
imputer.add_neighbors_features()

# if you want to create new models
full_data = imputer.multiple_imputation(save_models=True, save_logs=True)

# # if you already have fitted models (you need to put the path to the model in config.prediction.json)
# full_data = imputer.impute_by_saved_models()