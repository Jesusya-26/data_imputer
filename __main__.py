from data_imputer import DataImputer

dataset_path = "data_imputer/simulations/living_building_spb_140822_080633.geojson"
imputer = DataImputer(dataset_path)
imputer.add_neighbors_features()
full_data = imputer.multiple_imputation(save_models=True, save_logs=True)