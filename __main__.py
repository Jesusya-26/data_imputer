from data_imputer import DataImputer
from sklearn.metrics import log_loss, mean_absolute_error

input_data = "data_imputer/test_data/living_building_spb.geojson"
imputer = DataImputer(input_data)
imputer.simulation_omission(damage_degree=15, selected_columns=["floors"], save=True)

imputer.add_neighbors_features()

# if you want to create new models
# full_data = imputer.multiple_imputation(save_models=True, save_logs=True)

# if you already have fitted models (you need to put the path to the model in config.prediction.json)
full_data = imputer.impute_by_saved_models()

# to get quality metrics
imputer.get_quality_metrics(classification_metric=log_loss, regression_metric=mean_absolute_error)