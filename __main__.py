from data_imputer import DataImputer
from sklearn.metrics import f1_score, mean_absolute_error

input_data = "data_imputer/simulations/living_building_spb_170125_153305.geojson"
real_data = "data_imputer/test_data/living_building_spb_170125_153305.geojson"
imputer = DataImputer(input_data, real_data)
imputer.add_neighbors_features()

# if you want to create new models
full_data = imputer.multiple_imputation(save_models=True, save_logs=True)

# if you already have fitted models (you need to put the path to the model in config.prediction.json)
# full_data = imputer.impute_by_saved_models()

# to get quality metrics
imputer.get_quality_metrics(classification_metric=f1_score, regression_metric=mean_absolute_error)