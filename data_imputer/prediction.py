import glob
import json
import os

import numpy as np
import pandas as pd
import sklearn
from joblib import dump, load
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.exceptions import NotFittedError
from sklearn.experimental import enable_halving_search_cv
from sklearn.metrics import make_scorer
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor



"""If column name is in categorical_features list classification predictive method will be used.
For other columns will be used regression predictive method.
To find the best combination hyperparametrs we use sklearn implementation of Successive Halving algorithm. 
Candidate parameter values are specified by user in config/config_learning.json."""


def learn_and_predict(
    x_learn: DataFrame,
    y_learn: Series,
    x_predict: DataFrame,
    is_categorical: bool,
    is_positive: bool,
    bunch_scores: list,
    save_log: bool,
    save_model: bool,
    path: str | None,
):

    model = build_model(
        x_learn, y_learn, is_categorical, bunch_scores, save_model, save_log, path
    )
    prediction = predict_by_model(model, x_predict, is_positive)
    return prediction


def build_model(
    x_learn: DataFrame,
    y_learn: Series,
    is_categorical: bool,
    scores: dict,
    save_model: bool,
    save_log: bool,
    path: str | None = None,
):
    """
    Builds and trains a model for either regression or classification tasks,
    based on the provided data and configuration.

    Parameters:
    -----------
    x_learn : DataFrame
        Input features for training.
    y_learn : Series
        Target variable for training.
    is_categorical : bool
        Indicates whether the target variable is categorical (classification) or continuous (regression).
    scores : dict
        A dictionary to store model scores for each target variable.
    save_model : bool
        If True, saves the trained model to disk.
    save_log : bool
        If True, saves training logs to disk.
    path : str or None, optional
        Path to the configuration file and where models/logs should be saved. Defaults to the current working directory.

    Returns:
    --------
    opt : HalvingGridSearchCV
        The trained model object with optimized parameters.

    Raises:
    -------
    FileNotFoundError:
        If the configuration file is not found at the specified path.
    ValueError:
        If the configuration parameters are invalid or missing.
    NotFittedError:
        If the model fails to fit the data during training.
    """
    # Set default path if not provided
    if path is None:
        path = os.getcwd()

    config_path = os.path.join(path, "data_imputer", "config", "config_learning.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Load configuration
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    # Determine the task type
    method = "classification" if is_categorical else "regression"

    try:
        # Construct pipeline
        pipe = Pipeline(
            [
                ("transform", eval(config[method]["transform"])),
                ("model", eval(config[method]["model"])),
            ]
        )

        # Configure hyperparameter search space
        gbdt_search = config[method]["grid_search_param"]
        gbdt_search["model"] = eval(gbdt_search["model"])
        gbdt_search["model__max_features"] = eval(gbdt_search["model__max_features"])

        # Configure learning parameters
        learn_param = config[method]["learn_param"]

        # Correct the scorer definition to avoid issues with unsupported arguments
        scoring_params = learn_param.get("scoring", {})
        scoring_func = eval(scoring_params.pop("score_func"))
        scorer = make_scorer(scoring_func, **scoring_params)
        learn_param["scoring"] = scorer

        # Initialize and fit the model using HalvingGridSearchCV
        opt = HalvingGridSearchCV(pipe, gbdt_search, **learn_param)
        opt.fit(x_learn.to_numpy(), y_learn.to_numpy())

        # Save scores, model, and logs
        if y_learn.name not in scores:
            scores[y_learn.name] = []
        scores[y_learn.name].append(opt.best_score_)

        if save_model:
            save_model_object(opt, y_learn.name)
        if save_log:
            save_logs_object(opt, y_learn.name)

    except NotFittedError:
        print(f"Model fitting failed for target '{y_learn.name}'. Retrying...")
        return build_model(x_learn, y_learn, is_categorical, scores, save_model, save_log, path)
    except ValueError as e:
        print(f"Configuration error: {e}")
        raise

    return opt


def predict_by_model(model, x_predict: DataFrame, is_positive: bool):
    y_predict = model.predict(x_predict)
    y_predict = [0 if y < 0 else y for y in y_predict] if is_positive else y_predict
    return y_predict


def parse_config_models(columns: list, path: str | None = None) -> dict:
    if not path:
        path = os.getcwd()
    with open(
        path + "/data_imputer/config/config_prediction.json",
        encoding="utf-8",
    ) as f:
        config = json.load(f)

    folder = os.path.join(path, config["folder"])
    existed_models = os.listdir(folder)
    existed_models = {
        k: next((m for m in existed_models if k in m), None) for k in columns
    }
    existed_model_objects = {
        k: load(os.path.join(folder, v)) if v is not None else v
        for k, v in existed_models.items()
    }
    return existed_model_objects


def save_logs_object(model, model_name: str) -> None:
    cv_result = pd.DataFrame(model.cv_results_)
    folder_path = os.path.join(os.getcwd(), "data_imputer", "logs")
    folders = glob.glob(os.path.join(folder_path, "*/"))

    if not folders:
        raise ValueError(f"No subfolders found in {folder_path}")

    last_created_folder = max(folders, key = os.path.getmtime)
    log_file = os.path.join(last_created_folder, model_name + ".xlsx")
    cv_result.to_excel(log_file, engine="openpyxl")


def save_model_object(model, model_name: str) -> None:
    folder_path = os.path.join(os.getcwd(), "data_imputer", "fitted_model")
    folders = glob.glob(os.path.join(folder_path, "*/"))

    if not folders:
        raise ValueError(f"No subfolders found in {folder_path}")

    last_created_folder = max(folders, key = os.path.getmtime)
    path = os.path.join(last_created_folder, model_name + "_GBDT.joblib")
    dump(model, path)

