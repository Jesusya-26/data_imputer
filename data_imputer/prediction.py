"""
If column name is in categorical_features list classification predictive method will be used.
For other columns will be used regression predictive method.

To find the best combination hyperparametrs we use sklearn implementation of Successive Halving algorithm.
Candidate parameter values are specified by user in config/config_learning.json.
"""
import glob
import json
import os
from typing import Any, TextIO
from pathlib import Path

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
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor


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
    """
    Trains a model on the given learning data and predicts values on the provided prediction data.

    Parameters:
        x_learn (DataFrame): The feature data for training the model.
        y_learn (Series): The target values for training the model.
        x_predict (DataFrame): The feature data for making predictions.
        is_categorical (bool): Whether the target variable is categorical or continuous.
        is_positive (bool): If True, ensure all predictions are non-negative.
        bunch_scores (list): A list of scoring functions for model evaluation.
        save_log (bool): Whether to save a log of the training process.
        save_model (bool): Whether to save the trained model.
        path (str | None): The directory path where the model and logs should be saved (optional).

    Returns:
        list: The predicted values, possibly constrained to be non-negative.

    Raises:
        ValueError: If any of the arguments are not correctly provided.
        AttributeError: If there is an issue with the model training or prediction.
    """
    # Train the model using the provided training data and other parameters
    model = build_model(
        x_learn, y_learn, is_categorical, bunch_scores, save_model, save_log, path
    )

    # Make predictions using the trained model
    prediction = predict_by_model(model, x_predict, is_positive)

    # Return the predicted values
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
        return build_model(
            x_learn, y_learn, is_categorical, scores, save_model, save_log, path
        )
    except ValueError as e:
        print(f"Configuration error: {e}")
        raise

    return opt


def predict_by_model(model: Any, x_predict: DataFrame, is_positive: bool):
    """
    Predicts target values using the given model and applies optional constraints.

    Parameters:
        model (Any): The trained model object with a `predict` method.
        x_predict (DataFrame): The input data for prediction.
        is_positive (bool): If True, ensures all predicted values are non-negative.

    Returns:
        list: The predicted values, optionally constrained to be non-negative.

    Notes:
        - The function assumes the model has a `predict` method.
        - If `is_positive` is True, all negative predictions are replaced with 0.
    """
    # Predict target values
    y_predict = model.predict(x_predict)

    # Enforce non-negative predictions if required
    y_predict = [0 if y < 0 else y for y in y_predict] if is_positive else y_predict

    # Return predictions
    return y_predict


def parse_config_models(columns: list[str], path: str | Path | TextIO | None = None) -> dict[str, Any]:
    """
    Loads pre-trained models based on a configuration file and a list of column names.

    Parameters:
        columns (list[str]): A list of column names for which models should be loaded.
        path (Optional[str]): The base directory where the configuration file is located.
                              Defaults to the current working directory.

    Returns:
        dict[str, Optional[object]]: A dictionary mapping column names to their respective
                                     loaded model objects or `None` if the model does not exist.

    Raises:
        FileNotFoundError: If the configuration file or models folder is missing.
        KeyError: If the required keys are missing in the configuration file.
        JSONDecodeError: If the configuration file is not a valid JSON.
    """
    # Set default path if not provided
    if not path:
        path = os.getcwd()

    # Define the path to the configuration file
    config_file_path = os.path.join(
        path, "data_imputer", "config", "config_prediction.json"
    )

    # Ensure the configuration file exists
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"Configuration file not found: {config_file_path}")

    # Load the configuration file
    with open(config_file_path, encoding="utf-8") as f:
        config = json.load(f)

    # Ensure the 'folder' key exists in the configuration
    if "folder" not in config:
        raise KeyError("The configuration file is missing the 'folder' key.")

    # Define the path to the models folder
    folder = os.path.join(path, config["folder"])

    # Ensure the models folder exists
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Models folder not found: {folder}")

    # List all existing model files in the folder
    existing_models = os.listdir(folder)

    # Map column names to corresponding model files
    existed_models = {
        column: next((model for model in existing_models if column in model), None)
        for column in columns
    }

    # Load models for each column if the corresponding file exists
    existed_model_objects = {
        column: load(os.path.join(folder, model_file)) if model_file else None
        for column, model_file in existed_models.items()
    }

    return existed_model_objects


def save_logs_object(model: Any, model_name: str, base_path: str | Path | TextIO | None = None) -> None:
    """
    Saves cross-validation results of a model to an Excel file.

    Parameters:
        model (Any): The trained model object with `cv_results_` attribute.
        model_name (str): The name to use for the log file.
        base_path (str): The base directory for logs. Defaults to the current working directory.

    Raises:
        ValueError: If no subfolders are found in the logs directory.
        FileNotFoundError: If the logs directory does not exist.
    """
    if base_path is None:
        base_path = os.getcwd()

    # Path to the logs folder
    folder_path = os.path.join(base_path, "data_imputer", "logs")

    # Check if the folder exists
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The logs directory does not exist: {folder_path}")

    # Find all subdirectories in the logs folder
    folders = glob.glob(os.path.join(folder_path, "*/"))

    if not folders:
        raise ValueError(f"No subfolders found in {folder_path}")

    # Get the most recently created folder
    last_created_folder = max(folders, key=os.path.getmtime)

    # Create the log file path
    log_file = os.path.join(last_created_folder, model_name + ".xlsx")

    # Save cross-validation results to an Excel file
    cv_result = pd.DataFrame(model.cv_results_)
    cv_result.to_excel(log_file, engine="openpyxl")


def save_model_object(model: Any, model_name: str, base_path: str | Path | TextIO | None = None) -> None:
    """
    Saves a trained model object to a file.

    Parameters:
        model (Any): The trained model object to save.
        model_name (str): The name to use for the model file.
        base_path (str): The base directory for models. Defaults to the current working directory.

    Raises:
        ValueError: If no subfolders are found in the models directory.
        FileNotFoundError: If the models directory does not exist.
    """
    if base_path is None:
        base_path = os.getcwd()

    # Path to the fitted models folder
    folder_path = os.path.join(base_path, "data_imputer", "fitted_model")

    # Check if the folder exists
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The models directory does not exist: {folder_path}")

    # Find all subdirectories in the fitted models folder
    folders = glob.glob(os.path.join(folder_path, "*/"))

    if not folders:
        raise ValueError(f"No subfolders found in {folder_path}")

    # Get the most recently created folder
    last_created_folder = max(folders, key=os.path.getmtime)

    # Create the model file path
    model_file = os.path.join(last_created_folder, model_name + "_GBDT.joblib")

    # Save the model object
    dump(model, model_file)
