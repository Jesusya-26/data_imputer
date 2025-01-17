import json
import os
import warnings
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import TextIO

import geopandas as gpd
import numpy as np
import pandas as pd
from geopandas.geodataframe import GeoDataFrame
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from tqdm import auto, tqdm

from data_imputer import prediction, utils

warnings.filterwarnings("ignore")

# Defining constants for clarity
DISTANCE_WEIGHTED = "distance_weighted"
MEAN = "mean"
MEDIAN = "median"
SUPPORTED_METHODS = {DISTANCE_WEIGHTED, MEAN, MEDIAN}

# Defining constants for folder structure
FITTED_MODEL_DIR = "data_imputer/fitted_model"
LOGS_DIR = "data_imputer/logs"


class DataImputer:
    """
    A class to handle data imputation based on configuration settings.

    Attributes:
    ----------
    cwd : str
        Current working directory. If not provided, defaults to the system's current working directory.
    config_imputation : dict
        Configuration for imputation loaded from a JSON file.
    projection : str
        EPSG projection code for spatial data.
    index_column : str or None
        Name of the index column, if specified in the configuration.
    file_name : str
        Base name of the input data file (without extension).
    file_ext : str
        Extension of the input data file.
    input_data : pd.DataFrame
        The original data loaded from the provided path.
    time_start : str
        Timestamp indicating when the class was initialized.
    data : pd.DataFrame
        A copy of the input data for processing.
    categorical_features : list
        List of categorical features specified in the configuration.
    dtypes : pd.Series
        Data types of all columns except "geometry".
    nans_position : list
        Positions of NaN values in the data.
    imput_counter : int
        Counter tracking the number of imputations performed.
    iter_counter : int
        Counter tracking the number of iterations completed.
    num_iter : int
        Maximum number of iterations allowed.
    save_logs : bool
        Flag to indicate whether logs should be saved.
    save_models : bool
        Flag to indicate whether models should be saved.
    imputed_data : pd.DataFrame or None
        Data after imputation. None if not yet imputed.
    bunch_scores : Any or None
        Scores of imputation performance. None if not calculated.
    mean_score : float or None
        Mean score of imputation performance. None if not calculated.
    """

    def __init__(
        self,
        data_path: str | Path | TextIO,
        cwd: str | Path | TextIO | None = None
    ):
        """
        Initialize the DataImputer instance.

        Parameters:
        ----------
        data_path : str | Path | TextIO
            Path to the input data file (with omissions).
        real_data_path: str | Path | TextIO
            Path to the real data file (without omissions).
        cwd : str | Path | TextIO or None, optional
            Current working directory. If None, the system's current working directory is used.
        """

        # Set working directory
        self.cwd = os.getcwd() if cwd is None else cwd

        # Load configuration file
        config_path = os.path.join(
            self.cwd, "data_imputer", "config", "config_imputation.json"
        )
        try:
            with open(config_path, encoding="utf-8") as f:
                self.config_imputation = json.load(f)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Configuration file not found at: {config_path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON format in configuration file: {config_path}"
            ) from exc

        # Extract configuration settings
        self.projection = self.config_imputation.get("epsg_projection", 4326)
        self.index_column = self.config_imputation.get("index_column", "index")
        self.categorical_features = self.config_imputation.get("categorical_features", [])

        # File name and extension
        self.file_name, self.file_ext = os.path.splitext(os.path.basename(data_path))

        # Load input data
        self.input_data = self.get_data_from_path(data_path)
        self.data = self.input_data.copy()

        # Initialize attributes
        self.time_start = datetime.now().strftime("%d%m%y_%H%M%S")
        self.dtypes = self.data.dtypes.drop("geometry", errors="ignore")
        self.nans_position = utils.define_nans_positions(self.data)

        # Imputation state
        self.imput_counter = 0
        self.iter_counter = 0
        self.num_iter = 0
        self.save_logs = False
        self.save_models = False
        self.imputed_data = None
        self.bunch_scores = None
        self.mean_score = None

    def get_data_from_path(self, data_path: str | Path | TextIO) -> GeoDataFrame:
        """
        Reads geospatial data from the specified file path and processes it based on the configuration.

        Parameters:
        ----------
        data_path : str | Path | TextIO
            Path to the input file containing geospatial data.

        Returns:
        -------
        GeoDataFrame
            A GeoPandas GeoDataFrame with appropriate data types and the specified projection.

        Raises:
        ------
        ValueError
            If the file type is unsupported.
        """
        if self.file_ext == ".geojson":
            # Read and set CRS (Coordinate Reference System)
            data = gpd.read_file(data_path).to_crs(self.projection)
            # Set index column if specified in configuration
            data = data.set_index(self.index_column) if self.index_column else data
        else:
            raise NotImplementedError(
                "Reading from specified file type is not implemented"
            )

        # Adjust data types for compatibility with geojson saving
        data = self.check_dtypes(data)
        return data

    @staticmethod
    def check_dtypes(data: GeoDataFrame) -> GeoDataFrame:
        """
        Adjusts the data types of a GeoDataFrame to ensure compatibility with GeoJSON saving.

        This method:
        - Converts column data types using Pandas' `convert_dtypes()`.
        - Adjusts boolean and non-standard types for file compatibility.

        Parameters:
        ----------
        data : GeoDataFrame
            Input GeoDataFrame whose data types need adjustment.

        Returns:
        -------
        GeoDataFrame
            A GeoDataFrame with adjusted data types.
        """
        # Convert initial types to optimized versions
        data = data.convert_dtypes()

        # Iterate through columns and adjust incompatible data types
        for column, dtype in data.dtypes.items():
            if dtype.name == "boolean":
                data[column] = data[column].astype("Int32")  # Convert boolean to Int32
            elif dtype.name.startswith("string"):
                data[column] = data[column].astype("string")  # Ensure string dtype
            elif dtype.name.startswith("Float"):
                data[column] = data[column].astype("float32")  # Convert Float64 to float32

        return data

    def simulation_omission(
        self, damage_degree: int, selected_columns: list[str] | None = None, save: bool = True
    ) -> None:
        """
        Simulates missing data by randomly omitting values in the specified columns.

        Parameters:
        ----------
        damage_degree : int
            The percentage (0-100) of values to randomly omit (convert to NaN) in the specified columns.
        selected_columns : list, optional
            List of column names to apply the omission simulation. If None, all columns except 'geometry' are used.
            Default is None.
        save : bool, optional
            Whether to save the damaged data to a file. Default is True.

        Returns:
        -------
        None

        Side Effects:
        ------------
        - Updates `self.data` with the simulated missing values.
        - Updates `self.nans_position` to reflect the new NaN positions.
        - Optionally saves the simulated dataset to a file in the specified directory.

        Raises:
        ------
        ValueError
            If `damage_degree` is not between 0 and 100.
        """
        if not (0 <= damage_degree <= 100):
            raise ValueError("damage_degree must be between 0 and 100")

        # Use all columns except geometry if none are specified
        selected_columns = (
            self.data.drop(["geometry"], axis=1).columns.tolist()
            if not selected_columns
            else selected_columns
        )

        damaged_data = self.data.copy()  # Make a copy of the original data

        total_rows = len(damaged_data)
        num_rows_to_damage = int(total_rows * damage_degree / 100)

        # Randomly select rows and columns to damage
        rows_to_damage = np.random.choice(total_rows, size=num_rows_to_damage, replace=False)
        columns_to_damage = np.random.choice(selected_columns, size=num_rows_to_damage, replace=True)

        # Set selected values to NaN
        for row_idx, col_name in zip(rows_to_damage, columns_to_damage):
            damaged_data.loc[row_idx, col_name] = np.nan

        # Update internal attributes
        self.data = damaged_data
        self.nans_position = utils.define_nans_positions(damaged_data)

        # Save the damaged data to file if save flag is True
        if save:
            save_file_name = "_".join([self.file_name, self.time_start])
            utils.save_to_file(
                damaged_data.reset_index(),
                os.path.join(self.cwd, "data_imputer", "simulations"),
                save_file_name,
            )

    def add_neighbors_features(self) -> GeoDataFrame:
        """
        Adds features based on spatial neighbors to the dataset.

        This method calculates new features for each observation by averaging the values of its spatial neighbors.
        The neighbors are determined using one of two search methods:
        - "nearest_from_point": Finds neighbors based on a defined number or radius from a point.
        - "touches_boundary": Identifies neighbors by checking if polygons share boundaries.

        Returns:
        --------
        GeoDataFrame
            Updated GeoDataFrame with new features appended, suffixed with "_neigh".
            These features represent averaged values from neighboring observations.

        Side Effects:
        -------------
        - Updates `self.data` to include new neighbor-based features.
        - Updates `self.categorical_features` to include names of new neighbor-based categorical features.

        Raises:
        ------
        ValueError
            If the specified search method is not supported.
        """
        data = self.data.copy()

        # Determine neighbors based on the specified method
        if self.config_imputation["search_method"] == "nearest_from_point":
            data = utils.search_neighbors_from_point(
                data,
                self.config_imputation["num_neighbors"],
                self.config_imputation["neighbors_radius"],
            )
        elif self.config_imputation["search_method"] == "touches_boundary":
            data["neighbors"] = data.geometry.apply(
                lambda x: utils.search_neighbors_from_polygon(x, data.geometry)
            )
        else:
            raise ValueError(
                "Unsupported search method specified in the configuration."
            )

        # Drop the geometry column for processing
        tqdm.pandas(desc="Search for neighbors:")
        data_no_geom = data.drop(["geometry"], axis=1)

        # Calculate new features based on neighbors
        new_neighbors_features = data_no_geom.progress_apply(
            lambda row: data_no_geom.iloc[row["neighbors"]]
            .drop(["neighbors"], axis=1)
            .mean(),
            axis=1,
        )

        # Fill missing values in new neighbor-based features
        new_neighbors_features = new_neighbors_features.apply(
            lambda col: col.fillna(col.mean())
        )

        # Join new features back to the original data
        self.data = self.data.join(new_neighbors_features, rsuffix="_neigh")

        # Update the list of categorical features
        if len(self.categorical_features) > 0:
            self.categorical_features += [
                feature + "_neigh" for feature in self.categorical_features
            ]

    def multiple_imputation(
        self,
        positive_num: bool = True,
        save_logs: bool = False,
        save_models: bool = False,
    ) -> GeoDataFrame:
        """
        Performs multiple imputations on missing data using a chained iterative algorithm.

        This method handles missing values by iteratively imputing values based on the
        relationships between features. It can perform multiple imputations to account for
        uncertainty and variability in the imputation process. The final imputed dataset is
        saved as a GeoDataFrame.

        Parameters:
        -----------
        positive_num : bool, default=True
            Ensures that imputed numeric values are positive where applicable.
        save_logs : bool, default=False
            If True, saves logs of the imputation process.
        save_models : bool, default=False
            If True, saves intermediate models generated during the imputation process.

        Returns:
        --------
        GeoDataFrame
            A GeoDataFrame with missing values imputed and additional metadata.

        Raises:
        -------
        ValueError
            If the number of imputations (`num_imput`) or iterations (`num_iter`) is less than 1.

        Side Effects:
        -------------
        - Updates `self.imputed_data` with the final imputed dataset.
        - Updates `self.mean_score` with mean imputation scores.
        - Saves the imputed dataset to a file.
        """
        # Extract configuration values
        num_iter = self.config_imputation["num_iteration"]
        num_imput = self.config_imputation["num_imputation"]

        # Validate input parameters
        if num_imput < 1 or num_iter < 1:
            raise ValueError(
                "Number of imputations and iterations must be equal to or greater than 1."
            )

        multiple_imputation = []
        data = self.data.copy()
        self.num_iter = num_iter
        self.bunch_scores = {k: [] for k in self.nans_position.keys()}
        num_imputation = len(self.nans_position.keys()) * num_iter

        # Initial imputation using zero impute
        zero_imputed_data = self.zero_impute(
            self.config_imputation["initial_imputation_type"]
        )
        sort_data = utils.sort_columns(
            zero_imputed_data, self.config_imputation["sort_column_type"]
        )
        sorted_semantic_data = sort_data.drop(["geometry"], axis=1)
        base_semantic_columns = list(self.dtypes.index)

        # Perform multiple imputations
        for i in range(1, num_imput + 1):
            self.iter_counter = 0
            self.imput_counter = i
            progress_bar = tqdm(
                total=num_imputation,
                position=0,
                leave=True,
                desc=f"Iteration of Imputation {i}",
            )
            imputation = self.chained_calculation(
                sorted_semantic_data, progress_bar, positive_num, save_logs, save_models
            )
            imputation = imputation[base_semantic_columns]
            multiple_imputation.append(imputation)

        # Aggregate results from multiple imputations
        imputed_data = (
            pd.concat(multiple_imputation).groupby(level=0).mean()
            if num_imput > 1
            else imputation
        )

        # Ensure consistent data types
        imputed_data = imputed_data.astype(self.dtypes.to_dict())
        imputed_data = imputed_data.join(self.add_flag_columns())

        # Reintegrate geometry and save the result
        imputed_data = gpd.GeoDataFrame(imputed_data.join(data.geometry)).set_crs(
            self.projection
        )
        utils.save_to_file(
            imputed_data.reset_index(),
            self.cwd + "/data_imputer/imputed_data",
            "_".join([self.file_name, self.time_start]),
        )

        # Update object attributes with final results
        self.imputed_data = imputed_data
        self.mean_score = {k: np.mean(v) for k, v in self.bunch_scores.items()}

        return imputed_data

    def zero_impute(self, impute_method: str) -> GeoDataFrame:
        """
        Impute missing values in a GeoDataFrame using the specified method.

        Parameters:
        - impute_method (str): The imputation method to use. Supported methods:
            - "distance_weighted": Fills missing values based on distance-weighted calculations.
            - "mean": Fills missing values using the mean of the dataset.
            - "median": Fills missing values using the median of the dataset.

        Returns:
        - GeoDataFrame: The modified GeoDataFrame with imputed values.

        Raises:
        - ValueError: If an unsupported imputation method is specified.
        """

        # Ensure the imputation method is valid
        if impute_method not in SUPPORTED_METHODS:
            raise NotImplementedError(
                f"Specified impute type '{impute_method}' is not implemented."
            )

        # Perform the appropriate imputation based on the method
        if impute_method == DISTANCE_WEIGHTED:
            # Impute using distance-weighted values
            data_with_initial_vals = utils.calculate_distance_weighted_values(
                self.data, self.nans_position
            )
        else:
            # Impute using statistical methods (mean or median)
            data_with_initial_vals = utils.calculate_statistics(
                self.data, impute_method
            )

        # Ensure the data types are preserved after imputation
        try:
            data_with_initial_vals = utils.set_initial_dtypes(
                self.dtypes, data_with_initial_vals
            )
        except Exception as e:
            raise ValueError(f"Error while setting data types: {e}") from e

        # Return the modified GeoDataFrame
        return data_with_initial_vals

    def chained_calculation(
        self,
        data: DataFrame,
        progress_bar: tqdm,
        positive_num: bool,
        save_logs: bool,
        save_models: bool,
        learn: bool = True,
        models: dict = None,
    ) -> DataFrame:
        """
        Performs a chained iterative calculation to impute missing values.

        This method recursively processes the dataset, applying a specified imputation
        logic to columns with missing values. It leverages a user-defined number of
        iterations to improve imputation quality.

        Parameters:
        -----------
        data : DataFrame
            The input DataFrame containing missing values to be imputed.
        progress_bar : tqdm
            A progress bar instance for tracking iteration progress.
        positive_num : bool
            Ensures that imputed numeric values are positive, where applicable.
        save_logs : bool
            If True, logs of the iterative process are saved for debugging or review.
        save_models : bool
            If True, the trained models used during imputation are saved.
        learn : bool, default=True
            Specifies whether to train models during imputation.
        models : dict, optional
            Pre-trained models to be used for imputation. If None, models will be trained
            during the process.

        Returns:
        --------
        DataFrame
            The DataFrame with missing values imputed after iterative processing.
        """
        # Check if maximum iterations have been reached
        if self.iter_counter < self.num_iter:
            # Increment iteration counter
            self.iter_counter += 1

            # Set save options for logs and models
            self.set_save_options((save_logs, save_models))

            # Apply imputation logic column-wise
            predicted_data = data.apply(
                lambda x: (
                    self.make_iteration(
                        x, data, progress_bar, positive_num, learn, models
                    )
                    if x.name
                    in self.nans_position.keys()  # Apply only to columns with NaNs
                    else x  # Leave other columns unchanged
                )
            )

            # Recursive call for further iterations
            predicted_data = self.chained_calculation(
                predicted_data,
                progress_bar,
                positive_num,
                save_logs,
                save_models,
                learn,
                models,
            )

            return predicted_data

        # Base case: return the data when all iterations are complete
        return data

    def make_iteration(
        self,
        column: Series,
        data: DataFrame,
        progress_bar: tqdm,
        positive_num: bool,
        learn: bool = True,
        models: dict = None,
    ) -> Series:
        """
        Performs a single iteration of imputation for a given column.

        This method trains a predictive model to estimate missing values in the target column
        based on the remaining features in the dataset. If pre-trained models are provided,
        they are used instead of training new models.

        Parameters:
        -----------
        column : Series
            The target column with missing values to be imputed.
        data : DataFrame
            The complete dataset, including the target column and its features.
        progress_bar : tqdm
            A progress bar instance to track the imputation progress.
        positive_num : bool
            Ensures that imputed numeric values are positive, where applicable.
        learn : bool, default=True
            Specifies whether to train a new model for the imputation.
        models : dict, optional
            A dictionary of pre-trained models keyed by column names. If None, models will
            be trained during the process.

        Returns:
        --------
        Series
            The updated column with missing values imputed.
        """
        target_name = column.name
        features = data.drop([target_name], axis=1)

        # If learning mode is enabled, train a model and predict
        if learn:
            is_categorical = target_name in self.categorical_features
            args = [
                is_categorical,
                positive_num,
                self.bunch_scores,
                self.save_logs,
                self.save_models,
            ]

            # Prepare data for training
            y_learn = data[target_name].drop(self.nans_position[target_name])
            x_learn = features.loc[y_learn.index]
            x_predict = features.loc[self.nans_position[target_name]]

            # Train model and predict missing values
            y_predict = prediction.learn_and_predict(
                x_learn, y_learn, x_predict, *args, path=self.cwd
            )
        else:
            # Use pre-trained model for prediction
            x_predict = features.loc[self.nans_position[target_name]]
            y_predict = prediction.predict_by_model(
                models[target_name], x_predict, positive_num
            )

        # Update the column with predicted values
        column = column.astype("float64")
        column.update(pd.Series(y_predict, index=self.nans_position[target_name]))

        # Reset column data type to match the original
        predicted_column = utils.set_initial_dtypes(self.dtypes, column)

        # Update progress bar
        progress_bar.update(1)

        return predicted_column

    def set_save_options(self, save_options: tuple) -> None:
        """
        Set options for saving logs and models. If conditions are met,
        create corresponding directories for saving data.

        Parameters:
        - save_options (tuple): A tuple containing two boolean values
          (save_logs, save_models) indicating whether to save logs and models.

        Returns:
        - None

        Raises:
        - ValueError: If the save_options tuple is invalid.
        - OSError: If directory creation fails for any reason.
        """

        # Validate input
        if not (
            isinstance(save_options, tuple)
            and len(save_options) == 2
            and all(isinstance(x, bool) for x in save_options)
        ):
            raise ValueError("save_options must be a tuple of two boolean values.")

        # Set saving options based on iteration check
        self.save_logs, self.save_models = (
            save_options if self.num_iter == self.iter_counter else (False, False)
        )

        # Generate a unique filename using object attributes
        file_name = f"{self.file_name}_{self.time_start}_{self.imput_counter}"

        # Prepare paths for model and log directories
        model_path = os.path.join(self.cwd, FITTED_MODEL_DIR, file_name)
        log_path = os.path.join(self.cwd, LOGS_DIR, file_name)

        # Create directories if the save options are enabled
        if self.save_models:
            os.makedirs(model_path, exist_ok=True)  # Prevent crash if directory exists
        if self.save_logs:
            os.makedirs(log_path, exist_ok=True)  # Prevent crash if directory exists

    def add_flag_columns(self) -> pd.DataFrame:
        """
        Generate a DataFrame with boolean flags indicating which values were imputed.

        This function creates a new DataFrame where each column corresponds to the columns
        in the input data (except 'geometry'), and values are True if the corresponding
        value was imputed and False otherwise.

        Returns:
            pd.DataFrame: A DataFrame containing boolean flags for imputed values.

        Raises:
            KeyError: If 'geometry' column is missing but expected.
        """

        # Ensure the 'geometry' column exists before dropping it
        if "geometry" not in self.input_data.columns:
            raise KeyError("'geometry' column not found in the input data.")

        # Extract columns except 'geometry'
        flag_columns = self.input_data.drop(columns=["geometry"]).columns

        # Create a DataFrame filled with False
        flag_data = pd.DataFrame(
            False, columns=flag_columns, index=self.input_data.index
        )

        # Efficiently update the flags using a loop instead of apply
        for col in flag_columns:
            if col in self.nans_position:
                flag_data.loc[self.nans_position[col], col] = True

        # Rename columns to indicate they are flags for imputed values
        flag_data.columns = [f"{col}_is_imputed" for col in flag_columns]

        return flag_data

    def impute_by_saved_models(self, positive_num: bool = True) -> gpd.GeoDataFrame:
        """
        Perform imputation using pre-trained models specified in the configuration.

        Args:
            positive_num (bool): Whether to ensure positive numbers during imputation.

        Returns:
            gpd.GeoDataFrame: Imputed geospatial data.

        Raises:
            KeyError: If required configuration keys or models are missing.
            ValueError: If data structure is not as expected.
        """
        # Create a copy of the data to avoid modifying the original
        data = self.data.copy()

        # Load number of iterations from config
        try:
            num_iter = self.config_imputation["num_iteration"]
        except KeyError as exc:
            raise KeyError("Configuration is missing 'num_iteration'.") from exc

        self.num_iter = num_iter

        # Parse features with available models
        features_with_models = prediction.parse_config_models(list(data.columns))

        # Filter only the features with models available
        self.nans_position = {
            k: v
            for k, v in self.nans_position.items()
            if features_with_models.get(k) is not None
        }

        # Compute the number of imputation steps required
        num_imputation = len(self.nans_position) * num_iter

        # Initial zero-based imputation
        zero_imputed_data = self.zero_impute(
            self.config_imputation["initial_imputation_type"]
        )

        # Sort columns based on missing values for better efficiency
        sort_data = utils.sort_columns(zero_imputed_data, "nans_ascending")
        sorted_semantic_data = sort_data.drop(columns=["geometry"], errors="ignore")
        base_semantic_columns = list(self.dtypes.index)

        # Initialize a progress bar for visualization
        progress_bar = tqdm(
            total=num_imputation,
            position=0,
            leave=True,
            desc="Iteration of Imputation",
        )

        # Perform chained imputation with models
        imputation = self.chained_calculation(
            sorted_semantic_data,
            progress_bar,
            positive_num,
            save_logs=False,
            save_models=False,
            learn=False,
            models=features_with_models,
        )

        # Ensure column order and data types are restored
        imputation = imputation[base_semantic_columns]
        imputed_data = imputation.astype(self.dtypes.to_dict())

        # Convert back to GeoDataFrame and restore CRS
        imputed_data = gpd.GeoDataFrame(imputed_data.join(data.geometry)).set_crs(
            self.projection
        )

        # Save the imputed data to file
        utils.save_to_file(
            imputed_data.reset_index(),
            f"{self.cwd}/data_imputer/imputed_data",
            f"{self.file_name}_{self.time_start}",
        )

        # Update object attributes with final results
        self.imputed_data = imputed_data

        return imputed_data

    def get_quality_metrics(
        self, classification_metric: Callable, regression_metric: Callable
    ) -> dict:
        """
        Calculate quality metrics for imputed data and save them as a JSON file.

        Parameters:
        - classification_metric (Callable): Function to calculate classification quality metrics.
        - regression_metric (Callable): Function to calculate regression quality metrics.

        Returns:
        - dict: A dictionary containing quality metrics for each feature with missing values.
        """

        # Create a copy of the initial and imputed data without the 'geometry' column
        initial_data = self.input_data.copy().drop(["geometry"], axis=1)
        imputed_data = self.imputed_data.copy().drop(["geometry"], axis=1)

        # Check if the initial data still contains missing values
        if imputed_data.isna().any().any():
            raise ValueError(
                "Initial dataset contains missing values. Quality metrics cannot be calculated."
            )

        # Initialize a dictionary to store the quality metrics for each feature with missing values
        quality_metrics = {}

        # Loop through features with missing values and compute metrics
        for feature, missing_indices in self.nans_position.items():
            if (
                feature not in initial_data.columns
                or feature not in imputed_data.columns
            ):
                raise KeyError(f"Feature '{feature}' not found in the dataset.")

            # Ensure indices are valid
            valid_indices = [
                idx
                for idx in missing_indices
                if idx in initial_data.index and idx in imputed_data.index
            ]

            # Convert values to NumPy arrays for consistent handling
            true_values = np.array(
                initial_data[feature].loc[valid_indices], dtype="float"
            )
            predicted_values = np.array(
                imputed_data[feature].loc[valid_indices], dtype="float"
            )

            # Select the appropriate metric based on feature type
            if feature in self.categorical_features:
                quality_metrics[feature] = classification_metric(
                    true_values, predicted_values
                )
            else:
                quality_metrics[feature] = regression_metric(
                    true_values, predicted_values
                )

        # Construct the file path for saving the results
        path_to_save = os.path.join(
            self.cwd, "data_imputer", "quality_score", f"quality_score_{self.file_name}_{self.time_start}.json"
        )

        # Save the metrics as a JSON file with proper encoding
        try:
            with open(path_to_save, "w", encoding="utf-8") as file:
                json.dump(quality_metrics, file, indent=4)
        except IOError as e:
            raise IOError(f"Failed to save the quality metrics file: {e}") from e

        # Return the calculated quality metrics
        return quality_metrics
