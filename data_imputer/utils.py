import os

import geopandas as gpd
import numpy as np
import pandas as pd
from geopandas.geodataframe import GeoDataFrame
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from scipy.spatial import distance
from shapely.geometry import MultiPolygon, Polygon
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

tqdm.pandas()


def calculate_distance_weighted_values(data: GeoDataFrame, nans: dict) -> GeoDataFrame:
    """
    Imputes missing values in a GeoDataFrame based on distance-weighted averages
    of other data points' values.

    Parameters:
        data (GeoDataFrame): The input GeoDataFrame containing geometries and data columns.
        nans (dict): A dictionary where keys are column names and values are lists of indices with NaN.

    Returns:
        GeoDataFrame: A GeoDataFrame with missing values imputed.
    """
    # Extract coordinates from geometry column and convert to list of [x, y] points
    coord = data.geometry.centroid.apply(lambda point: [point.x, point.y]).tolist()

    # Save the geometry column and drop it from the DataFrame for numerical operations
    geom_column = data["geometry"]
    data = data.drop(["geometry"], axis=1)

    # Initialize a list to store distance-weighted values for each row
    initial_vals_list = []

    # Iterate through all rows in the data
    for i in tqdm(range(len(data)), desc="First imputation:"):
        # Calculate the distance matrix between the current point and all others
        matrix_distance = distance.cdist(coord, np.array([coord[i]]), "euclidean")

        # Handle division by zero: replace zeros with a small value to avoid infinite weights
        matrix_distance_div = np.divide(
            1,
            matrix_distance,
            out=np.zeros_like(matrix_distance),
            where=matrix_distance != 0,
        )

        # Calculate distance-weighted sum for the row, ignoring NaNs in the data
        distance_weighted_val = (
            np.nansum(data.to_numpy(na_value=0) * matrix_distance_div, axis=0)
            / matrix_distance_div.sum()
        )

        # Store the computed values for the current row
        initial_vals_list.append(distance_weighted_val.tolist())

    # Create a DataFrame from the calculated initial values
    initial_vals = pd.DataFrame(
        initial_vals_list, columns=data.columns, index=data.index
    )

    # Impute missing values using calculated distance-weighted values
    imputed_initial_vals = data.apply(
        lambda c: (
            c.astype("object").fillna(initial_vals[c.name].loc[nans[c.name]].to_dict())
            if c.name in nans
            else c
        )
    )

    # Add the geometry column back to the imputed DataFrame
    imputed_initial_vals = imputed_initial_vals.join(geom_column).set_geometry(
        "geometry"
    )

    return imputed_initial_vals


def calculate_statistics(data: GeoDataFrame, statistics: str) -> GeoDataFrame:
    """
    Imputes missing values in a GeoDataFrame using a specified statistical method.

    Parameters:
        data (GeoDataFrame): Input GeoDataFrame containing geometry and data columns.
        statistics (str): The name of the statistical method (e.g., 'mean', 'median')
                          to be applied to fill missing values.

    Returns:
        GeoDataFrame: A GeoDataFrame with missing values imputed.
    """
    # Extract the geometry column for later use
    geom_column = data["geometry"]

    # Temporarily remove the geometry column for numerical operations
    data = data.drop(["geometry"], axis=1)

    # Impute missing values column by column
    imputed_initial_vals = data.apply(
        lambda c: c.astype("object").fillna(getattr(c, statistics)() if getattr(c, statistics)() is not np.nan else 0)
    )

    # Recreate the GeoDataFrame by adding the geometry column back
    imputed_initial_vals = gpd.GeoDataFrame(
        imputed_initial_vals.join(geom_column), geometry="geometry"
    )

    return imputed_initial_vals


def set_initial_dtypes(
    dtypes: Series, data_to_round: DataFrame | GeoDataFrame | Series
) -> DataFrame | GeoDataFrame | Series:
    """
    Adjusts the data types of a DataFrame, GeoDataFrame, or Series based on a specified dtype mapping.
    If a column's target type is integer, rounds its values before conversion.

    Parameters:
        dtypes (Series): A pandas Series where the index represents column names and values specify target data types.
        data_to_round (DataFrame | GeoDataFrame | Series): The input data to process.

    Returns:
        DataFrame | GeoDataFrame | Series: The processed data with adjusted data types.
    """
    # Extend the dtype Series by adding entries with "_neigh" suffix
    dt_neigh = pd.concat([dtypes, dtypes.add_suffix("_neigh")])
    cols = dt_neigh.index  # Get all column names from the extended dtype mapping

    # Process if the input is a Series
    if isinstance(data_to_round, Series):
        column_name = data_to_round.name  # Get the column name
        if column_name in cols and "Int" in str(
            dt_neigh[column_name]
        ):  # Check target type
            # Round and convert to the specified type if integer
            rounded_data = data_to_round.round().astype(dt_neigh[column_name])
        else:
            rounded_data = data_to_round  # Leave unchanged if not integer
    else:
        # Process each column in the DataFrame or GeoDataFrame
        rounded_data = data_to_round.apply(
            lambda c: (
                c.round() if c.name in cols and "Int" in str(dt_neigh[c.name]) else c
            )
        )
        # Convert each column to the specified type
        rounded_data = rounded_data.apply(
            lambda c: c.astype(dt_neigh[c.name]) if c.name in cols else c
        )

    return rounded_data


def search_neighbors_from_polygon(loc, df: GeoDataFrame) -> list:
    """
    Find the indices of polygons touching a given polygon in a GeoDataFrame.

    Parameters:
    - loc (Polygon, MultiPolygon): A shapely Polygon object to compare against.
    - df (GeoDataFrame): A GeoDataFrame containing geometries.

    Returns:
    - list: A list of indices where polygons touch the input polygon.
    """

    # Validate input types
    if not isinstance(loc, (Polygon, MultiPolygon)):
        raise ValueError(
            "The 'loc' parameter must be a Shapely Polygon/MultiPolygon object."
        )
    if not isinstance(df, GeoDataFrame):
        raise ValueError("The 'df' parameter must be a GeoDataFrame.")
    if "geometry" not in df.columns:
        raise ValueError("The GeoDataFrame must contain a 'geometry' column.")

    # Vectorized operation using GeoPandas for better performance
    touching_polygons = df[df.geometry.touches(loc)]

    # Return the indices of the touching polygons
    return touching_polygons.index.tolist()


def search_neighbors_from_point(data: GeoDataFrame, n_neigh: int, radius: int) -> GeoDataFrame:
    """
    Search for neighbors from each point in a GeoDataFrame using NearestNeighbors.

    Parameters:
    - data (GeoDataFrame): A GeoPandas GeoDataFrame containing geometries.
    - n_neigh (int): Number of neighbors to find for each point.
    - radius (float): Radius within which neighbors should be searched.

    Returns:
    - GeoDataFrame: The modified GeoDataFrame with added columns:
        - 'dist': Mean distance to the neighbors.
        - 'neighbors': List of indices of the neighbors.
    """

    # Validate input types and values
    if not hasattr(data, "geometry"):
        raise ValueError("Input data must be a GeoDataFrame with a 'geometry' column.")
    if n_neigh <= 0 or radius <= 0:
        raise ValueError("n_neigh and radius must be positive numbers.")
    if len(data) == 0:
        raise ValueError("Input GeoDataFrame is empty.")

    # Extract centroids and their coordinates
    x = data.geometry.centroid.x
    y = data.geometry.centroid.y
    points = pd.DataFrame({"x": x, "y": y})

    # Initialize and fit the NearestNeighbors model
    neigh = NearestNeighbors(n_neighbors=n_neigh + 1, radius=radius)
    neigh.fit(points)

    # Compute the nearest neighbors
    neigh_dist, neigh_ind = neigh.kneighbors(points)

    # Remove the first neighbor (self-reference)
    # Ensure n_neigh is respected even if fewer neighbors are found
    neigh_ind = pd.Series([indices[1:] for indices in neigh_ind.tolist()])
    neigh_dist = np.array([distances[1:] for distances in neigh_dist])

    # Calculate the mean distance to the neighbors
    neigh_dist_mean = pd.Series([d.mean() if len(d) > 0 else np.nan for d in neigh_dist])

    # Reset the index for alignment purposes
    data_index = data.index
    data = data.reset_index(drop=True)

    # Add new columns with results
    data["dist"] = neigh_dist_mean
    data["neighbors"] = neigh_ind

    # Restore original index
    return data.set_index(data_index)


def sort_columns(
    data: DataFrame | GeoDataFrame, sort_type: str, random_state: int = None
) -> DataFrame | GeoDataFrame:
    """
    Sorts the columns of a DataFrame or GeoDataFrame based on a specified sorting type.

    Parameters:
        data (DataFrame | GeoDataFrame): The input data to sort.
        sort_type (str): The sorting criterion. Supported values:
            - "suffering": Randomly shuffles the columns.
            - "nans_ascending": Sorts columns by the count of missing values (ascending).
        random_state (int, optional): Seed for reproducibility when using "suffering".

    Returns:
        DataFrame | GeoDataFrame: A new DataFrame or GeoDataFrame with sorted columns.

    Raises:
        ValueError: If an unsupported `sort_type` is provided.
        TypeError: If `data` is not a DataFrame or GeoDataFrame.
    """
    # Validate the input type
    if not isinstance(data, (DataFrame, GeoDataFrame)):
        raise TypeError("Input data must be a DataFrame or GeoDataFrame.")

    if sort_type == "suffering":
        # Randomly shuffle the columns
        if random_state is not None:
            np.random.seed(random_state)
        shuffled_columns = np.random.permutation(data.columns)
        data = data[shuffled_columns]

    elif sort_type == "nans_ascending":
        # Sort columns by the count of NaN values in ascending order
        nan_counts = data.isna().sum()
        sorted_columns = nan_counts.sort_values(ascending=True).index
        data = data[sorted_columns]

    else:
        # Raise an error for unsupported sort types
        raise ValueError(f"Specified sort type '{sort_type}' is not implemented.")

    return data


def define_nans_positions(data: DataFrame | GeoDataFrame) -> dict[str, pd.Index]:
    """
    Identifies the positions of NaN (missing) values in a DataFrame or GeoDataFrame.

    Parameters:
        data (DataFrame | GeoDataFrame): The input data to analyze.

    Returns:
        dict[str, pd.Index]: A dictionary where:
            - Keys are column names containing NaN values.
            - Values are indices of rows with NaN values in the corresponding column.

    Raises:
        TypeError: If the input `data` is not a DataFrame or GeoDataFrame.
    """
    # Validate input type
    if not isinstance(data, (DataFrame, GeoDataFrame)):
        raise TypeError("Input must be a DataFrame or GeoDataFrame.")

    # Identify columns with NaN values
    imputed_columns = data.columns[data.isna().any()].tolist()

    # Create a dictionary mapping columns to indices of NaN values
    nans_positions = {
        column: data[column][data[column].isna()].index for column in imputed_columns
    }

    return nans_positions


def save_to_file(obj: GeoDataFrame, path: str, filename: str) -> None:
    """
    Saves a GeoDataFrame to a GeoJSON file after processing non-geometric columns.

    Parameters:
        obj (GeoDataFrame): The GeoDataFrame to save.
        path (str): The directory path where the file will be saved.
        filename (str): The name of the output file (without extension).

    Raises:
        ValueError: If the input GeoDataFrame is empty.
        FileNotFoundError: If the specified path does not exist.
        OSError: If there is an issue saving the file.
    """
    # Check if the input GeoDataFrame is empty
    if obj.empty:
        raise ValueError("The input GeoDataFrame is empty and cannot be saved.")

    # Ensure the specified path exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"The specified path does not exist: {path}")

    # Construct the full path to the output file
    full_path = os.path.join(path, f"{filename}.geojson")

    # Process the GeoDataFrame to convert non-geometric columns to float where applicable
    obj_to_save = obj.apply(
        lambda c: (
            pd.Series(
                c.to_numpy(na_value=np.nan, dtype=float), name=c.name, dtype="float"
            )
            if c.dtype.name != "geometry"
            else c
        )
    )

    # Save the processed GeoDataFrame to a GeoJSON file
    try:
        obj_to_save.to_file(full_path, driver="GeoJSON")
    except Exception as e:
        raise IOError(f"Failed to save file: {e}") from e
