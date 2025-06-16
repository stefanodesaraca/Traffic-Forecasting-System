import os
import time
import math
import gc
import sys
import inspect
import pickle
import warnings
from warnings import simplefilter
from scipy import stats
from datetime import datetime
from enum import Enum
from typing import Any, Literal, Generator, Protocol, cast, runtime_checkable
from pydantic import BaseModel as PydanticBaseModel, field_validator
from pydantic.types import PositiveFloat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from dask import delayed, compute
import dask.dataframe as dd
from dask.distributed import Client

from dask_ml.preprocessing import MinMaxScaler, LabelEncoder
from dask_ml.model_selection import GridSearchCV

from sklearn.base import BaseEstimator as ScikitLearnBaseEstimator
from sklearn.model_selection import TimeSeriesSplit

from sklearn.metrics import (
    make_scorer,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    root_mean_squared_error,
    PredictionErrorDisplay
)

from sktime.base import BaseEstimator as SktimeBaseEstimator
from pytorch_forecasting.models.base_model import BaseModel as PyTorchForecastingBaseModel

from tfs_exceptions import *
from tfs_utils import *
from tfs_ml_configs import *


simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)

dt_iso = "%Y-%m-%dT%H:%M:%S.%fZ"
dt_format = "%Y-%m-%dT%H"



class TFSPreprocessor:

    def __init__(self, data: dd.DataFrame, road_category: str, client: Client):
        self._data: dd.DataFrame = data
        self.road_category: str = road_category
        self.client: Client = client


    @property
    def shape(self) -> tuple[int, int]:
        return self._data.shape[0].compute(), self._data.shape[1]


    @staticmethod
    def sin_encoder(data: dd.Series | dd.DataFrame, timeframe: int) -> dd.Series | dd.DataFrame:
        """
        Apply sine transformation for cyclical encoding.

        Parameters
        ----------
        data : dd.Series or dd.DataFrame
            The data where to execute the transformation.
        timeframe : int
            A number of days indicating the timeframe for the cyclical transformation.

        Returns
        -------
        dd.Series or dd.DataFrame
            The sine-transformed data.

        Notes
        -----
        The order of the function parameters has a specific reason. Since this function
        will be used with Dask's map_partition() (which takes a function and its parameters
        as input), it's important that the first parameter of sin_transformer is actually
        the data where to execute the transformation itself by map_partition().

        For more information about Dask's map_partition() function:
        https://docs.dask.org/en/stable/generated/dask.dataframe.DataFrame.map_partitions.html
        """
        return np.sin(data * (2.0 * np.pi / timeframe))


    @staticmethod
    def cos_encoder(data: dd.Series | dd.DataFrame, timeframe: int) -> dd.Series | dd.DataFrame:
        """
        Apply cosine transformation for cyclical encoding.

        Parameters
        ----------
        data : dd.Series or dd.DataFrame
            The data where to execute the transformation.
        timeframe : int
            A number of days indicating the timeframe for the cyclical transformation.

        Returns
        -------
        dd.Series or dd.DataFrame
            The cosine-transformed data.

        Notes
        -----
        The order of the function parameters has a specific reason. Since this function
        will be used with Dask's map_partition() (which takes a function and its parameters
        as input), it's important that the first parameter of sin_transformer is actually
        the data where to execute the transformation itself by map_partition().

        For more information about Dask's map_partition() function:
        https://docs.dask.org/en/stable/generated/dask.dataframe.DataFrame.map_partitions.html
        """
        return np.cos((data - 1) * (2.0 * np.pi / timeframe))


    @staticmethod
    def arctan2_decoder(sin_val: float, cos_val: float) -> int | float:  # TODO VERIFY IF IT'S ACTUALLY AN INT (IF SO REMOVE | float)
        """
        Decode cyclical features using arctan2 function.

        Parameters
        ----------
        sin_val : float
            The sine component value.
        cos_val : float
            The cosine component value.

        Returns
        -------
        int or float
            The decoded angle in degrees.
        """
        angle_rad = np.arctan2(sin_val, cos_val)
        return (angle_rad * 360) / (2 * np.pi)


    def preprocess_volumes(self, z_score: bool = True) -> dd.DataFrame:
        """
        Preprocess traffic volume data for machine learning models.

        Parameters
        ----------
        z_score : bool, default=True
            Whether to apply Z-score outlier filtering.

        Returns
        -------
        dd.DataFrame
            The preprocessed traffic volume dataframe ready for ML models.

        Notes
        -----
        This method performs the following preprocessing steps:
        1. Cyclical variables encoding (hour, week, day, month)
        2. Outliers filtering with Z-Score (optional)
        3. TRP ID Target-Encoding
        4. Variables normalization
        5. Creating lag features (6h, 12h, 24h)
        6. Creating COVID dummy variables
        7. Dropping unnecessary columns
        """

        # ------------------ Cyclical variables encoding ------------------

        self._data["hour_sin"] = self._data["hour"].map_partitions(self.sin_encoder, timeframe=24)
        self._data["hour_cos"] = self._data["hour"].map_partitions(self.cos_encoder, timeframe=24)

        self._data["week_sin"] = self._data["week"].map_partitions(self.sin_encoder, timeframe=52)
        self._data["week_cos"] = self._data["week"].map_partitions(self.cos_encoder, timeframe=52)

        self._data["day_sin"] = self._data["day"].map_partitions(self.sin_encoder, timeframe=31)
        self._data["day_cos"] = self._data["day"].map_partitions(self.cos_encoder, timeframe=31)

        self._data["month_sin"] = self._data["month"].map_partitions(self.sin_encoder, timeframe=12)
        self._data["month_cos"] = self._data["month"].map_partitions(self.cos_encoder, timeframe=12)

        # print("\n\n")

        # ------------------ Outliers filtering with Z-Score ------------------

        if z_score:
            self._data = ZScore(self._data, "volume")

        self._data = self._data.sort_values(by=["date"], ascending=True)

        # ------------------ TRP ID Target-Encoding ------------------

        self._data["trp_id"] = self._data["trp_id"].astype("category")

        encoder = LabelEncoder(use_categorical=True)  # Using a label encoder to encode TRP IDs to include the effect of the non-independence of observations from each other inside the forecasting models
        self._data = self._data.assign(trp_id_encoded=encoder.fit_transform(self._data["trp_id"]))  # The assign methods returns the dataframe obtained as input with the new column (in this case called "trp_id_encoded") added
        self._data.persist()

        # print("Encoded TRP IDs:", sorted(volumes["trp_id_encoded"].unique().compute()))

        # ------------------ Variables normalization ------------------

        scaler = MinMaxScaler()
        self._data[["volume", "coverage"]] = scaler.fit_transform(self._data[["volume", "coverage"]])

        # ------------------ Creating lag features ------------------

        lag6h_column_names = (f"volumes_lag6h_{i}" for i in range(1, 7))
        lag12h_column_names = (f"volumes_lag12h_{i}" for i in range(1, 7))
        lag24h_column_names = (f"volumes_lag24h_{i}" for i in range(1, 7))

        for idx, n in enumerate(lag6h_column_names): self._data[n] = self._data["volume"].shift(idx + 6)  # 6 hours shift
        for idx, n in enumerate(lag12h_column_names): self._data[n] = self._data["volume"].shift(idx + 12)  # 12 hours shift
        for idx, n in enumerate(lag24h_column_names): self._data[n] = self._data["volume"].shift(idx + 24)  # 24 hours shift

        # print(volumes.head(10))
        # print(volumes.dtypes)

        # ------------------ Creating dummy variables to address to the low value for traffic volumes in some years due to covid ------------------

        self._data["is_covid_year"] = (self._data["year"].isin(get_covid_years())).astype("int")  # Creating a dummy variable which indicates if the traffic volume for a record has been affected by covid (because the traffic volume was recorded during one of the covid years)

        # ------------------ Dropping columns which won't be fed to the ML models ------------------

        self._data = self._data.drop(columns=["year", "month", "week", "day", "trp_id", "date"], axis=1).persist()  # Keeping year and hour data and the encoded_trp_id

        # print("Volumes dataframe head: ")
        # print(self._data.head(5), "\n")

        # print("Volumes dataframe tail: ")
        # print(self._data.tail(5), "\n")

        # print(self._data.compute().head(10))

        return self._data #TODO IN THE FUTURE THIS COULD BE DIFFERENT, POSSIBLY IT WON'T RETURN ANYTHING


    def preprocess_speeds(self, z_score: bool = True) -> dd.DataFrame:
        """
        Preprocess average speed data for machine learning models.

        Parameters
        ----------
        z_score : bool, default=True
            Whether to apply Z-score outlier filtering.

        Returns
        -------
        dd.DataFrame
            The preprocessed average speed dataframe ready for ML models.

        Notes
        -----
        This method performs the following preprocessing steps:
        1. Cyclical variables encoding (hour_start, week, day, month)
        2. Outliers filtering with Z-Score (optional)
        3. TRP ID Target-Encoding
        4. Variables normalization
        5. Creating lag features for mean_speed and percentile_85 (6h, 12h, 24h)
        6. Creating COVID dummy variables
        7. Dropping unnecessary columns
        """

        # ------------------ Cyclical variables encoding ------------------

        self._data["hour_start_sin"] = self._data["hour_start"].map_partitions(self.sin_encoder, timeframe=24)
        self._data["hour_start_cos"] = self._data["hour_start"].map_partitions(self.cos_encoder, timeframe=24)

        self._data["week_sin"] = self._data["week"].map_partitions(self.sin_encoder, timeframe=52)
        self._data["week_cos"] = self._data["week"].map_partitions(self.cos_encoder, timeframe=52)

        self._data["day_sin"] = self._data["day"].map_partitions(self.sin_encoder, timeframe=31)
        self._data["day_cos"] = self._data["day"].map_partitions(self.cos_encoder, timeframe=31)

        self._data["month_sin"] = self._data["month"].map_partitions(self.sin_encoder, timeframe=12)
        self._data["month_cos"] = self._data["month"].map_partitions(self.cos_encoder, timeframe=12)

        print("\n\n")

        # ------------------ Outliers filtering with Z-Score ------------------

        if z_score:
            self._data = ZScore(self._data, "mean_speed")

        self._data = self._data.sort_values(by=["date"], ascending=True)

        # ------------------ TRP ID Target-Encoding ------------------

        self._data["trp_id"] = self._data["trp_id"].astype("category")

        encoder = LabelEncoder(use_categorical=True)  # Using a label encoder to encode TRP IDs to include the effect of the non-independence of observations from each other inside the forecasting models
        self._data = self._data.assign(trp_id_encoded=encoder.fit_transform(self._data["trp_id"]))  # The assign methods returns the dataframe obtained as input with the new column (in this case called "trp_id_encoded") added
        self._data.persist()

        # print("Encoded TRP IDs:", sorted(self._data["trp_id_encoded"].unique().compute()))

        # ------------------ Variables normalization ------------------

        scaler = MinMaxScaler()
        self._data[["mean_speed", "percentile_85", "coverage"]] = scaler.fit_transform(self._data[["mean_speed", "percentile_85", "coverage"]])

        # ------------------ Creating lag features ------------------

        lag6h_column_names = (f"mean_speed_lag6h_{i}" for i in range(1, 7))
        lag12h_column_names = (f"mean_speed_lag12_{i}" for i in range(1, 7))
        lag24h_column_names = (f"mean_speed_lag24_{i}" for i in range(1, 7))
        percentile_85_lag6_column_names = (f"percentile_85_lag{i}" for i in range(1, 7))
        percentile_85_lag12_column_names = (f"percentile_85_lag{i}" for i in range(1, 7))
        percentile_85_lag24_column_names = (f"percentile_85_lag{i}" for i in range(1, 7))

        for idx, n in enumerate(lag6h_column_names): self._data[n] = self._data["mean_speed"].shift(idx + 6)  # 6 hours shift
        for idx, n in enumerate(lag12h_column_names): self._data[n] = self._data["mean_speed"].shift(idx + 12)  # 12 hours shift
        for idx, n in enumerate(lag24h_column_names): self._data[n] = self._data["mean_speed"].shift(idx + 24)  # 24 hours shift

        for idx, n in enumerate(percentile_85_lag6_column_names): self._data[n] = self._data["percentile_85"].shift(idx + 6)  # 6 hours shift
        for idx, n in enumerate(percentile_85_lag12_column_names): self._data[n] = self._data["percentile_85"].shift(idx + 12)  # 12 hours shift
        for idx, n in enumerate(percentile_85_lag24_column_names): self._data[n] = self._data["percentile_85"].shift(idx + 24)  # 24 hours shift

        # print(self._data.head(10))
        # print(self._data.dtypes)

        # ------------------ Creating dummy variables to address to the low value for traffic volumes in some years due to covid ------------------

        self._data["is_covid_year"] = self._data["year"].isin(get_covid_years()).astype("int")  # Creating a dummy variable which indicates if the average speed for a record has been affected by covid (because the traffic volume was recorded during one of the covid years)

        # ------------------ Dropping columns which won't be fed to the ML models ------------------

        self._data = self._data.drop(columns=["year", "month", "week", "day", "trp_id", "date"], axis=1).persist()

        # print("Average speeds dataframe head: ")
        # print(self._data.head(5), "\n")

        # print("Average speeds dataframe tail: ")
        # print(self._data.tail(5), "\n")

        return self._data #TODO IN THE FUTURE THIS COULD BE DIFFERENT, POSSIBLY IT WON'T RETURN ANYTHING



class BaseModel(PydanticBaseModel):
    class Config:
        arbitrary_types_allowed = True



class ModelWrapper(BaseModel):
    model_obj: Any
    target: Literal["traffic_volumes", "average_speed"]


    @field_validator("model_obj", mode="after")
    @classmethod
    def validate_model_obj(cls, model_obj) -> Any:
        if not issubclass(model_obj.__class__, ScikitLearnBaseEstimator | PyTorchForecastingBaseModel | SktimeBaseEstimator):
            raise WrongEstimatorTypeError(f"Object passed is not an estimator accepted from this class. Type of the estimator received: {type(model_obj)}")
        return model_obj


    @property
    def name(self) -> str:
        """
        Get the name of the model.

        Returns
        -------
        str
            The class name of the model object.
        """
        return self.model_obj.__class__.__name__


    @property
    def params(self) -> dict[Any, Any]:
        """
        Get the current model's parameters.

        Returns
        -------
        dict[Any, Any]
            The current model's parameters at the time of the call of this method.
        """
        return self.model_obj.get_params()


    @property
    def fit_state(self):
        """
        Get the fitting state of the model.

        Returns
        -------
        bool
            Whether the model has been fitted.
        """
        return self.model_obj.__sklearn_is_fitted__()


    @property
    def grid(self) -> dict[str, Any]:
        """
        Get the model's custom hyperparameter grid.

        Returns
        -------
        dict[str, Any]
            The model's grid for hyperparameter tuning.
        """
        return grids[self.target][self.name]


    def set(self, model_object: Any) -> None:
        """
        Set the model object in the wrapper. Basically just imputing a model object (not an instance) inside the wrapper

        Parameters
        ----------
        model_object : Any
            The model object to wrap.

        Returns
        -------
        None
        """
        setattr(self, "model_obj", model_object)
        return None


    def get(self) -> Any:
        """
        Get the model object.

        Returns
        -------
        Any
            The model object.

        Raises
        ------
        ModelNotSetError
            If no model has been passed to the wrapper class.
        """
        #For future development of this function:
        #When self.model_obj is a scikit-learn model (for example: RandomForestRegressor), using "if not self.model_obj" internally tries to evaluate the object in a boolean context.
        # For scikit-learn models like RandomForestRegressor, this can trigger special methods like __len__ or others
        # that assume the model has already been fitted (which would populate attributes like estimators_), leading to the AttributeError.
        if self.model_obj is None:
            raise ModelNotSetError("Model not passed to the wrapper class")
        return self.model_obj


    def fit(self, X_train: dd.DataFrame, y_train: dd.DataFrame) -> Any:
        """
        Train the model on the input data.

        Parameters
        ----------
        X_train : dd.DataFrame
            Predictor variables' data for training.
        y_train : dd.DataFrame
            The target variable's data for training.

        Returns
        -------
        Any
            The trained model object.
        """

        with joblib.parallel_backend("dask"):
            return self.model_obj.fit(X_train.compute(), y_train.compute())


    def predict(self, X_test: dd.DataFrame) -> Any:
        """
        Make predictions using the trained model.

        Parameters
        ----------
        X_test : dd.DataFrame
            Input features for prediction.

        Returns
        -------
        Any
            Predicted values.

        Raises
        ------
        TypeError
            If the model type is unsupported.

        Notes
        -----
        Currently supports ScikitLearnBaseEstimator. PyTorchForecastingBaseModel
        and SktimeBaseEstimator support is still to be implemented.
        """
        if isinstance(self.model_obj, ScikitLearnBaseEstimator):
            with joblib.parallel_backend("dask"):
                return self.model_obj.predict(X_test.compute()) # type: ignore[attr-defined] # <- WARNING: this comment is used to avoid seeing a useless warning since the model will indeed have a predict method, but the scikit-learn BaseEstimator class doesn't
        elif isinstance(self.model_obj, PyTorchForecastingBaseModel):
            pass #TODO STILL TO IMPLEMENT
        elif isinstance(self.model_obj, SktimeBaseEstimator):
            pass #TODO STILL TO IMPLEMENT
        else:
            raise TypeError(f"Unsupported model type: {type(self.model_obj)}")


    @staticmethod
    def evaluate_regression(y_test: dd.DataFrame, y_pred: dd.DataFrame, scorer: dict[str, callable]) -> dict[str, Any]:
        """
        Calculate prediction errors for regression model testing.

        Parameters
        ----------
        y_test : dd.DataFrame
            The true values of the target variable.
        y_pred : dd.DataFrame
            The predicted values of the target variable.
        scorer : dict[str, callable]
            The scorer which will be used to evaluate the regression.
            The order of the parameters to impute must be y_true first and y_pred second.
            Each scoring function must accept exactly and only the parameters mentioned above.

        Returns
        -------
        dict[str, Any]
            A dictionary of errors (positive floats) for each error metric.
        """
        return {k: np.round(s(y_test, y_pred), decimals=4) for k, s in scorer.items()}


    def export(self, filepath: str) -> None:
        """
        Export the model in joblib and pickle formats.

        Parameters
        ----------
        filepath : str
            The full filepath with filename included without file extension.

        Returns
        -------
        None

        Notes
        -----
        The model will be saved in two formats:
        - .joblib format using joblib.dump()
        - .pkl format using pickle.dump()

        The function will exit the program if export fails.
        """
        try:
            joblib.dump(self.model_obj, filepath + ".joblib", protocol=pickle.HIGHEST_PROTOCOL)
            with open(filepath + ".pkl", "wb") as ml_pkl_file:
                pickle.dump(self.model_obj, ml_pkl_file, protocol=pickle.HIGHEST_PROTOCOL)
            return None
        except Exception as e:
            print(f"\033[91mCouldn't export trained model. Safely exited the program. Error: {e}\033[0m")
            sys.exit(1)



class TFSLearner:
    """
    Base class for models that learn to predict traffic volumes, average speed,
    or other traffic-related data using machine learning or statistical methods.

    Parameters
    ----------
    model : callable (class instance)
        The model **class instance** (not an object). For example:
        `TFSLearner(model=EstimatorClass(), ...)` and **not**
        `TFSLearner(model=EstimatorClass, ...)`.

    road_category : str
        The category of the road where the TRP that recorded the data was located

    target : str
        The target variable to predict (e.g., 'traffic_volumes', 'average_speed').

    client : dask.distributed.Client
        A Dask distributed client used to parallelize computation.
    """

    def __init__(self, model: callable, road_category: str, target: Literal["traffic_volumes", "average_speed"], client: Client | None):
        self._scorer: dict[str, make_scorer] = {
            "r2": make_scorer(r2_score),
            "mean_squared_error": make_scorer(mean_squared_error),
            "root_mean_squared_error": make_scorer(root_mean_squared_error),
            "mean_absolute_error": make_scorer(mean_absolute_error)
        }
        self._client: Client = client
        self._road_category: str = road_category
        self._target: Literal["traffic_volumes", "average_speed"] = target
        self._model: ModelWrapper = ModelWrapper(model_obj=model, target=self._target)


    def get_model(self) -> ModelWrapper:
        return self._model


    def get_scorer(self) -> dict[str, make_scorer]:
        return self._scorer


    def _load_model(self) -> Any:
        """
        Load pre-existing model from its corresponding joblib file.

        Returns
        -------
        Any
            The model object.
        """
        return joblib.load(get_models_folder_path(self._target, self._road_category) + get_active_ops() + "_" + self._road_category + "_" + self._model.name + ".joblib")


    def export_gridsearch_results(self, results: pd.DataFrame) -> None:
        """
        Export GridSearchCV true best results to a JSON file.

        Parameters
        ----------
        results : pd.DataFrame
            The actual gridsearch results as a pandas dataframe.
        Returns
        -------
        None
        """

        true_best_params = {self._model.name: results["params"].loc[best_params[self._target][self._model.name]]} or {}
        true_best_params.update(model_definitions["auxiliary_parameters"][self._model.name]) # This is just to add the classic parameters which are necessary to get both consistent results and maximise the CPU usage to minimize training time. Also, these are the parameters that aren't included in the grid for the grid search algorithm
        true_best_params["best_GridSearchCV_model_scores"] = results.loc[best_params[self._target][self._model.name]].to_dict()  # to_dict() is used to convert the resulting series into a dictionary (which is a data type that's serializable by JSON)

        #TODO FIND A WAY TO LET THE FILE PATH BE CUSTOMIZABLE
        with open(get_models_parameters_folder_path(cast(Literal["traffic_volumes", "average_speed"], self._target), self._road_category) + get_active_ops() + "_" + self._road_category + "_" + self._model.name + "_parameters.json", "w", encoding="utf-8") as params_file:
            json.dump(true_best_params, params_file, indent=4)

        #TODO TESTING:
        results.to_json(f"./ops/{self._road_category}_{self._model.name}_gridsearch.json", indent=4)

        return None


    def gridsearch(self, X_train: dd.DataFrame, y_train: dd.DataFrame) -> pd.DataFrame | None:
        """
        Perform grid search cross-validation for hyperparameter tuning.

        Parameters
        ----------
        X_train : dd.DataFrame
            Predictors' training data.
        y_train : dd.DataFrame
            Target variable's training data.

        Returns
        -------
        None

        Raises
        ------
        TargetVariableNotFoundError
            If wrong target variable is specified.
        ScoringNotFoundError
            If scoring metric is not found.

        Notes
        -----
        Uses TimeSeriesSplit for cross-validation to respect temporal relationships
        in the data. The grid search uses multiple scoring metrics and refits on
        'mean_absolute_error'.
        """

        if self._target not in target_data.values():
            raise TargetVariableNotFoundError("Wrong target variable in GridSearchCV executor function")

        grid = self._model.grid
        model = self._model.get()

        t_start = datetime.now()
        print(f"{self._model.name} GridSearchCV started at {t_start}\n")

        gridsearch = GridSearchCV(
            model,
            param_grid=grid,
            scoring=self._scorer,
            refit="mean_absolute_error",
            return_train_score=True,
            n_jobs=get_ml_cpus(),
            scheduler=self._client,
            cv=TimeSeriesSplit(n_splits=5)  # A time series splitter for cross validation (for time series cross validation) is necessary since there's a relationship between the rows, thus we cannot use classic cross validation which shuffles the data because that would lead to a data leakage and incorrect predictions
        )  # The models_gridsearch_parameters is obtained from the tfs_models file

        with joblib.parallel_backend("dask"):
            gridsearch.fit(X=X_train, y=y_train)

        t_end = datetime.now()
        print(f"{self._model.name} GridSearchCV finished at {t_end}\n")
        print(f"Time passed: {t_end - t_start}")

        try:
            return pd.DataFrame(gridsearch.cv_results_)[
                [
                    "params",
                    "mean_fit_time",
                    "mean_test_r2",
                    "mean_train_r2",
                    "mean_test_mean_squared_error",
                    "mean_train_mean_squared_error",
                    "mean_test_root_mean_squared_error",
                    "mean_train_root_mean_squared_error",
                    "mean_test_mean_absolute_error",
                    "mean_train_mean_absolute_error",
                ]
            ]

        except KeyError as e:
            raise ScoringNotFoundError(f"\033[91mScoring not found. Parent error: {e}")

        finally:
            gc.collect()



class OnePointForecaster:

    def __init__(self, trp_id: str, road_category: str, target: Literal["traffic_volumes", "average_speed"], client: Client):
        self._trp_id: str = trp_id
        self._road_category: str = road_category
        self._target: Literal["traffic_volumes", "average_speed"] = target
        self._client: Client = client


    def get_training_records(self, training_mode: Literal[0, 1], road_category: str, limit: int | None = None) -> dd.DataFrame:
        """
        Parameters:
            training_mode: the training mode we want to use.
                0 - Stands for single-point training, so only the data from the TRP we want to predict future records for is used
                1 - Stands for multi-point training, where data from all TRPs of the same road category as the one we want to predict future records for is used
            road_category: the road category of the TRP which we want to predict data for
            limit: the maximum amount of records to return. The practice adopted is: latest records first. So we'll collect records starting from the latest one to the oldest one.
                   If None, we'll just return all records available.
                   Example: if we had a limit of 2000, we'll only collect the latest 2000 records.
        """
        if training_mode == 0:
            if limit is not None:
                return dd.from_delayed(delayed(dd.read_csv(read_metainfo_key(keys_map=["folder_paths", "data", self._target, "subfolders", "clean", "path"]) + self._trp_id + "_volumes_C.csv").tail(limit).persist()))
            else:
                return dd.read_csv(read_metainfo_key(keys_map=["folder_paths", "data", self._target, "subfolders", "clean", "path"]) + self._trp_id + "_volumes_C.csv")
        elif training_mode == 1:
            if limit is not None:
                return dd.from_delayed(delayed(
                    merge(get_trp_ids_by_road_category(target=self._target)[road_category]).tail(limit).persist()))
            else:
                return merge(get_trp_ids_by_road_category(target=self._target)[road_category])
        else:
            raise WrongTrainRecordsRetrievalMode("training_mode parameter value is not valid")


    def get_future_records(self, target_datetime: datetime) -> dd.DataFrame:
        """
        Generate records of the future to predict.

        Parameters
        ----------
        target_datetime : datetime
            The target datetime which we want to predict data for and before.

        Returns
        -------
        dd.DataFrame
            A dask dataframe of empty records for future predictions.
        """

        attr = {"volume": np.nan} if self._target == "traffic_volumes" else {"mean_speed": np.nan, "percentile_85": np.nan}

        def get_batches(iterable: list[dict[str, Any]] | Generator[dict[str, Any], None, None], batch_size: int) -> Generator[list[dict[str, Any]], Any, None, None]:
            """
            Returns a generator of a maximum of n elements where n is defined by the 'batch_size' parameter.

            Parameters:
                iterable: an iterable to slice into batches
                batch_size: the maximum size of each batch

            Returns:
                A generator of a batch
            """
            iterator = iter(iterable) if isinstance(iterable, list) else iterable
            while True:
                batch = []
                for _ in range(batch_size):
                    try:
                        batch.append(next(iterator))
                    except StopIteration:
                        break
                if not batch:
                    break
                yield batch

        last_available_data_dt = datetime.strptime(read_metainfo_key(keys_map=[self._target, "end_date_iso"]), DT_ISO)
        rows_to_predict = dd.from_delayed([delayed(pd.DataFrame)(batch) for batch in get_batches(
            ({
                **attr,
                "coverage": np.nan,
                "day": dt.strftime("%d"),
                "month": dt.strftime("%m"),
                "year": dt.strftime("%Y"),
                "hour": dt.strftime("%H"),
                "week": dt.strftime("%V"),
                "date": dt.strftime("%Y-%m-%d"),
                "trp_id": self._trp_id
            } for dt in pd.date_range(start=last_available_data_dt.strftime(DT_FORMAT), end=target_datetime.strftime(DT_FORMAT), freq="1h")), batch_size=max(1, math.ceil((target_datetime - last_available_data_dt).days * 0.20)))]).repartition(partition_size="512MB").persist()
            # The start parameter contains the last date for which we have data available, the end one contains the target date for which we want to predict data

        rows_to_predict["day"] = rows_to_predict["day"].astype("int")
        rows_to_predict["month"] = rows_to_predict["month"].astype("int")
        rows_to_predict["year"] = rows_to_predict["year"].astype("int")
        rows_to_predict["hour"] = rows_to_predict["hour"].astype("int")
        rows_to_predict["week"] = rows_to_predict["week"].astype("int")

        if self._target == "traffic_volumes":
            return TFSPreprocessor(data=rows_to_predict, road_category=self._road_category, client=self._client).preprocess_volumes(z_score=False)
        elif self._target == "average_speed":
            return TFSPreprocessor(data=rows_to_predict, road_category=self._road_category, client=self._client).preprocess_speeds(z_score=False)
        else:
            raise TargetVariableNotFoundError("Wrong target variable")


    @staticmethod
    def export_predictions(y_preds: dd.DataFrame, predictions_metadata: dict[Any, Any], predictions_filepath: str, metadata_filepath: str) -> None:
        """
        Export predictions and metadata to files.

        Parameters
        ----------
        y_preds : dd.DataFrame
            The predicted values to export.
        predictions_metadata : dict[Any, Any]
            Metadata associated with the predictions.
        predictions_filepath : str
            File path for saving predictions.
        metadata_filepath : str
            File path for saving metadata.

        Returns
        -------
        None

        Notes
        -----
        Exports predictions as CSV and metadata as JSON. Handles exceptions
        gracefully by printing error messages if export fails.
        """
        try:
            with open(metadata_filepath, "w") as m:
                json.dump(predictions_metadata, m, indent=4)
            dd.to_csv(y_preds, predictions_filepath, single_file=True, encoding="utf-8", **{"index": False})
        except Exception as e:
            print(f"Couldn't export data to {predictions_filepath}, error {e}")
        return None



class TFSReporter:
    ...

















# TODO CREATE THE A2BForecaster CLASS
