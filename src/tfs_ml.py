import os
import time
import math
import gc
import sys
import traceback
import logging
from datetime import datetime
from typing import Literal, Generator
from pydantic.types import PositiveFloat

import numpy as np
import pickle
import warnings
from warnings import simplefilter
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from dask import delayed, compute
import dask.dataframe as dd
from dask.distributed import Client
import joblib

from dask_ml.preprocessing import MinMaxScaler, LabelEncoder
from dask_ml.model_selection import GridSearchCV

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    make_scorer,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    root_mean_squared_error,
    PredictionErrorDisplay
)

from tfs_exceptions import *
from tfs_utils import *
from tfs_models import *


simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)

dt_iso = "%Y-%m-%dT%H:%M:%S.%fZ"
dt_format = "%Y-%m-%dT%H"



class TFSPreprocessor:

    def __init__(self, data: dd.DataFrame, road_category: str, target: Literal["traffic_volumes", "average_speed"], client: Client | None):
        self._data: dd.DataFrame = data


    @staticmethod
    def sin_encoder(data: dd.Series | dd.DataFrame, timeframe: int) -> dd.Series | dd.DataFrame:
        """
        The timeframe indicates a number of days.
        Details:
            - The order of the function parameters has a specific reason. Since this function will be used with Dask's map_partition() (which takes a function and its parameters as input), it's important that the first parameter
              of sin_transformer is actually the data where to execute the transformation itself by map_partition()
        """
        # For more information about Dask's map_partition() function: https://docs.dask.org/en/stable/generated/dask.dataframe.DataFrame.map_partitions.html
        return np.sin(data * (2.0 * np.pi / timeframe))


    @staticmethod
    def cos_encoder(data: dd.Series | dd.DataFrame, timeframe: int) -> dd.Series | dd.DataFrame:
        """
        The timeframe indicates a number of days
        Details:
            - The order of the function parameters has a specific reason. Since this function will be used with Dask's map_partition() (which takes a function and its parameters as input), it's important that the first parameter
              of sin_transformer is actually the data where to execute the transformation itself by map_partition()
        """
        # For more information about Dask's map_partition() function: https://docs.dask.org/en/stable/generated/dask.dataframe.DataFrame.map_partitions.html
        return np.cos((data - 1) * (2.0 * np.pi / timeframe))


    @staticmethod
    def arctan2_decoder(sin_val: float, cos_val: float) -> int | float:  # TODO VERIFY IF IT'S ACTUALLY AN INT (IF SO REMOVE | float)
        angle_rad = np.arctan2(sin_val, cos_val)
        return (angle_rad * 360) / (2 * np.pi)


    def preprocess_volumes(self, z_score: bool = True) -> dd.DataFrame:

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



class TFSLearner:
    """
    The base class for other classes which implement machine learning or statistical methods to learn a predict traffic volumes, average speed or other data about traffic.
    Parameters:
        client: a Dask distributed local cluster client to use to distribute processes
    """

    def __init__(self, road_category: str, target: Literal["traffic_volumes", "average_speed"], client: Client | None):
        self._scorer: dict = {
            "r2": make_scorer(r2_score),
            "mean_squared_error": make_scorer(mean_squared_error),
            "root_mean_squared_error": make_scorer(root_mean_squared_error),
            "mean_absolute_error": make_scorer(mean_absolute_error)
        }
        self._client: Client = client
        self._road_category: str = road_category
        self._target: Literal["traffic_volumes", "average_speed"] = target


    # TODO THIS METHOD WILL BE MODIFIED WHEN THE NAMES OF THE TARGET VARIABLES WILL BE STANDARDIZED, POSSIBLY USING ENUMS
    def _get_grid(self, model: str) -> dict[str, dict[str, Any]]:
        """
        Parameters:
            model: the name of the model which we want to collect the grid for

        Returns:
            The model's grid
        """
        return grids[self._target][model]


    def _get_model(self, model: str) -> Any:
        """
        Parameters:
            model: the name of the model which we want to collect the class of

        Returns:
            The model object
        """
        return model_definitions["function"][model]()

    # TODO RENAME "model_name" INTO SOMETHING ELSE IN ALL FUNCTIONS THAT HAVE model_name AS A PARAMETER
    def gridsearch(self, X_train: dd.DataFrame, y_train: dd.DataFrame, model_name: str) -> None: #TODO REMOVE THE model_name PARAMETER. IDEALLY gridsearch() WOULD JUST HAVE X_train AND y_train

        if self._target not in target_data.values():
            raise Exception("Wrong target variable in GridSearchCV executor function")

        grid = self._get_grid()
        model = self._get_model()

        t_start = datetime.now()
        print(f"{model_name} GridSearchCV started at {t_start}\n")

        gridsearch = GridSearchCV(
            model,
            param_grid=grid,
            scoring=self._scorer,
            refit="mean_absolute_error",
            return_train_score=True,
            n_jobs=retrieve_n_ml_cpus(),
            scheduler=self._client,
            cv=TimeSeriesSplit(n_splits=5)  # A time series splitter for cross validation (for time series cross validation) is necessary since there's a relationship between the rows, thus we cannot use classic cross validation which shuffles the data because that would lead to a data leakage and incorrect predictions
        )  # The models_gridsearch_parameters is obtained from the tfs_models file

        with joblib.parallel_backend("dask"):
            gridsearch.fit(X=X_train, y=y_train)

        t_end = datetime.now()
        print(f"{model_name} GridSearchCV finished at {t_end}\n")
        print(f"Time passed: {t_end - t_start}")

        try:
            gridsearch_results = pd.DataFrame(gridsearch.cv_results_)[
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

            print(f"============== {model_name} grid search results ==============\n")
            print(gridsearch_results, "\n")

            # print("GridSearchCV best estimator: ", gridsearch.best_estimator_)
            # print("GridSearchCV best parameters: ", gridsearch.best_params_)
            # print("GridSearchCV best score: ", gridsearch.best_score_)
            # print("GridSearchCV best combination index (in the results dataframe): ", gridsearch.best_index_, "\n", )
            # print(gridsearch.scorer_, "\n")

            self._export_gridsearch_results_test(gridsearch_results, model_name) #TODO TESTING

            #TODO EXPORT TRUE BEST PARAMETERS

            return gridsearch_results

        except KeyError as e:
            raise ScoringNotFoundError(f"\033[91mScoring not found. Parent error: {e}")

        finally:
            gc.collect()


    def _export_gridsearch_results(self, gridsearch_results: pd.DataFrame, model: Any) -> None:

        true_best_params = {model: gridsearch_results["params"].loc[best_params[self._target][model]]} or {}
        true_best_params.update(model_auxiliary_parameters[model]) # This is just to add the classic parameters which are necessary to get both consistent results and maximise the CPU usage to minimize training time. Also, these are the parameters that aren't included in the grid for the grid search algorithm
        true_best_params["best_GridSearchCV_model_index"] = best_params[self._target][model]
        true_best_params["best_GridSearchCV_model_scores"] = gridsearch_results.loc[true_best_params["best_GridSearchCV_model_index"]].to_dict()  # to_dict() is used to convert the resulting series into a dictionary (which is a data type that's serializable by JSON)

        with open(get_models_parameters_folder_path(target=self._target, road_category=self._road_category) + (get_active_ops() + "_" + self._road_category + "_" + model + "_" + "parameters") + ".json", "w", encoding="utf-8") as params_file:
            json.dump(true_best_params, params_file, indent=4)

        return None


    #TODO TESTING FUNCTION
    def _export_gridsearch_results_test(self, gridsearch_results: pd.DataFrame, model: str) -> None:
        gridsearch_results.to_json(f"./ops/{self._road_category}_{model}_gridsearch.json", indent=4)
        return None



    #TODO TO IMPROVE AND SIMPLIFY THIS METHOD
    def fit(self, X_train: dd.DataFrame, y_train: dd.DataFrame, model_name: str) -> None:

        # -------------- Filenames, etc. --------------

        model_filename = get_active_ops() + "_" + self._road_category + "_" + model_name
        model_params_filepath = get_models_parameters_folder_path(self._target, self._road_category) + get_active_ops() + "_" + self._road_category + "_" + model_name + "_" + "parameters" + ".json"

        # -------------- Parameters extraction --------------

        with open(model_params_filepath, "r") as parameters_file:
            parameters = json.load(parameters_file)[model_name]  # Extracting the model parameters

        # -------------- Training --------------

        #TODO model_definitions
        model = model_names_and_class_objects[model_name](**parameters)  # Unpacking the dictionary to set all parameters to instantiate the model's class object

        with joblib.parallel_backend("dask"):
            model.fit(X_train.compute(), y_train.compute())

        print(f"Successfully trained {model_name} with parameters: {parameters}")

        # -------------- Model exporting --------------

        models_folder_path = get_models_folder_path(self._target, self._road_category)

        try:
            joblib.dump(model, models_folder_path + model_filename + ".joblib", protocol=pickle.HIGHEST_PROTOCOL)
            with open(models_folder_path + model_filename + ".pkl", "wb") as ml_pkl_file:
                pickle.dump(model, ml_pkl_file, protocol=pickle.HIGHEST_PROTOCOL)
            return None

        except Exception as e:
            logging.error(traceback.format_exc())
            print(f"\033[91mCouldn't export trained model. Safely exited the program. Error: {e}\033[0m")
            sys.exit(1)


    def _load_model(self, model_name: str) -> Any:
        return joblib.load(get_models_folder_path(self._target, self._road_category) + get_active_ops() + "_" + self._road_category + "_" + model_name + ".joblib")


    def predict(self, X_test: dd.DataFrame, model: Any) -> None:
        with joblib.parallel_backend("dask"):
            return model.predict(X_test.compute())


    @staticmethod
    def evaluate(y_test: dd.DataFrame, y_pred: dd.DataFrame) -> dict[str, Any]:
        """
        Calculates the prediction errors for data that's already been recorded to test the accuracy of one or more models.

        Parameters:
            y_test: the true values of the target variable
            y_pred: the predicted values of the target variable

        Returns:
            A dictionary of errors (positive floats) for each error metric.
        """
        return {"r2": np.round(r2_score(y_true=y_test, y_pred=y_pred), 4),
                "mean_absolute_error": np.round(mean_absolute_error(y_true=y_test, y_pred=y_pred), 4),
                "mean_squared_error": np.round(mean_squared_error(y_true=y_test, y_pred=y_pred), 4),
                "root_mean_squared_error": np.round(root_mean_squared_error(y_true=y_test, y_pred=y_pred), 4)}



class OnePointForecaster:
    """
    self.trp_road_category: to find the right model to predict the data
    """

    def __init__(self, trp_id: str, road_category: str, target: Literal["traffic_volumes", "average_speed"]):
        self._trp_id: str = trp_id
        self._road_category: str = road_category
        self._n_records: int | None = None
        self._target: Literal["traffic_volumes", "average_speed"] = target


    def _get_future_records(self, target_datetime: datetime) -> dd.DataFrame:
        """
        Returns record of the future to predict.

        Parameters:
            target_datetime: the target datetime which we want to predict data for and before

        Returns:
            A dask dataframe of empty records
        """

        last_available_volumes_data_dt = datetime.strptime(read_metainfo_key(keys_map=["traffic_volumes", "end_date_iso"]), dt_iso)
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

        return dd.from_delayed([delayed(pd.DataFrame)(batch) for batch in get_batches(
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
            } for dt in pd.date_range(start=last_available_volumes_data_dt.strftime(dt_format), end=target_datetime.strftime(dt_format), freq="1h")), batch_size=max(1, math.ceil((target_datetime - last_available_volumes_data_dt).days * 0.20)))]).repartition(partition_size="512MB").persist()


    @staticmethod
    def _get_X(data: dd.DataFrame, target_col: str) -> dd.DataFrame:
        n_rows = data.shape[0].compute()
        p_70 = int(n_rows * 0.70)
        return dd.from_delayed(delayed(data.drop(columns=[target_col]).head(p_70)).persist()) # dd.from_delayed documentation: https://docs.dask.org/en/latest/generated/dask.dataframe.from_delayed.html


    @staticmethod
    def _get_y(data: dd.DataFrame, target_col: str) -> dd.DataFrame:
        n_rows = data.shape[0].compute()
        p_70 = int(n_rows * 0.70)
        return dd.from_delayed(delayed(data[[target_col]]).tail(n_rows - p_70)).persist()


    #TODO TO DIVIDE IN TWO PREPROCESSORS (TRAFFIC VOLUMES AND SPEEDS)
    def preprocess(self, target_datetime: datetime) -> dd.DataFrame:
        """
        Parameters:
            target_datetime: the target datetime which the user wants to predict data for

        Returns:
            A dask dataframe containing the dataset that will be used for the predictions
        """
        # Function workflow:
        # 1. Receiving the target datetime to predict as formal parameter of the function we'll:
        # 2. Calculate the number of hours from the last datetime available for the trp which we want to predict the data for and the nth day in the future
        # 3. Once the number of hours to predict has been calculated we'll multiply it by 24, which means that for each hour to predict we'll use 24 hours in the past as reference
        # 4. We'll get exactly n rows from the TRP's individual data (where n = d * 24 and d is the number of days in the future to predict)
        # 5. We'll create n rows (where each row will be one specific hour of the future to predict)
        # 6. Finally, we'll return the new dataset ready to be fed to the model

        target_datetime = target_datetime.strftime(dt_format)
        last_available_volumes_data_dt = datetime.strptime(read_metainfo_key(keys_map=["traffic_volumes", "end_date_iso"]), dt_iso).strftime(dt_format)

        attr = {"volume": np.nan} if self._target == "traffic_volumes" else {"mean_speed": np.nan, "percentile_85": np.nan}

        # Creating a datetime range with datetimes to predict. These will be inserted in the empty rows to be fed to the models for predictions
        rows_to_predict = []
        for dt in pd.date_range(start=last_available_volumes_data_dt, end=target_datetime, freq="1h"):
            dt_str = dt.strftime("%Y-%m-%d")
            rows_to_predict.append({
                **attr,
                "coverage": np.nan,
                "day": dt.strftime("%d"),
                "month": dt.strftime("%m"),
                "year": dt.strftime("%Y"),
                "hour": dt.strftime("%H"),
                "week": dt.strftime("%V"),
                "date": dt_str,
                "trp_id": self._trp_id
            })

        self._n_records = len(rows_to_predict) * 24  # Number of records to collect from the TRP's individual data

        rows_to_predict = dd.from_pandas(pd.DataFrame(rows_to_predict))
        rows_to_predict["day"] = rows_to_predict["day"].astype("int")
        rows_to_predict["month"] = rows_to_predict["month"].astype("int")
        rows_to_predict["year"] = rows_to_predict["year"].astype("int")
        rows_to_predict["hour"] = rows_to_predict["hour"].astype("int")
        rows_to_predict["week"] = rows_to_predict["week"].astype("int")

        rows_to_predict.persist()

        with dask_cluster_client(processes=False) as client:

            if self._target == "traffic_volumes":
                predictions_dataset = dd.concat([dd.read_csv(read_metainfo_key(keys_map=["folder_paths", "data", "traffic_volumes", "subfolders", "clean", "path"]) + self._trp_id + "_volumes_C.csv").tail(self._n_records), rows_to_predict], axis=0)
            elif self._target == "average_speed":
                predictions_dataset = dd.concat([dd.read_csv(read_metainfo_key(keys_map=["folder_paths", "data", "average_speed", "subfolders", "clean", "path"]) + self._trp_id + "_speeds_C.csv").tail(self._n_records), rows_to_predict], axis=0)

            predictions_dataset = predictions_dataset.repartition(partition_size="512MB")
            predictions_dataset = predictions_dataset.reset_index()
            predictions_dataset = predictions_dataset.drop(columns=["index"])

            if self._target == "traffic_volumes":
                predictions_dataset = TFSPreprocessor(data=predictions_dataset, road_category=self._road_category, target=self._target, client=client).preprocess_volumes(z_score=False)
            elif self._target == "average_speed":
                predictions_dataset = TFSPreprocessor(data=predictions_dataset, road_category=self._road_category, target=self._target, client=client).preprocess_speeds(z_score=False)

            #print(predictions_dataset.compute().tail(200))
            return predictions_dataset.persist()


    @staticmethod
    def export_predictions(y_preds: dd.DataFrame, predictions_metadata: dict[Any, Any], predictions_filepath: str, metadata_filepath: str) -> None:
        try:
            with open(metadata_filepath, "w") as m:
                json.dump(predictions_metadata, m, indent=4)
            dd.to_csv(y_preds, predictions_filepath, single_file=True, encoding="utf-8", **{"index": False})
        except Exception as e:
            print(f"Couldn't export data to {predictions_filepath}, error {e}")
        return None










# TODO CREATE THE A2BForecaster CLASS
