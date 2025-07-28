from contextlib import contextmanager
from datetime import datetime
from typing import Literal
from enum import Enum
import os
from functools import wraps
import asyncio
import pandas as pd
import dask.dataframe as dd
import geojson
from dateutil.relativedelta import relativedelta
from pydantic import BaseModel as PydanticBaseModel
from pydantic.types import PositiveInt
from dask.distributed import Client, LocalCluster

from exceptions import WrongSplittingMode, TargetDataNotAvailableError, NoDataError, WrongDBBrokerError
from src.brokers import AIODBBroker, DBBroker

pd.set_option("display.max_columns", None)



@contextmanager
def dask_cluster_client(processes=False):
    """
    - Initializing a client to support parallel backend computing and to be able to visualize the Dask client dashboard
    - Check localhost:8787 to watch real-time processing
    - By default, the number of workers is obtained by dask using the standard os.cpu_count()
    - More information about Dask local clusters here: https://docs.dask.org/en/stable/deploying-python.html
    """
    cluster = LocalCluster(processes=processes)
    client = Client(cluster)
    try:
        yield client
    finally:
        client.close()
        cluster.close()


class BaseModel(PydanticBaseModel):
    class Config:
        arbitrary_types_allowed = True


#TODO USE Pydantic's Base Model IN THE FUTURE
class GlobalDefinitions(Enum):
    TARGET_DATA = {"V": "volume", "MS": "mean_speed"}
    ROAD_CATEGORIES = ["E", "R", "F", "K", "P"]
    DEFAULT_MAX_FORECASTING_WINDOW_SIZE = 14

    HAS_VOLUME_CHECK = "has_volume"
    HAS_MEAN_SPEED_CHECK = "has_mean_speed"

    VOLUME = "volume"
    MEAN_SPEED = "mean_speed"

    NORWEGIAN_UTC_TIME_ZONE = "+01:00"

    COVID_YEARS = [2020, 2021, 2022]
    ML_CPUS = int(os.cpu_count() * 0.75)  # To avoid crashing while executing parallel computing in the GridSearchCV algorithm
    # The value multiplied with the n_cpu values shouldn't be above .80, otherwise processes could crash during execution



def check_target(target: str) -> bool:
    if not target in GlobalDefinitions.TARGET_DATA.value.keys() or not target in GlobalDefinitions.TARGET_DATA.value.values():
        return False
    return True


def ZScore(df: dd.DataFrame, column: str) -> dd.DataFrame:
        df["z_score"] = (df[column] - df[column].mean()) / df[column].std()
        return df[(df["z_score"] > -3) & (df["z_score"] < 3)].drop(columns="z_score").persist()


def split_by_target(data: dd.DataFrame, target: str, mode: Literal[0, 1]) -> tuple[dd.DataFrame, dd.DataFrame, dd.DataFrame, dd.DataFrame] | tuple[dd.DataFrame, dd.DataFrame]:
    """
    Splits the Dask DataFrame into training and testing sets based on the target column and mode.

    Parameters:
        data: dd.DataFrame
        target: str ("volume" or "mean_speed")
        mode: the mode which indicates the kind of split it's intended to execute.
                0 - Stands for the classic 4 section train-test-split (X_train, X_test, y_train, y_test)
                1 - Indicates a forecasted specific train-test-split (X, y)

    Returns:
        X_train, X_test, y_train, y_test
    """

    X = data.drop(columns=[target])
    y = data[[target]]

    if mode == 1:
        return X.persist(), y.persist()
    elif mode == 0:
        n_rows = data.shape[0].compute()
        p_70 = int(n_rows * 0.70)
        return (dd.from_pandas(X.head(p_70)),
                dd.from_pandas(X.tail(n_rows - p_70)),
                dd.from_pandas(y.head(p_70)),
                dd.from_pandas(y.tail(n_rows - p_70)))
    else:
        raise WrongSplittingMode("Wrong splitting mode imputed")


def merge(dfs: list[dd.DataFrame]) -> dd.DataFrame:
    """
    Dask Dataframes merger function
    Parameters:
        dfs: a list of Dask Dataframes to concatenate
    """
    try:
        return (dd.concat(dfs, axis=0)
                .repartition(partition_size="512MB")
                .sort_values(["zoned_dt_iso"], ascending=True)
                .persist())  # Sorting records by date
    except ValueError as e:
        raise NoDataError(f"No data to concatenate. Error: {e}")


def validate_db_broker(db_broker: AIODBBroker | DBBroker):
    def broker_validator(func: callable):
        if asyncio.iscoroutinefunction(func) and db_broker.__class__ is not AIODBBroker or db_broker is None:
            raise WrongDBBrokerError("Wrong DB broker used or not set. Your function is a coroutine and you need to use an asynchronous DB broker like AIODBBroker")
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return broker_validator



class RoadNetworkToolbox(BaseModel):

    def retrieve_edges(self) -> dict:
        with open(f"{self.get('folder_paths.rn_graph.edges.path')}/traffic-nodes-2024_2025-02-28.geojson", "r", encoding="utf-8") as e:
            return geojson.load(e)["features"]


    def retrieve_arches(self) -> dict:
        with open(f"{self.get('folder_paths.rn_graph.arches.path')}/traffic_links_2024_2025-02-27.geojson", "r", encoding="utf-8") as a:
            return geojson.load(a)["features"]



class ForecastingToolbox:
    def __init__(self, db_broker_async: AIODBBroker | None = None, db_broker: DBBroker | None = None):
        self._db_broker_async: AIODBBroker | None = db_broker_async
        self._db_broker: DBBroker | None = db_broker


    def validate_broker_intermediary(self, func: callable):
        if asyncio.iscoroutinefunction(func):
            @validate_db_broker(self._db_broker_async) #TODO SNCE IT VALIDATES wrapper (WHICH IS NOT A COROUTINE) THE ACTUAL PURPOSE TO VALIDATE THE TYPE OF FUNCTION (func, WHICH IS DECORATED BY validate_broker_intermediary) IS NOT SERVED
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper

        @validate_db_broker(self._db_broker)
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper


    @validate_broker_intermediary
    def set_forecasting_horizon(self, forecasting_window_size: PositiveInt = GlobalDefinitions.DEFAULT_MAX_FORECASTING_WINDOW_SIZE.value) -> None:
        max_forecasting_window_size: int = max(GlobalDefinitions.DEFAULT_MAX_FORECASTING_WINDOW_SIZE.value, forecasting_window_size)

        print("V = Volume | MS = Mean Speed")
        option = input("Target: ")
        print("Maximum number of days to forecast: ", max_forecasting_window_size)

        if option == "V":
            last_available_data_dt = self._db_broker.get_volume_date_boundaries()["max"]
        elif option == "MS":
            _, last_available_data_dt = self._db_broker.get_mean_speed_date_boundaries()["max"]
        else:
            raise ValueError("Wrong data option, try again")

        if not last_available_data_dt:
            raise Exception("End date not set. Run download or set it first")

        print("Latest data available: ", last_available_data_dt)
        print("Maximum settable date: ", relativedelta(last_available_data_dt, days=GlobalDefinitions.DEFAULT_MAX_FORECASTING_WINDOW_SIZE.value))

        horizon = input("Insert forecasting horizon (YYYY-MM-DDTHH): ") #TODO CONVERT INTO TIMEZONED DATETIME

        assert horizon > last_available_data_dt, "Forecasting target datetime is prior to the latest data available"
        assert (horizon - last_available_data_dt).days <= max_forecasting_window_size, f"Number of days to forecast exceeds the limit: {max_forecasting_window_size}"

        self._db_broker.send_sql(f"""UPDATE ForecastingSettings
                                     SET config = jsonb_set(
                                         config,
                                         '{'{volume_forecasting_horizon}' if option == "V" else '{mean_speed_forecasting_horizon}'}',
                                         to_jsonb('{horizon}'::timestamptz::text),
                                         TRUE
                                     )
                                     WHERE id = TRUE;""")

        return None

    @validate_broker_intermediary
    def get_forecasting_horizon(self, target: str) -> datetime:
        if not check_target(target):
            raise TargetDataNotAvailableError(f"Wrong target variable: {target}")
        return self._db_broker.send_sql(
            f"""SELECT config -> {'volume_forecasting_horizon' if target == "V" else 'mean_speed_forecasting_horizon'} AS volume_horizon
                FROM ForecastingSettings
                WHERE id = TRUE;"""
        )[target]

    @validate_broker_intermediary
    def reset_forecasting_horizon(self, target: str) -> None:
        if not check_target(target):
            raise TargetDataNotAvailableError(f"Wrong target variable: {target}")
        self._db_broker.send_sql(
            f"""UPDATE ForecastingSettings
                SET config = jsonb_set(config, '{'volume_forecasting_horizon' if target == "V" else 'mean_speed_forecasting_horizon'}', 'null'::jsonb)
                WHERE id = TRUE;"""
        )
        return None

    @validate_broker_intermediary
    async def set_forecasting_horizon_async(self, forecasting_window_size: PositiveInt = GlobalDefinitions.DEFAULT_MAX_FORECASTING_WINDOW_SIZE.value) -> None:
        """
        Parameters:
            forecasting_window_size: in days, so hours-speaking, let x be the windows size, this will be x*24.
                This parameter is needed since the predictions' confidence varies with how much in the future we want to predict, we'll set a limit on the number of days in future that the user may want to forecast
                This limit is set by default as 14 days, but can be overridden with this parameter

        Returns:
            None
        """
        max_forecasting_window_size: int = max(GlobalDefinitions.DEFAULT_MAX_FORECASTING_WINDOW_SIZE.value, forecasting_window_size)  # The maximum number of days that can be forecasted is equal to the maximum value between the default window size (14 days) and the maximum window size that can be set through the function parameter

        print("V = Volume | MS = Mean Speed")
        option = input("Target: ")
        print("Maximum number of days to forecast: ", max_forecasting_window_size)

        if option == "V":
            last_available_data_dt = (await self._db_broker_async.get_volume_date_boundaries_async())["max"]
        elif option == "MS":
            _, last_available_data_dt = (await self._db_broker_async.get_mean_speed_date_boundaries_async())["max"]
        else:
            raise ValueError("Wrong data option, try again")

        if not last_available_data_dt:
            raise Exception("End date not set. Run download or set it first")

        print("Latest data available: ", last_available_data_dt)
        print("Maximum settable date: ", relativedelta(last_available_data_dt, days=GlobalDefinitions.DEFAULT_MAX_FORECASTING_WINDOW_SIZE.value))

        horizon = input("Insert forecasting horizon (YYYY-MM-DDTHH): ")  #TODO CONVERT INTO TIMEZONED DATETIME
        # The month number must be zero-padded, for example: 01, 02, etc. #TODO NOT TO KEEP

        assert horizon > last_available_data_dt, "Forecasting target datetime is prior to the latest data available, so the data to be forecasted is already available"  # Checking if the imputed date isn't prior to the last one available. So basically we're checking if we already have the data that one would want to forecast
        assert (horizon - last_available_data_dt).days <= max_forecasting_window_size, f"Number of days to forecast exceeds the limit: {max_forecasting_window_size}"  # Checking if the number of days to forecast is less or equal to the maximum number of days that can be forecasted
        # The number of days to forecast
        # Checking if the target datetime isn't ahead of the maximum number of days to forecast

        await self._db_broker_async.send_sql_async(f"""UPDATE ForecastingSettings
                                                       SET config = jsonb_set(
                                                           config,
                                                           '{'{volume_forecasting_horizon}' if option == "V" else '{mean_speed_forecasting_horizon}'}'
                                                           to_jsonb('{horizon}'::timestamptz::text),
                                                           TRUE
                                                       )
                                                       WHERE id = TRUE;""")
        #The TRUE after to_jsonb(...) is needed to create the record in case it didn't exist before

        return None

    @validate_broker_intermediary
    async def get_forecasting_horizon_async(self, target: str) -> datetime:
            if not check_target(target):
                raise TargetDataNotAvailableError(f"Wrong target variable: {target}")
            return (await self._db_broker_async.send_sql_async(
                                                f"""SELECT config -> {'volume_forecasting_horizon' if target == "V" else 'mean_speed_forecasting_horizon'} AS volume_horizon
                                                    FROM ForecastingSettings
                                                    WHERE id = TRUE;"""))[target]

    @validate_broker_intermediary
    async def reset_forecasting_horizon_async(self, target: str) -> None:
        if not check_target(target):
            raise TargetDataNotAvailableError(f"Wrong target variable: {target}")
        await self._db_broker_async.send_sql_async(f"""UPDATE ForecastingSettings
                                                       SET config = jsonb_set(config, '{'volume_forecasting_horizon' if target == "V" else 'mean_speed_forecasting_horizon'}', 'null'::jsonb)
                                                       WHERE id = TRUE;""")
        return None








