from contextlib import contextmanager
from pathlib import Path
import datetime
from datetime import timezone, timedelta
from zoneinfo import ZoneInfo
from typing import Literal, Any, ClassVar
import os
import asyncio
import pandas as pd
import dask.dataframe as dd
from dateutil.relativedelta import relativedelta
from pydantic import BaseModel
from pydantic.types import PositiveInt
from dask.distributed import Client, LocalCluster

from exceptions import WrongSplittingMode, TargetDataNotAvailableError, NoDataError
from db_config import ProjectTables

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


class GlobalDefinitions(BaseModel):
    class Config:
        arbitrary_types_allowed = True
        frozen = True

    VOLUME_INGESTION_FIELDS: ClassVar[list[str]] = ["trp_id", "volume", "coverage", "is_mice", "zoned_dt_iso"]
    MEAN_SPEED_INGESTION_FIELDS: list[str] = ["trp_id", "mean_speed", "percentile_85", "coverage", "is_mice", "zoned_dt_iso"]
    TARGET_DATA: dict[str, str] = {"V": "volume", "MS": "mean_speed"}
    ROAD_CATEGORIES: list[str] = ["E", "R", "F", "K", "P"]
    DEFAULT_MAX_FORECASTING_WINDOW_SIZE: ClassVar[PositiveInt] = 14

    HAS_VOLUME_CHECK: str = "has_volume"
    HAS_MEAN_SPEED_CHECK: str = "has_mean_speed"

    VOLUME: str = "volume"
    MEAN_SPEED: str = "mean_speed"

    DT_ISO_TZ_FORMAT: str = "%Y-%m-%dT%H:%M:%S%z"
    DT_INPUT_FORMAT: str = "%Y-%m-%dT%H"
    NORWEGIAN_UTC_TIME_ZONE: str = "+01:00"
    NORWEGIAN_UTC_TIME_ZONE_TIMEDELTA: timezone = timezone(timedelta(hours=1))

    COVID_YEARS: list[int] = [2020, 2021, 2022]
    ML_CPUS: int = int(os.cpu_count() * 0.75)  # To avoid crashing while executing parallel computing in the GridSearchCV algorithm
    # The value multiplied with the n_cpu values shouldn't be above .80, otherwise processes could crash during execution

    MEAN_SPEED_DIR: ClassVar[Path] = Path("data", MEAN_SPEED)
    MODEL_GRIDS_DIR: Path = Path("data", "model_grids")

    MICE_COLS: list[str] = ["volume", "coverage"]



def check_target(target: str) -> bool:
    if not target in GlobalDefinitions.TARGET_DATA.keys() or not target in GlobalDefinitions.TARGET_DATA.values():
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


class ForecastingToolbox:
    def __init__(self, db_broker_async: Any | None = None, db_broker: Any | None = None):
        self._db_broker_async: Any | None = db_broker_async
        self._db_broker: Any | None = db_broker



    def set_forecasting_horizon(self, forecasting_window_size: PositiveInt = GlobalDefinitions.DEFAULT_MAX_FORECASTING_WINDOW_SIZE) -> None:
        max_forecasting_window_size: int = max(GlobalDefinitions.DEFAULT_MAX_FORECASTING_WINDOW_SIZE, forecasting_window_size)

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
        print("Maximum settable date: ", last_available_data_dt + relativedelta(last_available_data_dt, days=GlobalDefinitions.DEFAULT_MAX_FORECASTING_WINDOW_SIZE)) #Using cast to avoid type checker warnings

        horizon = datetime.datetime.strptime(input("Insert forecasting horizon (YYYY-MM-DDTHH): "), GlobalDefinitions.DT_INPUT_FORMAT).replace(tzinfo=GlobalDefinitions.NORWEGIAN_UTC_TIME_ZONE_TIMEDELTA).isoformat()

        assert horizon > last_available_data_dt, "Forecasting target datetime is prior to the latest data available"
        assert (horizon - last_available_data_dt).days <= max_forecasting_window_size, f"Number of days to forecast exceeds the limit: {max_forecasting_window_size}"

        self._db_broker.send_sql(f"""UPDATE "{ProjectTables.ForecastingSettings.value}"
                                     SET config = jsonb_set(
                                         config,
                                         '{'{volume_forecasting_horizon}' if option == "V" else '{mean_speed_forecasting_horizon}'}',
                                         to_jsonb('{horizon}'::timestamptz::text),
                                         TRUE
                                     )
                                     WHERE id = TRUE;""") #The horizon datetime value is already in zoned datetime format

        return None


    def get_forecasting_horizon(self, target: str) -> datetime.datetime:
        if not check_target(target):
            raise TargetDataNotAvailableError(f"Wrong target variable: {target}")
        return self._db_broker.send_sql(
            f"""SELECT config -> {'volume_forecasting_horizon' if target == "V" else 'mean_speed_forecasting_horizon'} AS volume_horizon
                FROM "{ProjectTables.ForecastingSettings.value}"
                WHERE id = TRUE;"""
        )[target]


    def reset_forecasting_horizon(self, target: str) -> None:
        if not check_target(target):
            raise TargetDataNotAvailableError(f"Wrong target variable: {target}")
        self._db_broker.send_sql(
            f"""UPDATE "{ProjectTables.ForecastingSettings.value}"
                SET config = jsonb_set(config, '{'volume_forecasting_horizon' if target == "V" else 'mean_speed_forecasting_horizon'}', 'null'::jsonb)
                WHERE id = TRUE;"""
        )
        return None


    async def set_forecasting_horizon_async(self, forecasting_window_size: PositiveInt = GlobalDefinitions.DEFAULT_MAX_FORECASTING_WINDOW_SIZE) -> None:
        """
        Parameters:
            forecasting_window_size: in days, so hours-speaking, let x be the windows size, this will be x*24.
                This parameter is needed since the predictions' confidence varies with how much in the future we want to predict, we'll set a limit on the number of days in future that the user may want to forecast
                This limit is set by default as 14 days, but can be overridden with this parameter

        Returns:
            None
        """
        max_forecasting_window_size: int = max(GlobalDefinitions.DEFAULT_MAX_FORECASTING_WINDOW_SIZE, forecasting_window_size)  # The maximum number of days that can be forecasted is equal to the maximum value between the default window size (14 days) and the maximum window size that can be set through the function parameter

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
        print("Maximum settable date: ", last_available_data_dt + relativedelta(last_available_data_dt, days=GlobalDefinitions.DEFAULT_MAX_FORECASTING_WINDOW_SIZE))

        horizon = datetime.datetime.strptime(input("Insert forecasting horizon (YYYY-MM-DDTHH): " + ":00:00.000" + GlobalDefinitions.NORWEGIAN_UTC_TIME_ZONE), GlobalDefinitions.DT_ISO_TZ_FORMAT)
        # The month number must be zero-padded, for example: 01, 02, etc.

        assert horizon > last_available_data_dt, "Forecasting target datetime is prior to the latest data available, so the data to be forecasted is already available"  # Checking if the imputed date isn't prior to the last one available. So basically we're checking if we already have the data that one would want to forecast
        assert (horizon - last_available_data_dt).days <= max_forecasting_window_size, f"Number of days to forecast exceeds the limit: {max_forecasting_window_size}"  # Checking if the number of days to forecast is less or equal to the maximum number of days that can be forecasted
        # The number of days to forecast
        # Checking if the target datetime isn't ahead of the maximum number of days to forecast

        await self._db_broker_async.send_sql_async(f"""UPDATE "{ProjectTables.ForecastingSettings.value}"
                                                       SET config = jsonb_set(
                                                           config,
                                                           '{'{volume_forecasting_horizon}' if option == "V" else '{mean_speed_forecasting_horizon}'}'
                                                           to_jsonb('{horizon}'::timestamptz::text),
                                                           TRUE
                                                       )
                                                       WHERE id = TRUE;""") #The horizon datetime value is already in zoned datetime format
        #The TRUE after to_jsonb(...) is needed to create the record in case it didn't exist before

        return None


    async def get_forecasting_horizon_async(self, target: str) -> datetime.datetime:
            if not check_target(target):
                raise TargetDataNotAvailableError(f"Wrong target variable: {target}")
            return (await self._db_broker_async.send_sql_async(
                                                f"""SELECT config -> {'volume_forecasting_horizon' if target == "V" else 'mean_speed_forecasting_horizon'} AS volume_horizon
                                                    FROM "{ProjectTables.ForecastingSettings.value}"
                                                    WHERE id = TRUE;"""))[target]


    async def reset_forecasting_horizon_async(self, target: str) -> None:
        if not check_target(target):
            raise TargetDataNotAvailableError(f"Wrong target variable: {target}")
        await self._db_broker_async.send_sql_async(f"""UPDATE "{ProjectTables.ForecastingSettings.value}"
                                                       SET config = jsonb_set(config, '{'volume_forecasting_horizon' if target == "V" else 'mean_speed_forecasting_horizon'}', 'null'::jsonb)
                                                       WHERE id = TRUE;""")
        return None








