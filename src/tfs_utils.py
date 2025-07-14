from contextlib import contextmanager
from datetime import datetime
from typing import Any, Literal, Generator
from typing_extensions import override
from enum import Enum
from pathlib import Path
import threading
import os
import json
import sys
from threading import Lock
import asyncio
import aiofiles
from functools import lru_cache
import pandas as pd
import dask.dataframe as dd
from cleantext import clean
import geojson
from dateutil.relativedelta import relativedelta
from pydantic import BaseModel as PydanticBaseModel
from pydantic.types import PositiveInt
import async_lru
from async_lru import alru_cache
from dask import delayed
import dask.distributed
from dask.distributed import Client, LocalCluster

from exceptions import TargetVariableNotFoundError, WrongSplittingMode


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



class GlobalDefinitions(Enum):
    CWD = os.getcwd()
    PROJECTS_HUB_DIR_NAME = "projects"
    PROJECTS_HUB_METADATA = "hub_metadata.json" # File
    PROJECT_METADATA = "project_metadata.json"

    DATA_DIR = "data"
    EDA_DIR = "eda"
    ML_DIR = "ml"
    RN_DIR = "rn_graph"

    TARGET_DATA = {"V": "volume", "MS": "mean_speed"}

    TRAFFIC_REGISTRATION_POINTS_FILE = "traffic_registration_points.json"
    ROAD_CATEGORIES = ["E", "R", "F", "K", "P"]
    DEFAULT_MAX_FORECASTING_WINDOW_SIZE = 14

    RAW_VOLUME_FILENAME_ENDING = "_volume"
    RAW_MEAN_SPEED_FILENAME_ENDING = "_mean_speed"

    CLEAN_VOLUME_FILENAME_ENDING= "_volume_C"
    CLEAN_MEAN_SPEED_FILENAME_ENDING= "_mean_speed_C"

    HAS_VOLUME_CHECK = "has_volume"
    HAS_MEAN_SPEED_CHECK = "has_mean_speed"

    VOLUME = "volume"
    MEAN_SPEED = "mean_speed"

    NORWEGIAN_UTC_TIME_ZONE = "+01:00"


class ProjectsHub:
    _instance = None

    #Making this class a singleton
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ProjectsHub, cls).__new__(cls)
        return cls._instance


    def create_project(self, name: str):

        #Creating the project's directory
        Path(self.hub / name, exist_ok=True).mkdir(exist_ok=True)
        self._metadata_manager.set(value=name, key="projects", mode="a+s")

        folder_structure = {
            "eda": {
                "shapiro_wilk_test": {},
                "plots": {
                    GlobalDefinitions.VOLUME.value: {},
                    GlobalDefinitions.MEAN_SPEED.value: {}
                }
            },
            "rn_graph": {
                "edges": {},
                "arches": {},
                "graph_analysis": {},
                "shortest_paths": {}
            },
            "ml": {
                    **{
                        sub: {target:
                               {
                                   rc: {} for rc in GlobalDefinitions.ROAD_CATEGORIES.value
                               } for target in GlobalDefinitions.TARGET_DATA.value.values()}
                        for sub in ("models_parameters", "models", "models_performance", "ml_reports")
                    },
                }
            }

        metadata_folder_structure = {}  # Setting/resetting the folders path dictionary to either write it for the first time or reset the previous one to adapt it with new updated folders, paths, etc.

        def create_nested_dirs(base_path: str, structure: dict[str, dict | None]) -> dict[str, Any]:
            result = {}
            for folder, subfolders in structure.items():
                folder_path = os.path.join(base_path, folder)
                os.makedirs(folder_path, exist_ok=True)
                if isinstance(subfolders, dict) and subfolders:
                    result[folder] = {
                        "path": folder_path + os.sep,
                        "subfolders": create_nested_dirs(folder_path, subfolders)
                    }
                else:
                    result[folder] = {"path": folder_path + os.sep,
                                      "subfolders": {}}
            return result

        # Creating main directories and respective subdirectories structure
        for key, sub_structure in folder_structure.items():
            main_dir = self.hub / name / key
            os.makedirs(main_dir, exist_ok=True)
            metadata_folder_structure[key] = create_nested_dirs(str(main_dir), sub_structure)

        self._write_project_metadata(dir_name=name, **{"metadata_folder_structure": metadata_folder_structure})  #Creating the project's metadata file

        return None

{
    "folder_paths": kwargs.get("metadata_folder_structure", {}),
    "forecasting": {"target_datetimes": {"V": None, "AS": None}},
    "trps": {}  # For each TRP we'll have {"id": metadata_filename}
}

class GeneralPurposeToolbox(BaseModel):

    @staticmethod
    def split_data(data: dd.DataFrame, target: str, mode: Literal[0, 1]) -> tuple[dd.DataFrame, dd.DataFrame, dd.DataFrame, dd.DataFrame] | tuple[dd.DataFrame, dd.DataFrame]:
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

        if target not in GlobalDefinitions.TARGET_DATA.value.values():
            raise TargetVariableNotFoundError("Wrong target variable in the split_data() function. Must be 'volume' or 'mean_speed'.")

        X = data.drop(columns=[target])
        y = data[[target]]

        if mode == 1:
            return X.persist(), y.persist()
        elif mode == 0:
            n_rows = data.shape[0].compute()
            p_70 = int(n_rows * 0.70)
            return dd.from_delayed(delayed(X.head(p_70))), dd.from_delayed(
                delayed(X.tail(n_rows - p_70))), dd.from_delayed(delayed(y.head(p_70))), dd.from_delayed(
                delayed(y.tail(n_rows - p_70)))
        else:
            raise WrongSplittingMode("Wrong splitting mode imputed")


    @staticmethod
    def merge(filepaths: list[str]) -> dd.DataFrame:
        """
        Data merger function for traffic volumes or average speed data
        Parameters:
            filepaths: a list of files to read data from
        """
        try:
            merged_data = dd.concat([dd.read_csv(trp) for trp in filepaths], axis=0)
            merged_data = merged_data.repartition(partition_size="512MB")
            merged_data = merged_data.sort_values(["date"], ascending=True)  # Sorting records by date
            return merged_data.persist()
        except ValueError as e:
            print(f"\033[91mNo data to concatenate. Error: {e}\033[0m")
            sys.exit(1)


    @staticmethod
    def ZScore(df: dd.DataFrame, column: str) -> dd.DataFrame:
        df["z_score"] = (df[column] - df[column].mean()) / df[column].std()
        return df[(df["z_score"] > -3) & (df["z_score"] < 3)].drop(columns="z_score").persist()


    @staticmethod
    def clean_text(text: str) -> str:
        return clean(text, no_emoji=True, no_punct=True, no_emails=True, no_currency_symbols=True, no_urls=True).replace(" ", "_").lower()


    @staticmethod
    def check_target(target: str) -> bool:
        if not target in [GlobalDefinitions.TARGET_DATA.value.keys(), GlobalDefinitions.TARGET_DATA.value.values()]:
            return False
        return True


    @property
    def covid_years(self) -> list[int]:
        return [2020, 2021, 2022]


    @property
    def ml_cpus(self) -> int:
        return int(os.cpu_count() * 0.75)  # To avoid crashing while executing parallel computing in the GridSearchCV algorithm
        # The value multiplied with the n_cpu values shouldn't be above .80, otherwise processes could crash during execution



class TRPToolbox(BaseModel):
    pjh: ProjectsHub
    model_config = {
        "ignored_types": (async_lru._LRUCacheWrapper,)
    }

    def get_trp_ids_by_road_category(self, target: str) -> dict[str, list[str]] | None:
        ...
        #TODO QUERY THE DB FOR THE id COLUMN IN TrafficRegistrationPoints AND GROUP BY ROAD_CATEGORY


class RoadNetworkToolbox(BaseModel):

    def retrieve_edges(self) -> dict:
        with open(f"{self.get('folder_paths.rn_graph.edges.path')}/traffic-nodes-2024_2025-02-28.geojson", "r", encoding="utf-8") as e:
            return geojson.load(e)["features"]


    def retrieve_arches(self) -> dict:
        with open(f"{self.get('folder_paths.rn_graph.arches.path')}/traffic_links_2024_2025-02-27.geojson", "r", encoding="utf-8") as a:
            return geojson.load(a)["features"]

    #TODO LOAD GRAPH DATA INTO DB



class ForecastingToolbox(BaseModel):
    gp_toolbox: GeneralPurposeToolbox

    def _get_speeds_dates(self, trp_ids: list[str] | Generator[str, None, None]) -> tuple[str, str]:
        """
        Extracts and returns the date of the first and last data available from all average speed files.
        Uses a generator of tuples internally so a generator of TRP IDs would be better to maximize performances.

        Parameters:
            trp_ids: a list or a generator of strings which represent IDs of each traffic registration point available

        Returns:
            tuple[str, str] <- The date of the first data available in first position and the one of the latest data available in second position
        """
        dt_start, dt_end = zip(*(
            (data["data_info"]["speeds"]["start_date"], data["data_info"]["speeds"]["end_date"])
            for trp_id in trp_ids
            if (data := self.tmm.get_trp_metadata(trp_id=trp_id))["checks"][GlobalDefinitions.HAS_MEAN_SPEED_CHECK.value]
        ), strict=True)
        return min(dt_start), max(dt_end)


    #TODO SET FORECASTING HORIZON IN PROJECT DB
    def set_forecasting_horizon(self, forecasting_window_size: PositiveInt = GlobalDefinitions.DEFAULT_MAX_FORECASTING_WINDOW_SIZE.value) -> None:
        """
        Parameters:
            forecasting_window_size: in days, so hours-speaking, let x be the windows size, this will be x*24.
                This parameter is needed since the predictions' confidence varies with how much in the future we want to predict, we'll set a limit on the number of days in future that the user may want to forecast
                This limit is set by default as 14 days, but can be overridden with this parameter

        Returns:
            None
        """
        max_forecasting_window_size: int = max(int(GlobalDefinitions.DEFAULT_MAX_FORECASTING_WINDOW_SIZE.value), forecasting_window_size)  # The maximum number of days that can be forecasted is equal to the maximum value between the default window size (14 days) and the maximum window size that can be set through the function parameter

        print("V = Volume | MS = Mean Speed")
        option = input("Target: ")
        print("Maximum number of days to forecast: ", max_forecasting_window_size)

        if option == "V":
            last_available_data_dt = self.tmm.get(key=GlobalDefinitions.VOLUME.value + ".end_date_iso") #TODO tmm HAS TO BE REPLACED
        elif option == "MS":
            _, last_available_data_dt = self._get_speeds_dates(self.get_global_trp_data())
            if last_available_data_dt is None:
                raise Exception("End date not found in metainfo file. Run download first or set it first")

            last_available_data_dt = datetime.strptime(last_available_data_dt, "%Y-%m-%d %H:%M:%S").strftime(GlobalDefinitions.DT_ISO.value)

        else:
            raise ValueError("Wrong data option, try again")

        print("Latest data available: ", datetime.strptime(last_available_data_dt, GlobalDefinitions.DT_ISO.value))
        print("Maximum settable date: ", relativedelta(datetime.strptime(last_available_data_dt, GlobalDefinitions.DT_ISO.value), days=14))

        dt = input("Insert Target Datetime (YYYY-MM-DDTHH): ")  # The month number must be zero-padded, for example: 01, 02, etc.

        assert datetime.strptime(dt, GlobalDefinitions.DT_FORMAT.value) > datetime.strptime(last_available_data_dt, GlobalDefinitions.DT_ISO.value), "Forecasting target datetime is prior to the latest data available, so the data to be forecasted is already available"  # Checking if the imputed date isn't prior to the last one available. So basically we're checking if we already have the data that one would want to forecast
        assert (datetime.strptime(dt, GlobalDefinitions.DT_FORMAT.value) - datetime.strptime(last_available_data_dt, GlobalDefinitions.DT_ISO.value)).days <= max_forecasting_window_size, f"Number of days to forecast exceeds the limit: {max_forecasting_window_size}"  # Checking if the number of days to forecast is less or equal to the maximum number of days that can be forecasted
        # The number of days to forecast
        # Checking if the target datetime isn't ahead of the maximum number of days to forecast

        if self.gp_toolbox.check_datetime_format(dt) and option in GlobalDefinitions.TARGET_DATA.value.keys():
            self.pmm.set(value=dt, key="forecasting.target_datetimes" + option, mode="e")
            print("Target datetime set to: ", dt, "\n\n")
            return None
        else:
            if not self.gp_toolbox.check_datetime_format(dt):
                raise ValueError("Wrong datetime format, try again")
            elif option not in list(GlobalDefinitions.TARGET_DATA.value.keys()):
                raise ValueError("Wrong target data option, try again")

        return None


    def get_forecasting_horizon(self, target: str) -> datetime:
        try:
            return datetime.strptime(self.pmm.get(key="forecasting.target_datetimes" + target), GlobalDefinitions.DT_FORMAT.value)
        except TypeError:
            raise Exception(f"\033[91mTarget datetime for {target} isn't set yet. Set it first and then execute a one-point forecast\033[0m")


    def reset_forecasting_horizon(self, target: str) -> None:
        try:
            self.pmm.set(value=None, key="forecasting.target_datetimes" + target, mode="e")
            print("Target datetime reset successfully\n\n")
            return None
        except KeyError:
            raise KeyError("Target datetime not found")









