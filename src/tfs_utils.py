from contextlib import contextmanager
from datetime import datetime
from typing import Any, Literal, Generator, TypedDict
from enum import Enum
from pathlib import Path
import threading
import os
import json
import sys
import asyncio
import aiofiles
from functools import lru_cache
import pandas as pd
import dask.dataframe as dd
from cleantext import clean
import geojson
from dateutil.relativedelta import relativedelta
from pydantic import BaseModel
from pydantic.types import PositiveInt
from async_lru import alru_cache
from dask import delayed
import dask.distributed
from dask.distributed import Client, LocalCluster

from tfs_exceptions import *
from _utils import definitions


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



class GlobalDefinitions(Enum):
    CWD: str = os.getcwd()
    GLOBAL_PROJECTS_DIR: str = "projects"
    GLOBAL_PROJECTS_METADATA: str = "global_metadata.json" # File
    PROJECT_METADATA: str = "project_metadata.json"

    DATA_DIR: str = "data"
    EDA_DIR: str = "eda"
    ML_DIR: str = "ml"
    RN_DIR: str = "rn_graph"

    TARGET_DATA: dict[str, str] = {"V": "volume", "MS": "mean_speed"}

    TRAFFIC_REGISTRATION_POINTS_FILE: str = "traffic_registration_points.json"
    ROAD_CATEGORIES: list[str] = ["E", "R", "F", "K", "P"]
    DEFAULT_MAX_FORECASTING_WINDOW_SIZE: int = 14

    DT_ISO: str = "%Y-%m-%dT%H:%M:%S.%fZ"
    DT_FORMAT: str = "%Y-%m-%dT%H"  # Datetime format, the hour (H) must be zero-padded and 24-h base, for example: 01, 02, ..., 12, 13, 14, 15, etc.



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
    def check_datetime_format(dt: str) -> bool:
        try:
            datetime.strptime(dt, GlobalDefinitions.DT_FORMAT.value)
            return True
        except ValueError:
            return False


    @staticmethod
    def ZScore(df: dd.DataFrame, column: str) -> dd.DataFrame:
        df["z_score"] = (df[column] - df[column].mean()) / df[column].std()
        return df[(df["z_score"] > -3) & (df["z_score"] < 3)].drop(columns="z_score").persist()


    @staticmethod
    def clean_text(text: str) -> str:
        return clean(text, no_emoji=True, no_currency_symbols=True).replace(" ", "_").lower()


    @property
    def covid_years(self) -> list[int]:
        return [2020, 2021, 2022]


    @property
    def ml_cpus(self) -> int:
        return int(os.cpu_count() * 0.75)  # To avoid crashing while executing parallel computing in the GridSearchCV algorithm
        # The value multiplied with the n_cpu values shouldn't be above .80, otherwise processes could crash during execution



class BaseMetadataManager:
    _instances = {} #The class (or subclass) name is the key and the value is the class instance.
    _locks = {} #The class (or subclass) name is the key and the value is the class instance.
    auto_save = True
    """By using a dictionary of instances, a dictionary of locks and the logic in the __new__ dunder method we make any subclass
       a singleton as well, but with a separated instance that doesn't belong to the father class (BaseMetadataManager) one"""

    def __new__(cls, path: str | Path, *args, **kwargs):
        if cls in cls._instances:
            return cls._instances[cls]

        if cls not in cls._locks:
            # Use a class-level lock for each subclass (safely initialize one)
            with threading.Lock():
                if cls not in cls._locks:
                    cls._locks[cls] = threading.Lock()

        # Double-checked locking
        if cls not in cls._instances:
            with cls._locks[cls]:
                if cls not in cls._instances:
                    cls._instances[cls] = super().__new__(cls)

        return cls._instances[cls]


    def _init(self, path: str | Path) -> None:
        self.path = path #Set the metadata path
        self._load() #Load metadata if exists, else set it to a default value (which at the moment is {}, see in _load())
        return None


    def _load(self) -> None:
        try:
            with open(self.path, 'r') as f:
                self.data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.data = {}
        return None


    def reload(self) -> None:
        """Reload metadata from disk."""
        self._load()
        return None


    def save(self) -> None:
        with open(self.path, 'w') as f:
            json.dump(self.data, f, indent=4)
        return None


    @staticmethod
    def _resolve_nested(key: str) -> list[str]:
        """Split a dotted key path into a list of keys."""
        return key.split('.') if isinstance(key, str) else key


    def get(self, key: str, default: Any | None = None) -> Any | None:
        keys = self._resolve_nested(key)
        # data = self.data has a specific reason to exist
        # This is how we preserve the whole original dictionary, but at the same time iterate over its keys and updating them
        # By doing to we'll assign the value (obtained from the value parameter of this method) to the right key, but preserving the rest of the dictionary
        data = self.data
        for k in keys:
            if isinstance(data, dict) and k in data:
                data = data[k]
            else:
                return default
        return data


    def has(self, key: str) -> bool:
        """Check if a nested key exists."""
        keys = self._resolve_nested(key)
        data = self.data
        for k in keys:
            if isinstance(data, dict) and k in data:
                data = data[k]
            else:
                return False
        return True


    def set(self, key: str, value: Any, mode: Literal["e", "a"]) -> None:
        keys = self._resolve_nested(key)
        data = self.data
        if mode == "e":
            for k in keys[:-1]:
                data = data[k]
            data[keys[-1]] = value
            if self.auto_save:
                self.save()
        elif mode == "a":
            for k in keys[:-1]:
                data = data[k]
            data[keys[-1]].append(value)
            if self.auto_save:
                self.save()
        return None


    def delete(self, key: str) -> None:
        keys = self._resolve_nested(key)
        data = self.data
        for k in keys[:-1]:
            if k not in data or not isinstance(data[k], dict):
                return None# Path doesn't exist, nothing to delete
            data = data[k]
        data.pop(keys[-1], None)
        if self.auto_save:
            self.save()
        return None



class GlobalMetadataManager(BaseMetadataManager):
    ...



class ProjectMetadataManager(BaseMetadataManager):
    ...



class TRPMetadataManager(BaseMetadataManager):


    def write_trp_metadata(self, trp_id: str, **kwargs: Any) -> None:
        """
        Writes metadata for a single TRP (Traffic Registration Point).

        Parameters:
            trp_id: an alphanumeric string identifier of the TRP
            **kwargs: parameters which can be added directly into the metadata at write time

        Returns:
             None
        """
        default_settings = {"raw_volumes_file": None, "has_volumes": False, "has_speeds": False, "trp_data": None}
        tracking = {**default_settings, **kwargs}  # Overriding default settings with kwargs
        metadata = {
            "id": trp_id,
            "trp_data": tracking["trp_data"],
            "files": {
                "volumes": {
                    "raw": tracking["raw_volumes_file"],
                    "clean": None
                },
                "speeds": {
                    "raw": None,
                    "clean": None
                }
            },
            "checks": {
                "has_volumes": tracking["has_volumes"],
                "has_speeds": tracking["has_speeds"]
            },
            "data_info": {
                "volumes": {
                    "start_date": None,
                    "end_date": None
                },
                "speeds": {
                    "start_date": None,
                    "end_date": None
                }
            }
        }

        with open(self.get(key="folder_paths.data.trp_metadata.path" + trp_id + "_metadata.json"), "w", encoding="utf-8") as metadata_writer:
            json.dump(metadata, metadata_writer, indent=4)

        return None


    def get_trp_metadata(self, trp_id: str) -> dict[Any, Any]:
        with open(self.get(key="folder_paths.data.trp_metadata.path") + f"{trp_id}_metadata.json", "r", encoding="utf-8") as trp_metadata:
            return json.load(trp_metadata)



class DirectoryManager(BaseModel):
    project_dir: str | Path
    toolbox: GeneralPurposeToolbox
    global_metadata_manager: GlobalMetadataManager
    project_metadata_manager: ProjectMetadataManager


    # ============ PATHS SECTION ============

    @property
    def cwd(self) -> Path:
        return Path.cwd()

    @property
    def global_projects_path(self) -> Path:
        return Path(self.cwd) / self.project_dir

    @property
    def global_metadata_path(self) -> Path:
        return self.global_projects_path / GlobalDefinitions.GLOBAL_PROJECTS_METADATA.value

    @property
    def current_project_path(self) -> Path:
        return self.global_projects_path / self.get_current_project()

    @property
    def current_project_metadata_path(self) -> Path:
        return self.global_projects_path / self.get_current_project() / GlobalDefinitions.PROJECT_METADATA.value

    @property
    def traffic_registration_points_file_path(self) -> Path:
        return self.global_projects_path / self.get_current_project() / GlobalDefinitions.DATA_DIR.value / GlobalDefinitions.TRAFFIC_REGISTRATION_POINTS_FILE.value


    # ============ CURRENT PROJECT SECTION ============

    def set_current_project(self, name: str) -> None:
        self.global_metadata_manager.set(value=self.toolbox.clean_text(name), key="common.current_project", mode="e")
        return None


    @lru_cache()
    def get_current_project(self) -> str | None:
        current_project = self.global_metadata_manager.get(key="common.current_project")
        if not current_project:
            raise ValueError("Current project not set")
        return current_project


    def reset_current_project(self) -> None:
        self.global_metadata_manager.set(value=None, key="common.current_project", mode="e")
        return None


    # ============ GLOBAL PROJECTS DIRECTORY SECTION ============

    def create_global_projects_dir(self) -> None:
        os.makedirs(self.cwd / GlobalDefinitions.GLOBAL_PROJECTS_DIR.value, exist_ok=True)

        #TODO CREATE GLOBAL METADATA FILE

        return None


    def delete_global_projects_dir(self) -> None:
        os.rmdir(self.cwd / GlobalDefinitions.GLOBAL_PROJECTS_DIR.value)
        return None


    # ============ INDIVIDUAL PROJECT SECTION ============

    def create_project(self, name: str):

        #Creating the project's directory
        os.makedirs(self.global_projects_path / self.toolbox.clean_text(name), exist_ok=True)

        self._write_project_metadata(project_dir_name=name)  #Creating the project's metadata file

        folder_structure = {
            "data": {
                **{sub: {
                        "raw": {},
                        "clean": {}
                    } for sub in ("volume", "mean_speed", "travel_times")
                },
                "trp_metadata": {}
                # No subfolders
            },
            "eda": {
                "shapiro_wilk_test": {},
                "plots": {
                    "volume": {},
                    "avg_speeds": {}
                }
            },
            "rn_graph": {
                "edges": {},
                "arches": {},
                "graph_analysis": {},
                "shortest_paths": {}
            },
            "ml": {
                "models_parameters": {
                    **{
                        sub: {target:
                               {
                                   rc: {} for rc in GlobalDefinitions.ROAD_CATEGORIES.value
                               } for target in GlobalDefinitions.TARGET_DATA.value.values()}
                        for sub in ("models_parameters", "models", "models_performance", "ml_reports")
                    },
                },
            }
        }

        metadata_folder_structure = self.project_metadata_manager.get(key="folder_paths")
        metadata_folder_structure = {}  # Setting/resetting the folders path dictionary to either write it for the first time or reset the previous one to adapt it with new updated folders, paths, etc.

        def create_nested_folders(base_path: str, structure: dict[str, dict | None]) -> dict[str, Any]:
            result = {}
            for folder, subfolders in structure.items():
                folder_path = os.path.join(base_path, folder)
                os.makedirs(folder_path, exist_ok=True)
                if isinstance(subfolders, dict) and subfolders:
                    result[folder] = {
                        "path": folder_path,
                        "subfolders": create_nested_folders(folder_path, subfolders)
                    }
                else:
                    result[folder] = {"path": folder_path,
                                      "subfolders": {}}
            return result

        # Creating main directories and respective subdirectories structure
        for key, sub_structure in folder_structure.items():
            main_dir = self.current_project_metadata_path / key
            os.makedirs(main_dir, exist_ok=True)
            metadata_folder_structure[key] = create_nested_folders(str(main_dir), sub_structure)

        self.project_metadata_manager.set(value=metadata_folder_structure, key="folder_paths", mode="e")

        return None


    def _write_project_metadata(self, project_dir_name: str) -> None:
        with open(Path(self.global_projects_path / self.toolbox.clean_text(project_dir_name) / GlobalDefinitions.PROJECT_METADATA.value), "w", encoding="utf-8") as tf:
            json.dump({
            "common": {
                "traffic_registration_points_file": str(Path(self.global_projects_path / self.toolbox.clean_text(project_dir_name) / GlobalDefinitions.DATA_DIR.value / GlobalDefinitions.TRAFFIC_REGISTRATION_POINTS_FILE.value)),
            },
            "volumes": {
                "n_days": None,  # The total number of days which we have data about
                "n_months": None,  # The total number of months which we have data about
                "n_years:": None,  # The total number of years which we have data about
                "n_weeks": None,  # The total number of weeks which we have data about
                "raw_filenames": [],  # The list of raw traffic volumes file names
                "clean_filenames": [],  # The list of clean traffic volumes file names
                "n_rows": [],  # The total number of records downloaded (clean volumes)
                "start_date_iso": None,
                "end_date_iso": None
            },
            "mean_speed": {
                "n_days": None,  # The total number of days which we have data about
                "n_months": None,  # The total number of months which we have data about
                "n_years": None,  # The total number of years which we have data about
                "n_weeks": None,  # The total number of weeks which we have data about
                "raw_filenames": [],  # The list of raw average speed file names
                "clean_filenames": [],  # The list of clean average speed file names
                "n_rows": [],  # The total number of records downloaded (clean average speeds)
                "start_date_iso": None,
                "end_date_iso": None
            },
            "folder_paths": {},
            "forecasting": {"target_datetimes": {"V": None, "AS": None}},
            "trps": {}  # For each TRP we'll have {"id": metadata_filename}
        }, tf, indent=4)
        return None



class TRPToolbox(BaseModel):
    trp_metadata_manager: TRPMetadataManager


    @lru_cache
    def get_global_trp_data(self):
        """
        This function returns json data about all TRPs (downloaded previously)
        """
        with open(self.get(key="common") + GlobalDefinitions.TRAFFIC_REGISTRATION_POINTS_FILE.value, "r", encoding="utf-8") as TRPs:
            return json.load(TRPs)


    #TODO EVALUATE A POSSIBLE CACHING OF THESE AS WELL. BUT KEEP IN MIND POTENTIAL CHANGES DUE TO RE-DOWNLOAD OF TRPS DURING THE SAME EXECUTION OF THE CODE
    def get_trp_ids(self) -> list[str]:
        with open(self.get(key="common" + GlobalDefinitions.TRAFFIC_REGISTRATION_POINTS_FILE.value), "r", encoding="utf-8") as f:
            return list(json.load(f).keys())


    def get_trp_ids_by_road_category(self, target: str) -> dict[str, list[str]] | None:
        road_categories = set(trp["location"]["roadReference"]["roadCategory"]["id"] for trp in self.get_global_trp_data().values())

        clean_data_folder = self.trp_metadata_manager.get(key="folder_paths.data." + target + ".subfolders.clean.path")

        check = "has_volumes" if target == "traffic_volumes" else "has_speeds"  # TODO THIS WILL BE REMOVED WHEN THE TARGET VARIABLE NAME PROBLEM WILL BE SOLVED
        data = "_volumes_C.csv" if target == "traffic_volumes" else "_speeds_C.csv"  # TODO THIS WILL BE REMOVED WHEN THE TARGET VARIABLE NAME PROBLEM WILL BE SOLVED

        return {k: d for k, d in {
            category: [clean_data_folder + trp_id + data for trp_id in
                       filter(lambda trp_id:
                              self.trp_metadata_manager.get_trp_metadata(trp_id)["trp_data"]["location"]["roadReference"]["roadCategory"]["id"] == category and get_trp_metadata(trp_id)["checks"][check], self.get_trp_ids())]
            for category in road_categories
        }.items() if len(d) >= 2}
        # Removing key value pairs from the dictionary where there are less than two dataframes to concatenate, otherwise this would throw an error in the merge() function



class RoadNetworkToolbox(BaseModel):
    global_metadata_manager: GlobalMetadataManager
    project_metadata_manager: ProjectMetadataManager


    def retrieve_edges(self) -> dict:
        with open(f"{self.project_metadata_manager.get('folder_paths.rn_graph.edges.path')}/traffic-nodes-2024_2025-02-28.geojson", "r", encoding="utf-8") as e:
            return geojson.load(e)["features"]


    def retrieve_arches(self) -> dict:
        with open(f"{self.project_metadata_manager.get('folder_paths.rn_graph.arches.path')}/traffic_links_2024_2025-02-27.geojson", "r", encoding="utf-8") as a:
            return geojson.load(a)["features"]



class ForecastingToolbox(BaseModel):
    toolbox: GeneralPurposeToolbox
    global_metadata_manager: GlobalMetadataManager
    trp_metadata_manager: TRPMetadataManager


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
            if (data := self.trp_metadata_manager.get_trp_metadata(trp_id=trp_id))["checks"]["has_speeds"]
        ), strict=True)
        return min(dt_start), max(dt_end)


    def set_forecasting_target_datetime(self, forecasting_window_size: PositiveInt = GlobalDefinitions.DEFAULT_MAX_FORECASTING_WINDOW_SIZE.value) -> None:
        """
        Parameters:
            forecasting_window_size: in days, so hours-speaking, let x be the windows size, this will be x*24.
                This parameter is needed since the predictions' confidence varies with how much in the future we want to predict, we'll set a limit on the number of days in future that the user may want to forecast
                This limit is set by default as 14 days, but can be overridden with this parameter

        Returns:
            None
        """
        max_forecasting_window_size: int = max(int(GlobalDefinitions.DEFAULT_MAX_FORECASTING_WINDOW_SIZE.value), forecasting_window_size)  # The maximum number of days that can be forecasted is equal to the maximum value between the default window size (14 days) and the maximum window size that can be set through the function parameter

        option = input("Press V to set forecasting target datetime for traffic volumes or AS for average speeds: ")
        print("Maximum number of days to forecast: ", max_forecasting_window_size)

        if option == GlobalDefinitions.TARGET_DATA.value["V"]:
            last_available_data_dt = self.global_metadata_manager.get(key="traffic_volumes.end_date_iso")
        elif option == GlobalDefinitions.TARGET_DATA.value["MS"]:
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

        if self.toolbox.check_datetime_format(dt) and option in GlobalDefinitions.TARGET_DATA.value.keys():
            self.global_metadata_manager.set(value=dt, key="forecasting.target_datetimes" + option, mode="e")
            print("Target datetime set to: ", dt, "\n\n")
            return None
        else:
            if not self.toolbox.check_datetime_format(dt):
                raise ValueError("Wrong datetime format, try again")
            elif option not in list(GlobalDefinitions.TARGET_DATA.value.keys()):
                raise ValueError("Wrong target data option, try again")


    def get_forecasting_target_datetime(self, target: str) -> datetime:
        try:
            return datetime.strptime(self.project_metadata_manager.get(key="forecasting.target_datetimes" + target), GlobalDefinitions.DT_FORMAT.value)
        except TypeError:
            raise Exception(f"\033[91mTarget datetime for {target} isn't set yet. Set it first and then execute a one-point forecast\033[0m")


    def reset_forecasting_target_datetime(self) -> None:
        try:
            self.project_metadata_manager.set(value=None, key="forecasting.target_datetimes" + input("Press V to reset forecasting target datetime for traffic volumes or MS for average speeds:"), mode="e")
            print("Target datetime reset successfully\n\n")
            return None
        except KeyError:
            raise KeyError("Target datetime not found")








def update_trp_metadata(trp_id: str, value: Any, mode: Literal["e", "a"]) -> None:
    metadata_filepath = self.trp_metadata_manager.set(value=value, key="folder_paths.data.trp_metadata.path" + trp_id + "_metadata.json", mode=mode)
    return None


async def update_trp_metadata_async(trp_id: str, value: Any, metadata_keys_map: list[str], mode: str) -> None:
    """
        Update TRP metadata asynchronously

        Parameters:
            trp_id: Traffic Registration Point ID
            value: Value to update or append
            metadata_keys_map: List of keys to navigate the JSON structure
            mode: 'equals' to set value or 'append' to add to a list

        Returns:
            None
        """
    modes = ["equals", "append"]
    metadata_filepath = read_metainfo_key(keys_map=["folder_paths", "data", "trp_metadata", "path"]) + trp_id + "_metadata.json"

    async with metadata_lock:
        async with aiofiles.open(metadata_filepath, "r", encoding="utf-8") as m:
            payload = json.loads(await m.read())

        metadata = payload

        if mode == "equals":
            for key in metadata_keys_map[:-1]:
                metadata = metadata[key]
            metadata[metadata_keys_map[-1]] = value  # Updating the metainfo file key-value pair
            async with aiofiles.open(metadata_filepath, "w") as f:
                await f.write(json.dumps(payload, indent=4))
        elif mode == "append":
            for key in metadata_keys_map[:-1]:
                metadata = metadata[key]
            metadata[metadata_keys_map[-1]].append(value)  # Appending a new value to the list (which is the value of this key-value pair)
            async with aiofiles.open(metadata_filepath, "w") as f:
                await f.write(json.dumps(payload, indent=4))
        elif mode not in modes:
            print("\033[91mWrong mode\033[0m")
            sys.exit(1)

    return None



@alru_cache()
async def get_active_ops_async() -> str:
    active_ops = await read_metainfo_key_async(keys_map=["common", "active_operation"])
    if not active_ops:
        print("\033[91mActive operation not set\033[0m")
        sys.exit(1)
    return active_ops




# ==================== TRP Utilities ====================



@alru_cache()
async def import_TRPs_data_async():
    """
    Asynchronously returns json data about all TRPs (downloaded previously)
    """
    async with aiofiles.open(await read_metainfo_key_async(keys_map=["common", "traffic_registration_points_file"]), "r", encoding="utf-8") as TRPs:
        return json.loads(await TRPs.read())



async def update_metainfo_async(value: Any, keys_map: list, mode: str) -> None:
    """
    This function is the asynchronous version of the update_metainfo() one. It inserts data into a specific right key-value pair in the metainfo.json file of the active operation.

    Parameters:
        value: the value which we want to insert or append for a specific key-value pair
        keys_map: the list which includes all the keys which bring to the key-value pair to update or to append another value to (the last key value pair has to be included).
                  The elements in the list must be ordered in which the keys are located in the metainfo dictionary
        mode: the mode which we intend to use for a specific operation on the metainfo file. For example: we may want to set a value for a specific key, or we may want to append another value to a list (which is the value of a specific key-value pair)
    """
    metainfo_filepath = f"{CWD}/{OPS_FOLDER}/{get_active_ops()}/metainfo.json"
    modes = ["equals", "append"]

    async with metainfo_lock:
        if not check_metainfo():
            raise FileNotFoundError(f'Metainfo file for "{get_active_ops()}" operation not found')

        async with aiofiles.open(metainfo_filepath, "r") as m:
                payload = json.loads(await m.read())

        # metainfo = payload has a specific reason to exist
        # This is how we preserve the whole original dictionary (loaded from the JSON file), but at the same time iterate over its keys and updating them
        # By doing to we'll assign the value (obtained from the value parameter of this method) to the right key, but preserving the rest of the dictionary
        metainfo = payload

        if mode == "equals":
            for key in keys_map[:-1]:
                metainfo = metainfo[key]
            metainfo[keys_map[-1]] = value  # Updating the metainfo file key-value pair
            async with aiofiles.open(metainfo_filepath, "w") as f:
                await f.write(json.dumps(payload, indent=4))
        elif mode == "append":
            for key in keys_map[:-1]:
                metainfo = metainfo[key]
            metainfo[keys_map[-1]].append(value)  # Appending a new value to the list (which is the value of this key-value pair)
            async with aiofiles.open(metainfo_filepath, "w") as f:
                await f.write(json.dumps(payload, indent=4))
        elif mode not in modes:
            print("\033[91mWrong mode\033[0m")
            sys.exit(1)

    return None




async def read_metainfo_key_async(keys_map: list) -> Any:
    """
    Reads a value from the metainfo file asynchronously

    Parameters:
        keys_map: a list of keys to navigate the JSON structure

    Returns:
        The value at the specified location in the JSON structure
    """
    payload = await load_metainfo_payload_async()
    for key in keys_map[:-1]:
        payload = payload[key]
    return payload[keys_map[-1]]


# ==================== ML Related Utilities ====================


def get_models_folder_path(target: Literal["traffic_volumes", "average_speed"], road_category: str) -> str:
    return {
        "traffic_volumes": read_metainfo_key(keys_map=["folder_paths", "ml", "models", "subfolders", "traffic_volumes", "subfolders", road_category, "path"]),
        "average_speed": read_metainfo_key(keys_map=["folder_paths", "ml", "models", "subfolders", "average_speed", "subfolders", road_category, "path"])
    }[target]


def get_models_parameters_folder_path(target: Literal["traffic_volumes", "average_speed"], road_category: str) -> str:
    return {
        "traffic_volumes": read_metainfo_key(keys_map=["folder_paths", "ml", "models_parameters", "subfolders", "traffic_volumes", "subfolders", road_category, "path"]),
        "average_speed": read_metainfo_key(keys_map=["folder_paths", "ml", "models_parameters", "subfolders", "average_speed", "subfolders", road_category, "path"])
    }[target]


