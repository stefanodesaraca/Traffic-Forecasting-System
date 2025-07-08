from contextlib import contextmanager
from datetime import datetime
from typing import Any, Literal, Generator
from typing_extensions import override
from enum import Enum
from pathlib import Path
from functools import wraps
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

from tfs_exceptions import TargetVariableNotFoundError, WrongSplittingMode


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

    DT_ISO = "%Y-%m-%dT%H:%M:%S.%fZ"
    DT_FORMAT = "%Y-%m-%dT%H"  # Datetime format, the hour (H) must be zero-padded and 24-h base, for example: 01, 02, ..., 12, 13, 14, 15, etc.

    RAW_VOLUME_FILENAME_ENDING = "_volume"
    RAW_MEAN_SPEED_FILENAME_ENDING = "_mean_speed"

    CLEAN_VOLUME_FILENAME_ENDING= "_volume_C"
    CLEAN_MEAN_SPEED_FILENAME_ENDING= "_mean_speed_C"

    HAS_VOLUME_CHECK = "has_volume"
    HAS_MEAN_SPEED_CHECK = "has_mean_speed"

    VOLUME = "volume"
    MEAN_SPEED = "mean_speed"



class BaseMetadataManager:
    _instances: dict = {} #The class (or subclass) name is the key and the value is the class instance.
    _locks: dict = {} #The class (or subclass) name is the key and the value is the class instance.
    auto_save: bool = True
    path: str | Path | None = None
    """By using a dictionary of instances, a dictionary of locks and the logic in the __new__ dunder method we make any subclass
       a singleton as well, but with a separated instance that doesn't belong to the father class (BaseMetadataManager) one"""

    def __new__(cls, path: str | Path | None = None, *args, **kwargs):
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
                    cls._instances[cls].path = path
                    cls._instances[cls]._init(cls._instances[cls].path)
        return cls._instances[cls]


    def _init(self, path: str | Path) -> None:
        self.path = path #Set the metadata path
        self._load() #Load metadata if exists, else set it to a default value (which at the moment is {}, see in _load())
        return None


    async def _init_async(self, path: str | Path) -> None:
        self.path = path
        await self._load_async()
        return None


    def _load(self) -> None:
        try:
            with open(self.path, 'r', encoding="utf-8") as f:
                self.data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.data = {}
        return None


    async def _load_async(self) -> None:
        try:
            async with aiofiles.open(self.path, 'r', encoding="utf-8") as f:
                self.data = json.loads(await f.read())
        except (FileNotFoundError, json.JSONDecodeError):
            self.data = {}
        return None


    def reload(self) -> None:
        """Reload metadata from disk."""
        self._load()
        return None


    async def reload_async(self) -> None:
        """Reload metadata from disk."""
        await self._load_async()
        return None


    def save(self) -> None:
        with open(self.path, 'w') as f:
            json.dump(self.data, f, indent=4)
        return None


    async def save_async(self) -> None:
        async with aiofiles.open(self.path, 'w', encoding="utf-8") as f:
            await f.write(json.dumps(self.data, indent=4))


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


    def set(self, key: str, value: Any, mode: Literal["e", "a", "a+s"]) -> None:
        keys = self._resolve_nested(key)
        data = self.data
        if mode == "e":
            for k in keys[:-1]:
                data = data[k]
            data[keys[-1]] = value
            if self.auto_save:
                print("HO SALVATO 1")
                self.save()
        elif mode == "a":
            for k in keys[:-1]:
                data = data[k]
            data[keys[-1]].append(value)
            if self.auto_save:
                print("HO SALVATO 2")
                self.save()
        elif mode == "a+s":
            for k in keys[:-1]:
                data = data[k]
            data[keys[-1]] = list(set(data[keys[-1]]).union({value}))
            if self.auto_save:
                self.save()
        return None


    async def set_async(self, key: str, value: Any, mode: Literal["e", "a", "a+s"]) -> None:
        keys = self._resolve_nested(key)
        data = self.data
        if mode == "e":
            for k in keys[:-1]:
                data = data[k]
            data[keys[-1]] = value
            if self.auto_save:
                await self.save_async()
        elif mode == "a":
            for k in keys[:-1]:
                data = data[k]
            data[keys[-1]].append(value)
            if self.save_async:
                await self.save_async()
        elif mode == "a+s":
            for k in keys[:-1]:
                data = data[k]
            data[keys[-1]] = list(set(data[keys[-1]]).union({value}))
            if self.save_async:
                await self.save_async()
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


    async def delete_async(self, key: str) -> None:
        keys = self._resolve_nested(key)
        data = self.data
        for k in keys[:-1]:
            if k not in data or not isinstance(data[k], dict):
                return
            data = data[k]
        data.pop(keys[-1], None)
        if self.auto_save:
            await self.save_async()



class ProjectsHubMetadataManager(BaseMetadataManager):
    ...



class ProjectMetadataManager(BaseMetadataManager):
    ...



class TRPMetadataManager(BaseMetadataManager):


    @property
    def _hub_metadata_manager(self) -> ProjectsHubMetadataManager:
        return ProjectsHubMetadataManager(ProjectsHub().hub / GlobalDefinitions.PROJECTS_HUB_METADATA.value)

    #Using @staticmethod since for instance methods that get used as decorators the instance itself (self) needs to be handled explicitly. More on that here: https://stackoverflow.com/questions/38524332/declaring-decorator-inside-a-class
    @staticmethod
    def trp_updated(func: callable) -> callable:
        @wraps(func)
        def wrapper(self, trp_id: str, *args, **kwargs):
            trp_metadata_fp = self._hub_metadata_manager.get(key="folder_paths.data.trp_metadata.path") + trp_id + "_metadata.json"
            print(trp_metadata_fp)
            print(self)
            self._init(path=trp_metadata_fp) #Every time the @trp_updated decorator gets called the metadata filepath gets updated by just replacing the old metadata with the TRP's one
            kwargs["trp_metadata_fp"] = trp_metadata_fp
            return func(*args, **kwargs)
        return wrapper


    @override
    def _init(self, path: str | Path | None) -> None:
        self.path = path #Set the metadata path
        self._load(self.path) #Load metadata if exists, else set it to a default value (which at the moment is {}, see in _load())
        return None


    @override
    async def _init_async(self, path: str | Path | None) -> None:
        self.path = path
        await self._load_async(self.path)
        return None


    @override
    def _load(self) -> None:
        try:
            with open(self.path, 'r', encoding="utf-8") as f:
                self.data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.data = {}
        return None


    @override
    async def _load_async(self) -> None:
        try:
            async with aiofiles.open(self.path, 'r', encoding="utf-8") as f:
                self.data = json.loads(await f.read())
        except (FileNotFoundError, json.JSONDecodeError):
            self.data = {}
        return None


    @trp_updated
    def set_trp_metadata(self, trp_id: str, **kwargs: Any) -> None:
        """
        Writes metadata for a single TRP (Traffic Registration Point).

        Parameters:
            trp_id: an alphanumeric string identifier of the TRP
            **kwargs: parameters which can be added directly into the metadata at write time

        Returns:
             None
        """
        default_settings = {"raw_volumes_file": None, GlobalDefinitions.HAS_VOLUME_CHECK.value: False, GlobalDefinitions.HAS_MEAN_SPEED_CHECK.value: False, "trp_data": None}
        tracking = {**default_settings, **kwargs}  # Overriding default settings with kwargs

        with open(kwargs["trp_metadata_fp"], "w", encoding="utf-8") as metadata_writer:
            json.dump({
            "id": trp_id,
            "trp_data": tracking["trp_data"],
            "files": {
                GlobalDefinitions.VOLUME.value: {
                    "raw": tracking["raw_volumes_file"],
                    "clean": None
                },
                GlobalDefinitions.MEAN_SPEED.value: {
                    "raw": None,
                    "clean": None
                }
            },
            "checks": {
                GlobalDefinitions.HAS_VOLUME_CHECK.value: tracking[GlobalDefinitions.HAS_VOLUME_CHECK.value],
                GlobalDefinitions.HAS_MEAN_SPEED_CHECK.value: tracking[GlobalDefinitions.HAS_MEAN_SPEED_CHECK.value]
            },
            "data_info": {
                GlobalDefinitions.VOLUME.value: {
                    "start_date": None,
                    "end_date": None
                },
                GlobalDefinitions.MEAN_SPEED.value: {
                    "start_date": None,
                    "end_date": None
                }
            }
        }, metadata_writer, indent=4)

        return None


    @trp_updated
    def get_trp_metadata(self, trp_id: str, **kwargs: Any) -> dict[Any, Any]:
        with open(kwargs["trp_metadata_fp"], "r", encoding="utf-8") as trp_metadata:
            return json.load(trp_metadata)

    #To update a single TRP's data just the set() method from the BaseMetadataManager father class and set the Path to the TRP's metadata file



class ProjectsHub:
    _instance = None

    #Making this class a singleton
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ProjectsHub, cls).__new__(cls)
        cls._instance._bootstrap_check()
        return cls._instance


    @property
    def hub(self) -> Path:
        return Path.cwd() / GlobalDefinitions.PROJECTS_HUB_DIR_NAME.value

    @property
    def _metadata_manager(self):
        return ProjectsHubMetadataManager(self.hub / GlobalDefinitions.PROJECTS_HUB_METADATA.value)

    @property
    def metadata(self) -> dict["str", Any] | None:
        try:
            return self._metadata_manager.get(key="metadata")
        except FileNotFoundError:
            return None

    @property
    def projects(self) -> list[str]:
        return self._metadata_manager.get(key="projects")


    def create_hub(self) -> None:
        Path(self.hub).mkdir(parents=True, exist_ok=True)
        if not Path(self.hub / GlobalDefinitions.PROJECTS_HUB_METADATA.value).exists():
            self._write_hub_metadata()
        return None


    def delete_hub(self) -> None:
        os.rmdir(self.hub)
        return None


    def _bootstrap_check(self) -> None:
        self.create_hub()  # Instantiating a hub (in case it already exists nothing will happen)

        if not self.projects:
            print("No projects set. Set one now")
            project_name = clean(input("Impute project name: "), no_punct=True, no_emoji=True, no_emails=True, no_currency_symbols=True, no_urls=True)
            self.create_project(project_name)
            self.set_current_project(project_name)
            print("Newly created project automatically set as active")

        if self.projects and self.metadata["current_project"] is None:
            print("No current project set. Set one of the available ones")
            print("Available projects:\n", self.projects)
            self.set_current_project(clean(input("Impute project name: "), no_punct=True, no_emoji=True, no_emails=True, no_currency_symbols=True, no_urls=True))

        return None


    def _write_hub_metadata(self) -> None:
        with open(self.hub / GlobalDefinitions.PROJECTS_HUB_METADATA.value, "w", encoding="utf-8") as gm:
            json.dump({
                "metadata": {
                    "current_project": None,
                    "lang": "en"
                },
                "projects": []
            }, gm, indent=4)
        return None


    def set_current_project(self, name: str) -> None:
        self._metadata_manager.set(value=name, key="metadata.current_project", mode="e")
        return None


    def get_current_project(self, errors: bool = True) -> str | None:
        current_project = self.metadata["current_project"]
        if errors and not current_project:
            raise ValueError("Current project not set")
        elif errors is False and not current_project:
            print("\033[91mCurrent project not set\033[0m")
            return None
        return current_project


    def reset_current_project(self) -> None:
        self._metadata_manager.set(value=None, key="metadata.current_project", mode="e")
        return None


    def create_project(self, name: str):

        #Creating the project's directory
        Path(self.hub / name, exist_ok=True).mkdir(exist_ok=True)
        self._metadata_manager.set(value=name, key="projects", mode="a+s")

        folder_structure = {
            "data": {
                **{sub: {
                        "raw": {},
                        "clean": {}
                    } for sub in (GlobalDefinitions.VOLUME.value, GlobalDefinitions.MEAN_SPEED.value, "travel_times")
                },
                "trp_metadata": {}
                # No subfolders
            },
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
            main_dir = self.hub / name / key
            os.makedirs(main_dir, exist_ok=True)
            metadata_folder_structure[key] = create_nested_folders(str(main_dir), sub_structure)

        self._write_project_metadata(dir_name=name, **{"metadata_folder_structure": metadata_folder_structure})  #Creating the project's metadata file

        return None


    def _write_project_metadata(self, dir_name: str, **kwargs: Any) -> None:
        with open(Path(self.hub / dir_name / GlobalDefinitions.PROJECT_METADATA.value), "w", encoding="utf-8") as tf:
            json.dump({
            "common": {
                "traffic_registration_points_file": str(Path(dir_name, GlobalDefinitions.DATA_DIR.value, GlobalDefinitions.TRAFFIC_REGISTRATION_POINTS_FILE.value)),
            },
            "volume": {
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
            "folder_paths": kwargs.get("metadata_folder_structure", {}),
            "forecasting": {"target_datetimes": {"V": None, "AS": None}},
            "trps": {}  # For each TRP we'll have {"id": metadata_filename}
        }, tf, indent=4)
        return None


    def delete_project(self, name: str) -> None:
        os.remove(self.hub / name)
        return None

    @property
    def trps_fp(self) -> Path:
        return Path(self.hub, self.get_current_project(), GlobalDefinitions.DATA_DIR.value, GlobalDefinitions.TRAFFIC_REGISTRATION_POINTS_FILE.value)

    @property
    @lru_cache
    def trps_data(self):
        """
        This function returns json data about all TRPs (downloaded previously)
        """
        with open(self.trps_fp, "r", encoding="utf-8") as trps:
            return json.load(trps)

    @property
    @alru_cache()
    async def trps_data_async(self):
        async with aiofiles.open(self.trps_fp, "r", encoding="utf-8") as trps:
            return await json.loads(await trps.read())



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
    tmm: TRPMetadataManager
    model_config = {
        "ignored_types": (async_lru._LRUCacheWrapper,)
    }


    #TODO EVALUATE A POSSIBLE CACHING OF THESE AS WELL. BUT KEEP IN MIND POTENTIAL CHANGES DUE TO RE-DOWNLOAD OF TRPS DURING THE SAME EXECUTION OF THE CODE
    def get_trp_ids(self) -> list[str]:
        with open(self.get(key="common" + GlobalDefinitions.TRAFFIC_REGISTRATION_POINTS_FILE.value), "r", encoding="utf-8") as f:
            return list(json.load(f).keys())
    #TODO THIS METHOD IS NOT ASYNC, CHECK IF IT GETS USED IN ASYNC METHODS OR FUNCTIONS

    def get_trp_ids_by_road_category(self, target: str) -> dict[str, list[str]] | None:

        #TODO ADD check_target() HERE

        road_categories = set(trp["location"]["roadReference"]["roadCategory"]["id"] for trp in self.get_global_trp_data().values())
        clean_data_folder = self.tmm.get(key="folder_paths.data." + target + ".subfolders.clean.path")
        check = "has_" + target
        data = GlobalDefinitions.CLEAN_VOLUME_FILENAME_ENDING.value + ".csv" if target == GlobalDefinitions.TARGET_DATA.value["V"] else GlobalDefinitions.CLEAN_MEAN_SPEED_FILENAME_ENDING.value + ".csv"  # TODO THIS WILL BE REMOVED WHEN THE TARGET VARIABLE NAME PROBLEM WILL BE SOLVED

        return {k: d for k, d in {
            category: [clean_data_folder + trp_id + data for trp_id in
                       filter(lambda trp_id:
                              self.tmm.get_trp_metadata(trp_id)["trp_data"]["location"]["roadReference"]["roadCategory"]["id"] == category and self.tmm.get_trp_metadata(trp_id)["checks"][check], self.get_trp_ids())]
            for category in road_categories
        }.items() if len(d) >= 2}
        # Removing key value pairs from the dictionary where there are less than two dataframes to concatenate, otherwise this would throw an error in the merge() function



class RoadNetworkToolbox(BaseModel):
    project_metadata_manager: ProjectMetadataManager


    def retrieve_edges(self) -> dict:
        with open(f"{self.project_metadata_manager.get('folder_paths.rn_graph.edges.path')}/traffic-nodes-2024_2025-02-28.geojson", "r", encoding="utf-8") as e:
            return geojson.load(e)["features"]


    def retrieve_arches(self) -> dict:
        with open(f"{self.project_metadata_manager.get('folder_paths.rn_graph.arches.path')}/traffic_links_2024_2025-02-27.geojson", "r", encoding="utf-8") as a:
            return geojson.load(a)["features"]



class ForecastingToolbox(BaseModel):
    gp_toolbox: GeneralPurposeToolbox
    pmm: ProjectMetadataManager
    tmm: TRPMetadataManager


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

        option = input("V = Volumes | MS = Mean Speed")
        print("Maximum number of days to forecast: ", max_forecasting_window_size)

        if option == GlobalDefinitions.TARGET_DATA.value["V"]:
            last_available_data_dt = self.pmm.get(key=GlobalDefinitions.VOLUME.value + ".end_date_iso")
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
            return datetime.strptime(self.project_metadata_manager.get(key="forecasting.target_datetimes" + target), GlobalDefinitions.DT_FORMAT.value)
        except TypeError:
            raise Exception(f"\033[91mTarget datetime for {target} isn't set yet. Set it first and then execute a one-point forecast\033[0m")


    def reset_forecasting_horizon(self, target: str) -> None:
        try:
            self.project_metadata_manager.set(value=None, key="forecasting.target_datetimes" + target, mode="e")
            print("Target datetime reset successfully\n\n")
            return None
        except KeyError:
            raise KeyError("Target datetime not found")









