import os
from typing import Any, ClassVar
from datetime import timedelta, timezone
from pathlib import Path
from enum import Enum
from pydantic import BaseModel
from pydantic.types import PositiveInt
from psycopg.rows import tuple_row, dict_row

from dbsecrets import superuser, superuser_password


class GlobalDefinitions(BaseModel):
    class Config:
        arbitrary_types_allowed = True
        frozen = True

    VOLUME_INGESTION_FIELDS: ClassVar[list[str]] = ["trp_id", "volume", "coverage", "is_mice", "zoned_dt_iso"]
    MEAN_SPEED_INGESTION_FIELDS: ClassVar[list[str]] = ["trp_id", "mean_speed", "percentile_85", "coverage", "is_mice", "zoned_dt_iso"]
    TARGET_DATA: ClassVar[dict[str, str]] = {"V": "volume", "MS": "mean_speed"}
    ROAD_CATEGORIES: ClassVar[list[str]] = ["E", "R", "F", "K", "P"]
    DEFAULT_MAX_FORECASTING_WINDOW_SIZE: ClassVar[PositiveInt] = 14

    HAS_VOLUME_CHECK: ClassVar[str] = "has_volume"
    HAS_MEAN_SPEED_CHECK: ClassVar[str] = "has_mean_speed"

    VOLUME: ClassVar[str] = "volume"
    MEAN_SPEED: ClassVar[str] = "mean_speed"

    DT_ISO_TZ_FORMAT: ClassVar[str] = "%Y-%m-%dT%H:%M:%S%z"
    DT_INPUT_FORMAT: ClassVar[str] = "%Y-%m-%dT%H"
    NORWEGIAN_UTC_TIME_ZONE: ClassVar[str] = "+01:00"
    NORWEGIAN_UTC_TIME_ZONE_TIMEDELTA: ClassVar[timezone] = timezone(timedelta(hours=1))

    COORDINATES_REFERENCE_SYSTEM: ClassVar[int] = 4326 #WGS84

    COVID_YEARS: ClassVar[list[int]] = [2020, 2021, 2022]
    ML_CPUS: ClassVar[int] = int(os.cpu_count() * 0.75)  # To avoid crashing while executing parallel computing in the GridSearchCV algorithm
    # The value multiplied with the n_cpu values shouldn't be above .80, otherwise processes could crash during execution

    MEAN_SPEED_DIR: ClassVar[Path] = Path("data", MEAN_SPEED)
    MODEL_GRIDS_FILE: ClassVar[Path] = Path("data", "model_grids.json")

    MICE_COLS: ClassVar[list[str]] = ["volume", "coverage"]

    NETWORKX_BACKEND: ClassVar[str] = "networkx"
    CUDF_BACKEND: ClassVar[str] = "cudf"
    GRAPH_PROCESSING_BACKENDS: ClassVar[list[str]] = [NETWORKX_BACKEND, CUDF_BACKEND]

    NON_PREDICTORS: ClassVar[list[str]] = ["zoned_dt_iso"]
    ENCODED_FEATURES: ClassVar[list[str]] = ["trp_id"]
    VOLUME_SCALED_FEATURES: ClassVar[list[str]] = [VOLUME, "coverage"]
    MEAN_SPEED_SCALED_FEATURES: ClassVar[list[str]] = [MEAN_SPEED, "percentile_85", "coverage"]

    DEFAULT_DASK_DF_PARTITION_SIZE: ClassVar[str] = "512MB"

    OSLO_COUNTY_ID = "3"



class DBConfig(Enum):
    SUPERUSER = superuser
    SUPERUSER_PASSWORD = superuser_password
    TFS_USER = "tfs"
    TFS_PASSWORD = "tfs"
    TFS_ROLE = "tfs_admin"
    TFS_ROLE_PASSWORD = "tfs"

    HUB_DB = "tfs_hub"
    MAINTENANCE_DB = "postgres"

    DB_HOST = "localhost"



class AIODBManagerInternalConfig(Enum):
    PUBLIC_SCHEMA = "public"
    HUB_DB_TABLES_SCHEMA = "public" #NOTE BY DEFAULT ALL TABLES FOR WHICH ISN'T SPECIFIED A SCHEMA GET ASSIGNED TO THE "public" SCHEMA. IN THE FUTURE WE'LL DEFINE SPECIFIC SCHEMAS FOR THE TABLES USED IN THE PROJECT



class HubDBTables(Enum):
    Projects = "Projects"



class HUBDBConstraints(Enum):
    ONE_CURRENT_PROJECT = "one_current_project"



class ProjectTables(Enum):
    RoadCategories = "RoadCategories"
    CountryParts = "CountryParts"
    Counties = "Counties"
    Municipalities = "Municipalities"
    TrafficRegistrationPoints = "TrafficRegistrationPoints"
    Volume = "Volume"
    MeanSpeed = "MeanSpeed"
    MLModels = "MLModels"
    TrainedModels = "TrainedModels"
    BaseModels = "BaseModels"
    ModelGridSearchCVResults = "ModelGridSearchCVResults"
    ForecastingSettings = "ForecastingSettings"
    RoadGraphNodes = "RoadGraphNodes"
    RoadGraphLinks = "RoadGraphLinks"
    RoadNetworks = "RoadNetworks"
    TollStations = "TollStations"
    FunctionClasses = "FunctionClasses"
    RoadLink_Municipalities = "RoadLink_Municipalities"
    RoadLink_Counties = "RoadLink_Counties"
    RoadLink_TollStations = "RoadLink_TollStations"
    RoadLink_TrafficRegistrationPoints = "RoadLink_TrafficRegistrationPoints"



class ProjectConstraints(Enum):
    UNIQUE_VOLUME_PER_TRP_AND_TIME = "unique_volume_per_trp_and_time"
    UNIQUE_MEAN_SPEED_PER_TRP_AND_TIME = "unique_mean_speed_per_trp_and_time"
    UNIQUE_MODEL_ROAD_TARGET_PARAMS = "unique_model_road_target_params"



class ProjectViews(Enum):
    TrafficRegistrationPointsMetadataView = "TrafficRegistrationPointsMetadataView"
    VolumeMeanSpeedDateRangesView = "VolumeMeanSpeedDateRangesView"



class FunctionClasses(Enum):
    A = "Nasjonale hovedveger"
    B = "Regionale hovedveger"
    C = "Lokale hovedveger"
    D = "Lokale samleveger"
    E = "Lokale adkomstveger"



class RowFactories(BaseModel):
    class Config:
        arbitrary_types_allowed = True
        frozen = True

    factories: ClassVar[dict[str, Any]] = {
        "tuple_row": tuple_row,
        "dict_row": dict_row
    }


