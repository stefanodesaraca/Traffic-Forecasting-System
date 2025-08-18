from enum import Enum
from typing import Any, ClassVar
from pydantic import BaseModel
from psycopg.rows import tuple_row, dict_row

from dbsecrets import superuser, superuser_password



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
    TrafficRegistrationPointsMetadata = "TrafficRegistrationPointsMetadata"
    MLModels = "MLModels"
    TrainedModels = "TrainedModels"
    BaseModels = "BaseModels"
    ModelGridSearchCVResults = "ModelGridSearchCVResults"
    ForecastingSettings = "ForecastingSettings"
    RoadGraphNodes = "RoadGraphNodes"
    RoadGraphLinks = "RoadGraphLinks"



class ProjectConstraints(Enum):
    UNIQUE_VOLUME_PER_TRP_AND_TIME = "unique_volume_per_trp_and_time"
    UNIQUE_MEAN_SPEED_PER_TRP_AND_TIME = "unique_mean_speed_per_trp_and_time"



class ProjectViews(Enum):
    TrafficRegistrationPointsMetadataView = "TrafficRegistrationPointsMetadataView"
    VolumeMeanSpeedDateRangesView = "VolumeMeanSpeedDateRangesView"



class RowFactories(BaseModel):
    class Config:
        arbitrary_types_allowed = True
        frozen = True

    factories: ClassVar[dict[str, Any]] = {
        "tuple_row": tuple_row,
        "dict_row": dict_row
    }
