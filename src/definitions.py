import os
from typing import Any, ClassVar
from datetime import timedelta, timezone
from pathlib import Path
from enum import Enum
from pydantic import BaseModel
from pydantic.types import PositiveInt
from psycopg.rows import tuple_row, dict_row
from folium import CircleMarker, Icon
from matplotlib.colors import LinearSegmentedColormap


from proj_secrets import superuser, superuser_password


class GlobalDefinitions(BaseModel):
    class Config:
        arbitrary_types_allowed = True
        frozen = True

    VOLUME_INGESTION_FIELDS: ClassVar[list[str]] = ["trp_id", "volume", "coverage", "is_mice", "zoned_dt_iso"]
    MEAN_SPEED_INGESTION_FIELDS: ClassVar[list[str]] = ["trp_id", "mean_speed", "percentile_85", "coverage", "is_mice", "zoned_dt_iso"]
    TARGET_DATA: ClassVar[dict[str, str]] = {"V": "volume", "MS": "mean_speed"}
    ROAD_CATEGORIES: ClassVar[list[str]] = ["E", "R", "F", "K", "P"]
    HIGH_SPEED_ROAD_CATEGORIES: ClassVar[list[str]] = ["E", "R"]

    HAS_VOLUME_CHECK: ClassVar[str] = "has_volume"
    HAS_MEAN_SPEED_CHECK: ClassVar[str] = "has_mean_speed"

    VOLUME: ClassVar[str] = "volume"
    MEAN_SPEED: ClassVar[str] = "mean_speed"

    DT_ISO_TZ_FORMAT: ClassVar[str] = "%Y-%m-%dT%H:%M:%S%z"
    DT_INPUT_FORMAT: ClassVar[str] = "%Y-%m-%dT%H"
    NORWEGIAN_UTC_TIME_ZONE: ClassVar[str] = "+01:00"
    NORWEGIAN_UTC_TIME_ZONE_TIMEDELTA: ClassVar[timezone] = timezone(timedelta(hours=1))

    WGS84_REFERENCE_SYSTEM: ClassVar[int] = 4326 #WGS84

    COVID_YEARS: ClassVar[list[int]] = [2020, 2021, 2022]
    ML_CPUS: ClassVar[int] = int(os.cpu_count() * 0.75)  # To avoid crashing while executing parallel computing in the GridSearchCV algorithm
    # The value multiplied with the n_cpu values shouldn't be above .80, otherwise processes could crash during execution

    MEAN_SPEED_DIR: ClassVar[Path] = Path("data", MEAN_SPEED)
    MODEL_GRIDS_FILE: ClassVar[Path] = Path("data", "model_grids.json")
    MUNICIPALITIES_AUXILIARY_DATA: ClassVar[Path] = Path("data", "road_network", "kommuner.csv")
    MODELS_BEST_PARAMS: ClassVar[Path] = Path("data", "models_best_params.json")

    MICE_COLS: ClassVar[list[str]] = ["volume", "coverage"]

    NETWORKX_BACKEND: ClassVar[str] = "networkx"
    CUDF_BACKEND: ClassVar[str] = "cudf"
    GRAPH_PROCESSING_BACKENDS: ClassVar[list[str]] = [NETWORKX_BACKEND, CUDF_BACKEND]

    PREPROCESSING_SORTING_COLUMNS: ClassVar[list[str]] = ["zoned_dt_iso"] #Sorting by zoned_dt_iso since the trp_id will be removed at the end of the preprocessing phase and this lets us use the TimeSeriesSplit CV splitting method correctly
    NON_PREDICTORS: ClassVar[list[str]] = ["trp_id", "lat", "lon", "zoned_dt_iso"]
    CATEGORICAL_FEATURES: ClassVar[list[str]] = ["trp_id"]
    VOLUME_TO_SCALE_FEATURES: ClassVar[list[str]] = [VOLUME, "coverage"]
    MEAN_SPEED_TO_SCALE_FEATURES: ClassVar[list[str]] = [MEAN_SPEED, "percentile_85", "coverage"]

    DEFAULT_DASK_DF_PARTITION_SIZE: ClassVar[str] = "512MB"

    OSLO_COUNTY_ID: ClassVar[str] = "3"

    DEFAULT_MAX_FORECASTING_WINDOW_SIZE: ClassVar[PositiveInt] = 1
    HOUR_TIMEFRAME: ClassVar[PositiveInt] = 24
    DAYS_TIMEFRAME: ClassVar[PositiveInt] = 31
    MONTHS_TIMEFRAME: ClassVar[PositiveInt] = 12
    WEEKS_TIMEFRAME: ClassVar[PositiveInt] = 53
    SHORT_TERM_LAGS: ClassVar[list[int]] = [24, 36, 48, 60, 72]



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
    ModelBestParameters = "ModelBestParameters"



class ProjectConstraints(Enum):
    UNIQUE_VOLUME_PER_TRP_AND_TIME = "unique_volume_per_trp_and_time"
    UNIQUE_MEAN_SPEED_PER_TRP_AND_TIME = "unique_mean_speed_per_trp_and_time"
    UNIQUE_MODEL_ROAD_TARGET_PARAMS = "unique_model_road_target_params"



class ProjectViews(Enum):
    TrafficRegistrationPointsMetadataView = "TrafficRegistrationPointsMetadataView"
    VolumeMeanSpeedDateRangesView = "VolumeMeanSpeedDateRangesView"
    BestGridSearchResults = "BestGridSearchResults"



class ProjectMaterializedViews(Enum):
    TrafficDataByCountyMView = "TrafficDataByCountyMView"
    TrafficDataByMunicipalityMView = "TrafficDataByMunicipalityMView"
    TrafficDataByRoadCategoryMView = "TrafficDataByRoadCategoryMView"



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



class FoliumMapTiles(Enum):
    CARTO_DB_POSITRON = "CartoDB positron"
    OPEN_STREET_MAPS = "OpenStreetMap"
    ESRI_SATELLITE = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}" #TODO TO FILL THIS UP WITH z, y, x
    #https://gis.stackexchange.com/questions/290861/python-folium-package-for-satellite-map
    #https://python-visualization.github.io/folium/latest/user_guide/raster_layers/tiles.html



class IconStyles(Enum):
    SOURCE_NODE_STYLE = {
        "icon_color": "#4a90e2",
        "icon": "glyphicon-home"
    }
    DESTINATION_NODE_STYLE = {
        "icon_color": "#c75146",
        "icon": "glyphicon-flag"
    }
    TRP_LINK_STYLE = {
        "icon_color": "#E7F527",
        "icon": "glyphicon-modal-window"
    }
    TOLL_STATION_LINK_STYLE = {
        "icon_color": "#F4550B",
        "icon": "glyphicon-euro"
    }
    PUBLIC_TRANSPORT_ONLY_LINK_STYLE = {
        "icon_color": "#F40B5D",
        "icon": "glyphicon-asterisk"
    }
    #Icons taken from: https://getbootstrap.com/docs/3.3/components/
    #Colors taken from:
    # https://www.color-hex.com/color-palette/1063914
    # https://www.color-hex.com/color-palette/1063917
    # https://htmlcolorcodes.com/



class Icons(Enum):
    SOURCE_NODE: CircleMarker
    DESTINATION_NODE: Icon



class TrafficClasses(Enum): #Import as TrafficClasses
    LOW = "#108863"
    LOW_AVERAGE = "#789741"
    AVERAGE = "#e0a71f"
    HIGH_AVERAGE = "#f49d1f"
    HIGH = "#a91d1d"
    STOP_AND_GO = "#470C00"

    TRAFFIC_PRIORITIES = {
        LOW.name: 0,
        LOW_AVERAGE.name: 1,
        AVERAGE.name: 2,
        HIGH_AVERAGE.name: 3,
        HIGH.name: 4,
        STOP_AND_GO.name: 5
    }

    CMAP = LinearSegmentedColormap.from_list("TrafficClassesCMAP", [LOW, LOW_AVERAGE, AVERAGE, HIGH_AVERAGE, HIGH, STOP_AND_GO], N=256)

    #Colors taken from:
    # https://www.color-hex.com/color-palette/1064006
    # https://www.color-hex.com/color-palette/1063978
    # https://www.color-hex.com/color-palette/1064022



class RoadCategoryTraitLengthWeightMultipliers(Enum):
    E = 0.5
    R = 0.7
    F = 1.3
    K = 1.5
    P = 2.0



class MapDefaultConfigs(Enum):
    ZOOM = 8
