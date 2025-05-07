from datetime import datetime
from typing import Any, Generator
import os
import json
import pprint
import sys
import traceback
import logging
import asyncio
import aiofiles
from functools import lru_cache
import pandas as pd
import dask.dataframe as dd
from cleantext import clean
import geopandas as gpd
import geojson
from dateutil.relativedelta import relativedelta
from geopandas import GeoDataFrame
from pydantic.types import PositiveInt


pd.set_option("display.max_columns", None)


cwd = os.getcwd()
ops_folder = "ops"
dt_iso = "%Y-%m-%dT%H:%M:%S.%fZ"
dt_format = "%Y-%m-%dT%H"  # Datetime format, the hour (H) must be zero-padded and 24-h base, for example: 01, 02, ..., 12, 13, 14, 15, etc.
# In this case we'll only ask for the hour value since, for now, it's the maximum granularity for the predictions we're going to make
metainfo_filename = "metainfo"
target_data = {"V": "traffic_volumes", "AS": "average_speeds"} #TODO (IN THE FUTURE) CONVERT AS TO "MS" AND "mean_speed"
target_data_temp = {"V": "traffic_volumes", "AS": "mean_speed"} #NOTE TEMPORARY SOLUTION FOR THE ABOVE TODO
default_max_forecasting_window_size = 14
active_ops_filename = "active_ops"
metainfo_lock = asyncio.Lock()


# TODO CREATE AN import_single_trp_data FUNCTION WHICH GETS THE SINGLE TRP's DATA FROM THE by_trp dict IN METAINFO

# ==================== TRP Utilities ====================

@lru_cache()
def import_TRPs_data():
    """
    This function returns json data about all TRPs (downloaded previously)
    """
    f = read_metainfo_key(keys_map=["common", "traffic_registration_points_file"])
    assert os.path.isfile(f), "Traffic registration points file missing"
    with open(f, "r", encoding="utf-8") as TRPs:
        return json.load(TRPs)


# ------------ Metadata ------------

def get_trp_metadata(trp_id: str) -> dict:
    with open(read_metainfo_key(keys_map=["folder_paths", "trp_metadata", "path"]) + f"{trp_id}_metadata.json", "r", encoding="utf-8") as json_trp_metadata:
        return json.load(json_trp_metadata)


def write_trp_metadata(trp_id: str, data: dict) -> None:
    metadata = {
        "id": trp_id,
        "info": {
            "road_category": data["location"]["roadReference"]["roadCategory"]["id"],
            "lat": data["location"]["coordinates"]["latLon"]["lat"],
            "lon": data["location"]["coordinates"]["latLon"]["lon"],
            "county_name": data["location"]["county"]["name"],
            "county_number": data["location"]["county"]["number"],
            "geographic_number": data["location"]["county"]["geographicNumber"],
            "country_part": data["location"]["county"]["countryPart"]["name"],
            "municipality_name": data["location"]["municipality"]["name"],
            "traffic_registration_type": data["trafficRegistrationType"],
            "first_data": data["dataTimeSpan"]["firstData"],
            "first_data_with_quality_metrics": data["dataTimeSpan"]["firstDataWithQualityMetrics"],
            "latest_volume_by_day": data["dataTimeSpan"]["latestData"]["volumeByDay"],
            "latest_volume_byh_hour": data["dataTimeSpan"]["latestData"]["volumeByHour"],
            "latest_volume_average_daily_by_year": data["dataTimeSpan"]["latestData"]["volumeAverageDailyByYear"],
            "latest_volume_average_daily_by_season": data["dataTimeSpan"]["latestData"]["volumeAverageDailyBySeason"],
            "latest_volume_average_daily_by_month": data["dataTimeSpan"]["latestData"]["volumeAverageDailyByMonth"],
        },
        "number_of_data_nodes": None,
        "files": {
            "volumes": {
                "raw": None,
                "clean": None
            },
            "speeds": {
                "raw": None,
                "clean": None
            }
        },
        "checks": {
            "has_volumes": False,
            "has_speeds": False
        },
        "data_info": {
            "start_date": None,
            "end_date": None
        }
    }

    with open(read_metainfo_key(keys_map=["folder_paths", "trp_metadata", "path"]) + trp_id + "_metadata.json", "w") as metadata_writer:
        json.dump(metadata, metadata_writer, indent=4)

    return None


def update_trp_metadata(trp_id: str, value: Any, metadata_keys_map: list[str], mode: str) -> None:
    modes = ["equals", "append"]
    metadata_filepath = read_metainfo_key(keys_map=["folder_paths", "trp_metadata", "path"]) + trp_id + "_metadata.json"
    with open(metadata_filepath, "r", encoding="utf-8") as f:
        payload = json.load(f)

    # metadata = payload has a specific reason to exist
    # This is how we preserve the whole original dictionary (loaded from the JSON file), but at the same time iterate over its keys and updating them
    # By doing to we'll assign the value (obtained from the value parameter of this method) to the right key, but preserving the rest of the dictionary
    metadata = payload

    if mode == "equals":
        for key in metadata_keys_map[:-1]:
            metadata = metadata[key]
        metadata[metadata_keys_map[-1]] = value  # Updating the metainfo file key-value pair
        with open(metadata_filepath, "w", encoding="utf-8") as m:
            json.dump(payload, m, indent=4)
    elif mode == "append":
        for key in metadata_keys_map[:-1]:
            metadata = metadata[key]
        metadata[metadata_keys_map[-1]].append(value)  # Appending a new value to the list (which is the value of this key-value pair)
        with open(metadata_filepath, "w", encoding="utf-8") as m:
            json.dump(payload, m, indent=4)
    elif mode not in modes:
        print("\033[91mWrong mode\033[0m")
        sys.exit(1)

    return None








# TODO IMPROVE THIS FUNCTION, AND SET THAT THIS RETURNS A generator OF str
def get_trp_ids() -> Generator[str, None, None]:
    raw_volumes_folder = read_metainfo_key(keys_map=["folder_paths", "data", "traffic_volumes", "subfolders", "raw", "path"])
    # If there aren't volumes files yet, then just return all the TRP IDs available in the traffic_registration_points_file
    if len(os.listdir(raw_volumes_folder)) == 0:
        return (trp["id"] for trp in import_TRPs_data()["trafficRegistrationPoints"]) # This list may contain IDs of TRPs which don't have a volumes file associated with them #TODO CHANGE THIS. IF THERE AREN'T ANY VOLUMES FILES THIS SHOULDN'T RETURN ALL TRP IDs
    else:
        return (trp["id"] for trp in import_TRPs_data()["trafficRegistrationPoints"]
                if trp["id"] in [get_trp_id_from_filename(file) for file in os.listdir(raw_volumes_folder)]) # Keep only the TRPs which actually have a volumes file associated with them. Using list() on a set comprehension to avoid duplicates
                #TODO ALSO, FIND A WAY TO KNOW DIRECTLY WHICH TRPS HAVE VOLUMES DATA AND WHICH DON'T SO IT WON'T BE NECESSARY TO CHECK EVERYTIME os.listdir(raw_volumes_folder) AND GET THE TRP NAME FROM THE FILENAMES


def get_trp_id_from_filename(filename: str) -> str:
    return filename.split("_")[0]


def get_all_available_road_categories() -> list:
    return list(set(trp["location"]["roadReference"]["roadCategory"]["id"] for trp in import_TRPs_data()["trafficRegistrationPoints"]))


def get_trp_road_category(trp_id: str) -> str:
    return get_trp_metadata(trp_id)["road_category"]


def retrieve_trp_road_category(trp_id: str) -> str:
    return get_trp_metadata(trp_id)["road_category"]


def retrieve_trp_clean_volumes_filepath_by_id(trp_id: str) -> str:
    return get_trp_metadata(trp_id)["clean_volumes_filepath"]


def retrieve_trp_clean_average_speed_filepath_by_id(trp_id: str) -> str:
    return get_trp_metadata(trp_id)["clean_average_speed_filepath"]


# ==================== Volumes Utilities ====================


# ==================== Average Speeds Utilities ====================


# ==================== ML Related Utilities ====================

#TODO SIMPLIFY THESE TWO FUNCTIONS BELOW (MEANING CREATE ONLY ONE WHICH CAN SERVE BOTH FUNCTIONALITIES
def get_ml_models_folder_path(target: str, road_category: str) -> str:
    folder_paths = {
        "volume": read_metainfo_key(keys_map=["folder_paths", "ml", "models", "subfolders", "traffic_volumes", "subfolders", road_category, "path"]),
        "average_speed": read_metainfo_key(keys_map=["folder_paths", "ml", "models", "subfolders", "average_speed", "subfolders", road_category, "path"])
    }
    if folder_paths[target]:
        return folder_paths[target]
    else:
        raise Exception("Wrong target variable in the get_ml_models_folder_path() function")


def get_ml_models_parameters_folder_path(target: str, road_category: str) -> str:
    folder_paths = {
        "volume": read_metainfo_key(keys_map=["ml", "models_parameters", "subfolders", "traffic_volumes", "subfolders", road_category, "path"]),
        "average_speed": read_metainfo_key(keys_map=["ml", "models_parameters", "subfolders", "average_speed", "subfolders", road_category, "path"])
    }
    if folder_paths[target]:
        return folder_paths[target]
    else:
        raise Exception("Wrong target variable in the get_ml_model_parameters_folder_path() function")


# ==================== Forecasting Settings Utilities ====================

# TODO FIND A WAY TO CHECK WHICH IS THE LAST DATETIME AVAILABLE FOR AVERAGE SPEED (CLEAN)

def write_forecasting_target_datetime(forecasting_window_size: PositiveInt = default_max_forecasting_window_size) -> None:
    """
    Parameters:
        forecasting_window_size: in days, so hours-speaking, let x be the windows size, this will be x*24
    """
    assert os.path.isdir(read_metainfo_key(keys_map=["folder_paths", "data", "traffic_volumes", "subfolders", "clean", "path"])), "Clean traffic volumes folder missing. Initialize an operation first and then set a forecasting target datetime"
    assert os.path.isdir(read_metainfo_key(keys_map=["folder_paths", "data", "average_speed", "subfolders", "clean", "path"])), "Clean average speeds folder missing. Initialize an operation first and then set a forecasting target datetime"

    max_forecasting_window_size = max(default_max_forecasting_window_size, forecasting_window_size)  # The maximum number of days that can be forecasted is equal to the maximum value between the default window size (14 days) and the maximum window size that can be set through the function parameter

    option = input("Press V to set forecasting target datetime for traffic volumes or AS for average speeds: ")
    print("Maximum number of days to forecast: ", max_forecasting_window_size)

    last_available_data_dt = read_metainfo_key(keys_map=[target_data[option], "end_date_iso"])
    if last_available_data_dt is None:
        logging.error(traceback.format_exc())
        raise Exception("End date not found in metainfo file. Run download first or set it first")

    print("Latest data available: ", datetime.strptime(last_available_data_dt, dt_iso))
    print("Maximum settable date: ", relativedelta(datetime.strptime(last_available_data_dt, dt_iso), days=14))

    dt = input("Insert Target Datetime (YYYY-MM-DDTHH): ") # The month number must be zero-padded, for example: 01, 02, etc.

    forecasting_window_size = (datetime.strptime(dt, dt_format) - datetime.strptime(last_available_data_dt, dt_iso)).days  # The number of days to forecast

    assert datetime.strptime(dt, dt_format) > datetime.strptime(last_available_data_dt, dt_iso), "Forecasting target datetime is prior to the latest data available, so the data to be forecasted is already available"  # Checking if the imputed date isn't prior to the last one available. So basically we're checking if we already have the data that one would want to forecast
    assert forecasting_window_size <= max_forecasting_window_size, f"Number of days to forecast exceeds the limit: {max_forecasting_window_size}"  # Checking if the number of days to forecast is less or equal to the maximum number of days that can be forecasted

    if check_datetime(dt) is True and option in list(target_data.keys()):
        update_metainfo(value=dt, keys_map=["forecasting", "target_datetimes", option], mode="equals",)
        print("Target datetime set to: ", dt, "\n\n")
        return None
    else:
        if check_datetime(dt) is False:
            print("\033[91mWrong datetime format, try again\033[0m")
            sys.exit(1)
        elif option not in list(target_data.keys()):
            print("\033[91mWrong data option, try again\033[0m")
            sys.exit(1)


def read_forecasting_target_datetime(data_kind: str) -> datetime:
    try:
        target_dt = read_metainfo_key(keys_map=["forecasting", "target_datetimes", data_kind])
        return datetime.strptime(target_dt, dt_format)
    except TypeError:
        print(f"\033[91mTarget datetime for {data_kind} isn't set yet. Set it first and then execute a one-point forecast\033[0m")
        sys.exit(1)
    except FileNotFoundError:
        print("\033[91mTarget Datetime File Not Found\033[0m")
        sys.exit(1)


def rm_forecasting_target_datetime() -> None:
    try:
        print("For which data kind do you want to remove the forecasting target datetime?")
        option = input("Press V to set forecasting target datetime for traffic volumes or AS for average speeds:")
        update_metainfo(None, ["forecasting", "target_datetimes", option], mode="equals")
        print("Target datetime file deleted successfully\n\n")
        return None
    except KeyError:
        print("\033[91mTarget datetime not found\033[0m")
        sys.exit(1)


# ==================== Operations' Settings Utilities ====================


# The user sets the current operation
def write_active_ops_file(ops_name: str) -> None:
    ops_name = clean_text(ops_name)
    assert os.path.isfile(f"{ops_folder}/{ops_name}") is True, f"{ops_name} operation folder not found. Create an operation with that name first."
    with open(f"{active_ops_filename}.txt", "w", encoding="utf-8") as ops_file:
        ops_file.write(ops_name)
    return None


# Reading operations file, it indicates which road network we're taking into consideration
@lru_cache()
def get_active_ops() -> str:
    try:
        with open(f"{active_ops_filename}.txt", "r", encoding="utf-8") as ops_file:
            return ops_file.read()
    except FileNotFoundError:
        print("\033[91mOperations file not found\033[0m")
        sys.exit(1)


# TODO TO IMPLEMENT
def del_active_ops_file() -> None:
    try:
        os.remove(f"{active_ops_filename}.txt")
    except FileNotFoundError:
        print("\033[91mCurrent Operation File Not Found\033[0m")
    return None


#TODO TO OPTMIZE
# If the user wants to create a new operation, this function will be called
def create_ops_folder(ops_name: str) -> None:
    ops_name = clean_text(ops_name)
    os.makedirs(f"{ops_folder}/{ops_name}", exist_ok=True)

    write_metainfo(ops_name)

    main_folders = ["data", "eda", "rn_graph", "ml"]
    data_subfolders = [
        "traffic_volumes",
        "average_speed",
        "travel_times",
        "trp_metadata",
    ]
    data_sub_subfolders = ["raw", "clean"]  # To isolate raw data from the clean one
    eda_subfolders = [f"{ops_name}_shapiro_wilk_test", f"{ops_name}_plots"]
    eda_sub_subfolders = ["traffic_volumes", "avg_speeds"]
    rn_graph_subfolders = [
        f"{ops_name}_edges",
        f"{ops_name}_arches",
        f"{ops_name}_graph_analysis",
        f"{ops_name}_shortest_paths",
    ]
    ml_subfolders = ["models_parameters", "models", "models_performance", "ml_reports"]
    ml_sub_subfolders = ["traffic_volumes", "average_speed"]
    ml_sub_sub_subfolders = [road_category for road_category in ["E", "R", "F", "K", "P"]]

    with open(f"{ops_folder}/{ops_name}/{metainfo_filename}.json", "r", encoding="utf-8") as m:
        metainfo = json.load(m)
    metainfo["folder_paths"] = {}  # Setting/resetting the folders path dictionary to either write it for the first time or reset the previous one to adapt it with new updated folders, paths, etc.

    for mf in main_folders:
        main_f = f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_{mf}/"
        os.makedirs(main_f, exist_ok=True)
        metainfo["folder_paths"][mf] = {}

    # Data subfolders
    for dsf in data_subfolders:
        data_sub = f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_data/{dsf}/"
        os.makedirs(data_sub, exist_ok=True)
        metainfo["folder_paths"]["data"][dsf] = {"path": data_sub, "subfolders": {}}

        # Data sub-subfolders
        for dssf in data_sub_subfolders:
            if dsf != "trp_metadata":
                data_2sub = f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_data/{dsf}/{dssf}_{dsf}/"
                os.makedirs(data_2sub, exist_ok=True)
                metainfo["folder_paths"]["data"][dsf]["subfolders"][dssf] = {"path": data_2sub}

    for e in eda_subfolders:
        eda_sub = f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_eda/{e}/"
        os.makedirs(eda_sub, exist_ok=True)
        metainfo["folder_paths"]["eda"][e] = {"path": eda_sub, "subfolders": {}}

        for esub in eda_sub_subfolders:
            if e != f"{ops_name}_shapiro_wilk_test":
                eda_2sub = f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_eda/{e}/{esub}_eda_plots/"
                os.makedirs(eda_2sub, exist_ok=True)
                metainfo["folder_paths"]["eda"][e]["subfolders"][esub] = {"path": eda_2sub}

    # Graph subfolders
    for gsf in rn_graph_subfolders:
        gsf_sub = f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_rn_graph/{gsf}/"
        os.makedirs(gsf_sub, exist_ok=True)
        metainfo["folder_paths"]["rn_graph"][gsf] = {"path": gsf_sub, "subfolders": None}

    # Machine learning subfolders
    for mlsf in ml_subfolders:
        ml_sub = f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_ml/{ops_name}_{mlsf}/"
        os.makedirs(ml_sub, exist_ok=True)
        metainfo["folder_paths"]["ml"][mlsf] = {"path": ml_sub, "subfolders": {}}

        # Machine learning sub-subfolders
        for mlssf in ml_sub_subfolders:
            ml_2sub = f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_ml/{ops_name}_{mlsf}/{ops_name}_{mlssf}_{mlsf}/"
            os.makedirs(ml_2sub, exist_ok=True)
            metainfo["folder_paths"]["ml"][mlsf]["subfolders"][mlssf] = {"path": ml_2sub,"subfolders": {}}

            for mlsssf in ml_sub_sub_subfolders:
                ml_3sub = f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_ml/{ops_name}_{mlsf}/{ops_name}_{mlssf}_{mlsf}/{ops_name}_{mlsssf}_{mlssf}_{mlsf}/"
                os.makedirs(ml_3sub, exist_ok=True)
                metainfo["folder_paths"]["ml"][mlsf]["subfolders"][mlssf]["subfolders"][mlsssf] = {"path": ml_3sub}

    with open(f"{ops_folder}/{ops_name}/{metainfo_filename}.json", "w", encoding="utf-8") as m:
        json.dump(metainfo, m, indent=4)

    return None


# TODO TO IMPLEMENT
def del_ops_folder(ops_name: str) -> None:
    try:
        os.rmdir(ops_name)
        print(f"{ops_name} Operation Folder Deleted")
    except FileNotFoundError:
        print("\033[91mOperation Folder Not Found\033[0m")
    return None


# ------------ Metainfo File ------------

def write_metainfo(ops_name: str) -> None:
    target_folder = f"{ops_folder}/{ops_name}/"
    assert os.path.isdir(target_folder) is True, f"{target_folder} folder not found. Have you created the operation first?"

    metainfo = {
        "common": {
            "n_raw_traffic_volumes": None,
            "n_clean_traffic_volumes": None,
            "n_raw_average_speeds": None,
            "n_clean_average_speeds": None,
            "raw_volumes_size": None,
            "clean_volumes_size": None,
            "raw_average_speeds_size": None,
            "clean_average_speeds_size": None,
            "traffic_registration_points_file": f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_data/traffic_registration_points.json"
        },
        "traffic_volumes": {
            "start_date_iso": None,  # The start date which was inserted in the download section of the menu in ISO format
            "end_date_iso": None,  # The end date which was inserted in the download section of the menu in ISO format
            "start_year": None,  # The year obtained from the start_date_iso datetime
            "start_month": None,  # The month obtained from the start_date_iso datetime
            "start_day": None,  # The day obtained from the start_date_iso datetime
            "start_hour": None,  # The hour obtained from the start_date_iso datetime
            "end_year": None,  # The year obtained from the end_date_iso datetime
            "end_month": None,  # The month obtained from the end_date_iso datetime
            "end_day": None,  # The day obtained from the end_date_iso datetime
            "end_hour": None,  # The hour obtained from the end_date_iso datetime
            "n_days": None,  # The total number of days which we have data about
            "n_months": None,  # The total number of months which we have data about
            "n_years:": None,  # The total number of years which we have data about
            "n_weeks": None,  # The total number of weeks which we have data about
            "raw_filenames": [],  # The list of raw traffic volumes file names
            "clean_filenames": [],  # The list of clean traffic volumes file names
            "n_rows": [],  # The total number of records downloaded (clean volumes)
            "raw_volumes_start_date": None,  # The first date available for raw volumes files
            "raw_volumes_end_date": None,  # The last date available for raw volumes files
            "clean_volumes_start_date": None,  # The first date available for clean volumes files
            "clean_volumes_end_date": None  # The last date available for clean volumes files
        },
        "average_speeds": {
            "start_date_iso": None,  # The start date which was inserted in the download section of the menu in ISO format
            "end_date_iso": None,  # The end date which was inserted in the download section of the menu in ISO format
            "start_year": None,  # The year obtained from the start_date_iso datetime
            "start_month": None,  # The month obtained from the start_date_iso datetime
            "start_day": None,  # The day obtained from the start_date_iso datetime
            "start_hour": None,  # The hour obtained from the start_date_iso datetime
            "end_year": None,  # The year obtained from the end_date_iso datetime
            "end_month": None,  # The month obtained from the end_date_iso datetime
            "end_day": None,  # The day obtained from the end_date_iso datetime
            "end_hour": None,  # The hour obtained from the end_date_iso datetime
            "n_days": None,  # The total number of days which we have data about
            "n_months": None,  # The total number of months which we have data about
            "n_years": None,  # The total number of years which we have data about
            "n_weeks": None,  # The total number of weeks which we have data about
            "raw_filenames": [],  # The list of raw average speed file names
            "clean_filenames": [],  # The list of clean average speed file names
            "n_rows": [],  # The total number of records downloaded (clean average speeds)
            "raw_avg_speed_start_date": None,  # The first date available for raw average speed files
            "raw_avg_speed_end_date": None,  # The last date available for raw average speed files
            "clean_avg_speed_start_date": None,  # The first date available for clean average speed files
            "clean_avg_speed_end_date": None  # The last date available for clean average speed files
        },
        "metadata_files": [],
        "folder_paths": {},
        "forecasting": {"target_datetimes": {"V": None, "AS": None}},
        "by_trp_id": {
            "trp_ids": {}  # TODO ADD IF A RAW FILE HAS A CORRESPONDING CLEAN ONE (FOR BOTH TV AND AVG SPEEDS)
        }
    }

    with open(target_folder + metainfo_filename + ".json", "w", encoding="utf-8") as tf:
            json.dump(metainfo, tf, indent=4)

    return None


def check_metainfo_file() -> bool:
    return os.path.isfile(f"{cwd}/{ops_folder}/{get_active_ops()}/metainfo.json")  # Either True (if file exists) or False (in case the file doesn't exist)


def update_metainfo(value: Any, keys_map: list, mode: str) -> None:
    """
    This function inserts data into a specific right key-value pair in the metainfo.json file of the active operation.

    Parameters:
        value: the value which we want to insert or append for a specific key-value pair
        keys_map: the list which includes all the keys which bring to the key-value pair to update or to append another value to (the last key value pair has to be included).
                  The elements in the list must be ordered in which the keys are located in the metainfo dictionary
        mode: the mode which we intend to use for a specific operation on the metainfo file. For example: we may want to set a value for a specific key, or we may want to append another value to a list (which is the value of a specific key-value pair)
    """
    metainfo_filepath = f"{cwd}/{ops_folder}/{get_active_ops()}/metainfo.json"
    modes = ["equals", "append"]

    if check_metainfo_file() is True:
        with open(metainfo_filepath, "r", encoding="utf-8") as m:
            payload = json.load(m)
    else:
        raise FileNotFoundError(f'Metainfo file for "{get_active_ops()}" operation not found')

    # metainfo = payload has a specific reason to exist
    # This is how we preserve the whole original dictionary (loaded from the JSON file), but at the same time iterate over its keys and updating them
    # By doing to we'll assign the value (obtained from the value parameter of this method) to the right key, but preserving the rest of the dictionary
    metainfo = payload

    if mode == "equals":
        for key in keys_map[:-1]:
            metainfo = metainfo[key]
        metainfo[keys_map[-1]] = value  # Updating the metainfo file key-value pair
        with open(metainfo_filepath, "w", encoding="utf-8") as m:
            json.dump(payload, m, indent=4)
    elif mode == "append":
        for key in keys_map[:-1]:
            metainfo = metainfo[key]
        metainfo[keys_map[-1]].append(value)  # Appending a new value to the list (which is the value of this key-value pair)
        with open(metainfo_filepath, "w", encoding="utf-8") as m:
            json.dump(payload, m, indent=4)
    elif mode not in modes:
        print("\033[91mWrong mode\033[0m")
        sys.exit(1)

    return None


async def update_metainfo_async(value: Any, keys_map: list, mode: str) -> None:
    """
    This function is the asynchronous version of the update_metainfo() one. It inserts data into a specific right key-value pair in the metainfo.json file of the active operation.

    Parameters:
        value: the value which we want to insert or append for a specific key-value pair
        keys_map: the list which includes all the keys which bring to the key-value pair to update or to append another value to (the last key value pair has to be included).
                  The elements in the list must be ordered in which the keys are located in the metainfo dictionary
        mode: the mode which we intend to use for a specific operation on the metainfo file. For example: we may want to set a value for a specific key, or we may want to append another value to a list (which is the value of a specific key-value pair)
    """
    metainfo_filepath = f"{cwd}/{ops_folder}/{get_active_ops()}/metainfo.json"
    modes = ["equals", "append"]

    async with metainfo_lock:
        if check_metainfo_file() is True:
            async with aiofiles.open(metainfo_filepath, "r") as m:
                content = await m.read()
                payload = json.loads(content)
        else:
            raise FileNotFoundError(f'Metainfo file for "{get_active_ops()}" operation not found')

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

#TODO TRY CACHING THIS FUNCTION (DO TESTS BEFORE)
def read_metainfo_key(keys_map: list) -> Any:
    """
    This function reads data from a specific key-value pair in the metainfo.json file of the active operation.

    Parameters:
        keys_map: the list which includes all the keys which bring to the key-value pair to read (the one to read included)
    """

    if check_metainfo_file() is True:
        with open(f"{cwd}/{ops_folder}/{get_active_ops()}/metainfo.json", "r", encoding="utf-8") as m:
            payload = json.load(m)
    else:
        raise FileNotFoundError(f'Metainfo file for "{get_active_ops()}" operation not found')

    for key in keys_map[:-1]:
        payload = payload[key]

    return payload[keys_map[-1]]  # Returning the metainfo key-value pair


# ==================== Auxiliary Utilities ====================


def split_data(data: dd.DataFrame, target: str) -> tuple[dd.DataFrame, dd.DataFrame, dd.DataFrame, dd.DataFrame]:
    """
    Splits the Dask DataFrame into training and testing sets based on the target column.

    Parameters:
        data: dd.DataFrame
        target: str ("volume" or "mean_speed")

    Returns:
        X_train, X_test, y_train, y_test
    """

    if target not in ["volume", "mean_speed"]:
        raise ValueError("Wrong target variable in the split_data() function. Must be 'volume' or 'mean_speed'.")

    X = data.drop(columns=[target])
    y = data[[target]]

    # print("X shape: ", f"({len(X)}, {len(X.columns)})", "\n")
    # print("y shape: ", f"({len(y)}, {len(y.columns)})", "\n")

    n_rows = data.shape[0].compute()
    p_70 = int(n_rows * 0.70)

    X_train = dd.from_pandas(X.head(p_70)).persist()
    X_test = dd.from_pandas(X.tail(len(X) - p_70)).persist()

    # print(X_train.head(10))
    # print(X_test.head(10))

    y_train = dd.from_pandas(y.head(p_70)).persist()
    y_test = dd.from_pandas(y.tail(len(y) - p_70)).persist()

    # print(y_train.head(10))
    # print(y_test.head(10))

    # print(X_train.tail(5), "\n", X_test.tail(5), "\n", y_train.tail(5), "\n", y_test.tail(5), "\n")
    # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    return X_train, X_test, y_train, y_test


def merge(trp_filepaths: list[str]) -> dd.DataFrame:
    """
    Data merger function for traffic volumes or average speed data
    Parameters:
        trp_filepaths: a list of files to read data from
    """
    try:
        merged_data = dd.concat([dd.read_csv(trp) for trp in trp_filepaths], axis=0)
        merged_data = merged_data.repartition(partition_size="512MB")
        merged_data = merged_data.sort_values(["date"], ascending=True)  # Sorting records by date
        merged_data = merged_data.persist()
        return merged_data
    except ValueError as e:
        print(f"\033[91mNo data to concatenate. Error: {e}")
        sys.exit(1)


def check_datetime(dt: str) -> bool:
    try:
        datetime.strptime(dt, dt_format)
        return True
    except ValueError:
        return False


@lru_cache()
def get_eda_plots_folder_path(sub: str = None) -> str:
    folder_paths = {
        "volumes":read_metainfo_key(keys_map=["folder_paths", "eda", f"{get_active_ops()}_plots", "traffic_volumes_eda_plots"]),
        "avg_speeds":read_metainfo_key(keys_map=["folder_paths", "eda", f"{get_active_ops()}_plots", "avg_speeds_eda_plots"])
    }
    if sub is not None and folder_paths[sub]:
        return folder_paths[sub]
    elif sub is None:
        return read_metainfo_key(keys_map=["folder_paths", "eda", f"{get_active_ops()}_plots"])
    else:
        raise Exception("Wrong plots path")


def ZScore(df: dd.DataFrame, column: str) -> dd.DataFrame:
    df["z_score"] = (df[column] - df[column].mean()) / df[column].std()
    return df[(df["z_score"] > -3) & (df["z_score"] < 3)].drop(columns="z_score").persist()


def get_24_hours() -> Generator[str, None, None]:
    return (f"{i:02}" for i in range(24))


def get_covid_years() -> list[int]:
    return [2020, 2021, 2022]


def clean_text(text: str) -> str:
    return clean(text, no_emoji=True, no_currency_symbols=True).replace(" ", "_").lower()


def retrieve_n_ml_cpus() -> int:
    return int(os.cpu_count() * 0.75) # To avoid crashing while executing parallel computing in the GridSearchCV algorithm
    # The value multiplied with the n_cpu values shouldn't be above .80, otherwise processes could crash during execution


# ==================== *** Road Network Utilities *** ====================

# ==================== Edges Utilities ====================


def retrieve_edges() -> gpd.GeoDataFrame:
    with open(f"{read_metainfo_key(['folder_paths', 'rn_graph', f'{get_active_ops()}_edges', 'path'])}/traffic-nodes-2024_2025-02-28.geojson", "r", encoding="utf-8") as e:
        return geojson.load(e)["features"]


# ==================== Links Utilities ====================


def retrieve_arches() -> gpd.GeoDataFrame:
    with open(f"{read_metainfo_key(['folder_paths', 'rn_graph', f'{get_active_ops()}_arches', 'path'])}/traffic_links_2024_2025-02-27.geojson", "r", encoding="utf-8") as a:
        return geojson.load(a)["features"]



# ==================== TrafficRegistrationPoints Utilities ====================
