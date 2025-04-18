from datetime import datetime
import os
import json
import pandas as pd
import dask.dataframe as dd
from cleantext import clean
from typing import Any
import geopandas as gpd
import geojson
from geopandas import GeoDataFrame
import pprint

pd.set_option("display.max_columns", None)


cwd = os.getcwd()
ops_folder = "ops"
dt_format = "%Y-%m-%dT%H"  #Datetime format, the hour (H) must be zero-padded and 24-h base, for example: 01, 02, ..., 12, 13, 14, 15, etc.
#In this case we'll only ask for the hour value since, for now, it's the maximum granularity for the predictions we're going to make
metainfo_filename = "metainfo"
target_data = ["V", "AS"]
active_ops_filename = "active_ops"

# ==================== TRP Utilities ====================

def import_TRPs_data():
    '''
    This function returns json data about all TRPs (downloaded previously)
    '''
    assert os.path.isfile(get_traffic_registration_points_file_path()) is True, "Traffic registration points file missing"
    traffic_registration_points_path = get_traffic_registration_points_file_path()
    with open(traffic_registration_points_path, "r") as TRPs:
        trp_info = json.load(TRPs)

    return trp_info

#TODO IMPROVE THIS FUNCTION, AND SET THAT THIS RETURNS A LIST OF str
def get_trp_id_list() -> list[str]:

    trp_info = import_TRPs_data()

    #If there aren't volumes files yet, then just return all the TRP IDs available in the traffic_registration_points_file
    if len(os.listdir(get_raw_traffic_volumes_folder_path())) == 0:
        trp_id_list = [trp["id"] for trp in trp_info["trafficRegistrationPoints"]] #This list may contain IDs of TRPs which don't have a volumes file associated with them
    else:
        trp_id_list = [trp["id"] for trp in trp_info["trafficRegistrationPoints"] if trp["id"] in [get_trp_id_from_filename(file) for file in os.listdir(get_raw_traffic_volumes_folder_path())]] #Keep only the TRPs which actually have a volumes file associated with them

    return trp_id_list


def get_trp_id_from_filename(filename: str) -> str:
    return filename.split("_")[0]


def get_all_available_road_categories() -> list:
    trp_info = import_TRPs_data()
    trp_road_category_list = list(set([trp["location"]["roadReference"]["roadCategory"]["id"] for trp in trp_info["trafficRegistrationPoints"]]))
    return trp_road_category_list


def get_trp_road_category(trp_id: str) -> str:
    return get_trp_metadata(trp_id)["road_category"]


def get_traffic_registration_points_file_path() -> str:
    '''
    This function returns the path to the traffic_measurement_points.json file which contains all TRPs' data (downloaded previously)
    '''
    ops_name = get_active_ops()
    traffic_registration_points_path = f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_data/traffic_registration_points.json"
    return traffic_registration_points_path


def get_trp_metadata(trp_id: str) -> dict:

    ops_name = get_active_ops()
    trp_metadata_file = f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_data/trp_metadata/{trp_id}_metadata.json"

    with open(trp_metadata_file, "r") as json_trp_metadata:
        trp_metadata = json.load(json_trp_metadata)

    return trp_metadata


def write_trp_metadata(trp_id: str) -> None:

    ops_name = get_active_ops()
    trps = import_TRPs_data()
    trp_data = [i for i in trps["trafficRegistrationPoints"] if i["id"] == trp_id][0]

    raw_volume_files_folder_path = get_raw_traffic_volumes_folder_path()
    raw_volume_files = get_raw_traffic_volume_file_list()
    trp_volumes_file = [f for f in raw_volume_files if trp_id in f] #Find the right TRP file by checking if the TRP ID is in the filename and if it is in the raw traffic volumes folder

    if len(trp_volumes_file) == 1:
        trp_volumes_file = trp_volumes_file[0]
    else: return None


    with open(raw_volume_files_folder_path + trp_volumes_file, "r") as f: volumes = json.load(f)

    trp_metadata_filepath = f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_data/trp_metadata/"
    trp_metadata_filename = f"{trp_id}_metadata"

    assert os.path.isdir(get_raw_traffic_volumes_folder_path()) is True, "Raw traffic volumes folder missing"
    assert os.path.isdir(get_clean_traffic_volumes_folder_path()) is True, "Clean traffic volumes folder missing"

    assert os.path.isdir(get_raw_average_speed_folder_path()) is True, "Raw average speed folder missing"
    assert os.path.isdir(get_clean_average_speed_folder_path()) is True, "Clean average speed folder missing"

    check_metainfo_file()

    metadata = {"trp_id": trp_data["id"],
                "name": trp_data["name"],
                "raw_volumes_filepath": f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_data/traffic_volumes/raw_traffic_volumes/{[file for file in os.listdir(get_raw_traffic_volumes_folder_path()) if trp_id in file][0] if len([file for file in os.listdir(get_raw_traffic_volumes_folder_path()) if trp_id in file]) != 0 else ''}",
                "clean_volumes_filepath": f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_data/traffic_volumes/clean_traffic_volumes/{[file for file in os.listdir(get_clean_traffic_volumes_folder_path()) if trp_id in file][0] if len([file for file in os.listdir(get_clean_traffic_volumes_folder_path()) if trp_id in file]) != 0 else ''}",
                "raw_average_speed_filepath": f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_data/average_speed/raw_average_speed/{[file for file in os.listdir(get_raw_average_speed_folder_path()) if trp_id in file][0] if len([file for file in os.listdir(get_raw_average_speed_folder_path()) if trp_id in file]) != 0 else ''}",
                "clean_average_speed_filepath": f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_data/average_speed/clean_average_speed/{[file for file in os.listdir(get_clean_average_speed_folder_path()) if trp_id in file][0] if len([file for file in os.listdir(get_clean_average_speed_folder_path()) if trp_id in file]) != 0 else ''}",
                "road_category": trp_data["location"]["roadReference"]["roadCategory"]["id"],
                "lat": trp_data["location"]["coordinates"]["latLon"]["lat"],
                "lon": trp_data["location"]["coordinates"]["latLon"]["lon"],
                "county_name": trp_data["location"]["county"]["name"],
                "county_number": trp_data["location"]["county"]["number"],
                "geographic_number": trp_data["location"]["county"]["geographicNumber"],
                "country_part": trp_data["location"]["county"]["countryPart"]["name"],
                "municipality_name": trp_data["location"]["municipality"]["name"],
                "traffic_registration_type": trp_data["trafficRegistrationType"],
                "first_data": trp_data["dataTimeSpan"]["firstData"],
                "first_data_with_quality_metrics": trp_data["dataTimeSpan"]["firstDataWithQualityMetrics"],
                "latest_volume_by_day": trp_data["dataTimeSpan"]["latestData"]["volumeByDay"],
                "latest_volume_byh_hour": trp_data["dataTimeSpan"]["latestData"]["volumeByHour"],
                "latest_volume_average_daily_by_year": trp_data["dataTimeSpan"]["latestData"]["volumeAverageDailyByYear"],
                "latest_volume_average_daily_by_season": trp_data["dataTimeSpan"]["latestData"]["volumeAverageDailyBySeason"],
                "latest_volume_average_daily_by_month": trp_data["dataTimeSpan"]["latestData"]["volumeAverageDailyByMonth"],
                "number_of_data_nodes": len(volumes["trafficData"]["volume"]["byHour"]["edges"])} #(Volumes data nodes)

    metadata_filepath = trp_metadata_filepath + trp_metadata_filename + ".json"
    with open(metadata_filepath, "w") as json_metadata:
        json.dump(metadata, json_metadata, indent=4)

    update_metainfo(trp_metadata_filename, ["metadata_filenames"], "append")
    update_metainfo(metadata_filepath, ["metadata_filepaths"], "append")

    return None


def retrieve_trp_road_category(trp_id: str) -> str:
    trp_road_category = get_trp_metadata(trp_id)["road_category"]
    return trp_road_category


def retrieve_trp_clean_volumes_filepath_by_id(trp_id: str):
    return get_trp_metadata(trp_id)["clean_volumes_filepath"]


def retrieve_trp_clean_average_speed_filepath_by_id(trp_id: str):
    return get_trp_metadata(trp_id)["clean_average_speed_filepath"]


# ==================== Volumes Utilities ====================

def get_raw_traffic_volumes_folder_path() -> str:
    '''
    This function returns the path to the raw_traffic_volumes folder where all the raw traffic volume files are located
    '''
    ops_name = get_active_ops()
    raw_traffic_volumes_folder_path = f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_data/traffic_volumes/raw_traffic_volumes/"
    return raw_traffic_volumes_folder_path


def get_clean_traffic_volumes_folder_path() -> str:
    '''
    This function returns the path for the clean_traffic_volumes folder where all the cleaned traffic volumes data files are located
    '''
    ops_name = get_active_ops()
    clean_traffic_volumes_folder_path = f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_data/traffic_volumes/clean_traffic_volumes/"
    return clean_traffic_volumes_folder_path


def get_raw_traffic_volume_file_list() -> list:
    '''
    This function returns the name of every file contained in the raw_traffic_volumes folder, so every specific TRP's volumes
    '''

    ops_name = get_active_ops()
    traffic_volumes_folder_path = f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_data/traffic_volumes/raw_traffic_volumes/"

    # Identifying all the raw traffic volume files
    volume_files = os.listdir(traffic_volumes_folder_path)
    #print("Raw traffic volumes files: ", volume_files, "\n\n")

    return volume_files


def import_volumes_data(file):
    '''
    This function returns json traffic volumes data about a specific TRP
    '''
    with open(file, "r") as f: data = json.load(f)
    return data


def get_clean_volume_files_list() -> list:

    clean_traffic_volumes_folder_path = get_clean_traffic_volumes_folder_path()

    clean_traffic_volumes = [clean_traffic_volumes_folder_path + vf for vf in os.listdir(get_clean_traffic_volumes_folder_path())]
    print("Clean traffic volumes files: ", clean_traffic_volumes)

    return clean_traffic_volumes


def merge_volumes_data(trp_filepaths_list: list, road_category: str) -> dd.DataFrame:

    dataframes_list = [dd.read_csv(trp) for trp in trp_filepaths_list]
    merged_data = dd.concat(dataframes_list, axis=0)
    merged_data = merged_data.repartition(partition_size="512MB")
    merged_data = merged_data.sort_values(["year", "month", "day"], ascending=True)
    merged_data = merged_data.persist()
    #print(merged_data.head(10))
    #print(merged_data.dtypes)
    print(f"Shape of the merged volumes data for road category {road_category}: ", (merged_data.shape[0].compute(), merged_data.shape[1]))

    return merged_data


# ==================== Average Speed Utilities ====================

def get_raw_average_speed_folder_path() -> str:
    '''
    This function returns the path for the raw_average_speed folder where all the average speed files are located. Each file contains the average speeds for one TRP
    '''
    ops_name = get_active_ops()
    average_speed_folder_path = f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_data/average_speed/raw_average_speed/"

    return average_speed_folder_path


def get_clean_average_speed_folder_path() -> str:
    '''
    This function returns the path for the clean_average_speed folder where all the cleaned average speed data files are located
    '''
    ops_name = get_active_ops()
    clean_average_speed_folder_path = f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_data/average_speed/clean_average_speed/"
    return clean_average_speed_folder_path


def get_raw_avg_speed_file_list() -> list:
    '''
    This function returns the name of every file contained in the raw_average_speed folder
    '''
    ops_name = get_active_ops()
    average_speed_folder_path = f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_data/average_speed/raw_average_speed/"

    # Identifying all the raw average speed files
    average_speed_files = os.listdir(average_speed_folder_path)
    print("Average speed files: ", average_speed_files, "\n\n")

    return average_speed_files


def import_avg_speed_data(file_path: str) -> pd.DataFrame:
    '''
    This function returns the average speed data for a specific TRP
    '''
    data = pd.read_csv(file_path, sep=";", engine="c")
    return data


def get_clean_average_speed_files_list() -> list:
    return [get_clean_average_speed_folder_path() + f for f in os.listdir(get_clean_average_speed_folder_path())]


def merge_avg_speed_data(trp_filepaths_list: list) -> dd.DataFrame:

    dataframes_list = [dd.read_csv(trp) for trp in trp_filepaths_list]
    merged_data = dd.concat(dataframes_list, axis=0)
    merged_data = merged_data.sort_values(["year", "month", "day"], ascending=True)
    merged_data = merged_data.persist()

    return merged_data


# ==================== ML Related Utilities ====================

def get_ml_models_folder_path(target: str, road_category: str) -> str:

    ops_name = get_active_ops()

    if target == "volume":
        ml_folder_path = f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_ml/{ops_name}_models/{ops_name}_traffic_volumes_models/{ops_name}_{road_category}_traffic_volumes_models/"
    elif target == "mean_speed":
        ml_folder_path = f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_ml/{ops_name}_models/{ops_name}_average_speed_models/{ops_name}_{road_category}_average_speed_models/"
    else:
        raise Exception("Wrong target variable in the get_ml_models_folder_path() function")

    return ml_folder_path


def get_ml_model_parameters_folder_path(target: str, road_category: str) -> str:

    ops_name = get_active_ops()

    if target == "volume":
        ml_parameters_folder_path = f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_ml/{ops_name}_models_parameters/{ops_name}_traffic_volumes_models_parameters/{ops_name}_{road_category}_traffic_volumes_models_parameters/"
    elif target == "mean_speed":
        ml_parameters_folder_path = f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_ml/{ops_name}_models_parameters/{ops_name}_average_speed_models_parameters/{ops_name}_{road_category}_average_speed_models_parameters/"
    else:
        raise Exception("Wrong target variable in the get_ml_model_parameters_folder_path() function")

    return ml_parameters_folder_path


# ==================== Forecasting Settings Utilities ====================

#TODO FIND A WAY TO CHECK WHICH IS THE LAST DATETIME AVAILABLE FOR BOTH AVERAGE SPEED (CLEAN) AND TRAFFIC VOLUMES (CLEAN)

def write_forecasting_target_datetime() -> None:

    assert os.path.isdir(get_clean_traffic_volumes_folder_path()), "Clean traffic volumes folder missing. Initialize an operation first and then set a forecasting target datetime"
    assert os.path.isdir(get_clean_average_speed_folder_path()), "Clean average speeds folder missing. Initialize an operation first and then set a forecasting target datetime"

    option = str(input("Press V to set forecasting target datetime for traffic volumes or AS for average speeds: "))
    dt = str(input("Insert Target Datetime (YYYY-MM-DDTHH): ")) #The month number must be zero-padded, for example: 01, 02, etc.

    if check_datetime(dt) is True and option in target_data:
        update_metainfo(value=dt, keys_map=["forecasting", "target_datetimes", option], mode="equals")
        print("Target datetime set to: ", dt, "\n\n")
        return None
    else:
        if check_datetime(dt) is False:
            print("\033[91mWrong datetime format, try again\033[0m")
            exit(code=1)
        elif option not in target_data:
            print("\033[91mWrong data forecasting target datetime, try again\033[0m")
            exit(code=1)


def read_forecasting_target_datetime(data_kind: str) -> datetime:
    try:
        target_dt = read_metainfo_key(keys_map=["forecasting", "target_datetimes", data_kind])
        target_dt = datetime.strptime(target_dt, dt_format)
        return target_dt
    except TypeError:
        print(f"\033[91mTarget datetime for {data_kind} isn't set yet. Set it first and then execute a one-point forecast\033[0m")
        exit(code=1)
    except FileNotFoundError:
        print("\033[91mTarget Datetime File Not Found\033[0m")
        exit(code=1)


def rm_forecasting_target_datetime() -> None:
    try:
        print("For which data kind do you want to remove the forecasting target datetime?")
        option = input("Press V to set forecasting target datetime for traffic volumes or AS for average speeds:" )
        update_metainfo(None, ["forecasting", "target_datetimes", option], mode="equals")
        print("Target datetime file deleted successfully\n\n")
        return None
    except KeyError:
        print("\033[91mTarget datetime not found\033[0m")
        exit(code=1)


# ==================== Operations' Settings Utilities ====================

#The user sets the current operation
def write_active_ops_file(ops_name: str) -> None:
    ops_name = clean_text(ops_name)
    assert os.path.isfile(f"{ops_folder}/{ops_name}") is True, f"{ops_name} operation folder not found. Create an operation with that name first."
    with open(f"{active_ops_filename}.txt", "w") as ops_file: ops_file.write(ops_name)
    return None


#Reading operations file, it indicates which road network we're taking into consideration
def get_active_ops():
    try:
        with open(f"{active_ops_filename}.txt", "r") as ops_file: op = ops_file.read()
        return op
    except FileNotFoundError:
        print("\033[91mOperations file not found\033[0m")
        exit(code=1)


def del_active_ops_file() -> None:
    try:
        os.remove(f"{active_ops_filename}.txt")
    except FileNotFoundError:
        print("\033[91mCurrent Operation File Not Found\033[0m")
    return None


#If the user wants to create a new operation, this function will be called
def create_ops_folder(ops_name: str) -> None:

    ops_name = clean_text(ops_name)
    os.makedirs(f"{ops_folder}/{ops_name}", exist_ok=True)

    write_metainfo(ops_name)

    main_folders = [f"data", f"eda", f"rn_graph", f"ml"]
    data_subfolders = ["traffic_volumes", "average_speed", "travel_times", "trp_metadata"]
    data_sub_subfolders = ["raw", "clean"] #To isolate raw data from the clean one
    eda_subfolders = [f"{ops_name}_shapiro_wilk_test", f"{ops_name}_plots"]
    eda_sub_subfolders = ["traffic_volumes", "avg_speeds"]
    rn_graph_subfolders = [f"{ops_name}_edges", f"{ops_name}_arches", f"{ops_name}_graph_analysis", f"{ops_name}_shortest_paths"]
    ml_subfolders = ["models_parameters", "models", "models_performance", "ml_reports"]
    ml_sub_subfolders = ["traffic_volumes", "average_speed"]
    ml_sub_sub_subfolders = [road_category for road_category in ["E", "R", "F", "K", "P"]]

    with open(f"{ops_folder}/{ops_name}/{metainfo_filename}.json", "r") as m: metainfo = json.load(m)
    metainfo["folder_paths"] = {} #Setting/resetting the folders path dictionary to either write it for the first time or reset the previous one to adapt it with new updated folders, paths, etc.

    for mf in main_folders:
        main_f = f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_{mf}/"
        os.makedirs(main_f, exist_ok=True)
        metainfo["folder_paths"][mf] = {}

    # Data subfolders
    for dsf in data_subfolders:
        data_sub = f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_data/{dsf}/"
        os.makedirs(data_sub, exist_ok=True)
        metainfo["folder_paths"]["data"][dsf] = {"path": data_sub,
                                                "subfolders": {}}

        #Data sub-subfolders
        for dssf in data_sub_subfolders:
            if dsf != "trp_metadata":
                data_2sub = f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_data/{dsf}/{dssf}_{dsf}/"
                os.makedirs(data_2sub, exist_ok=True)
                metainfo["folder_paths"]["data"][dsf]["subfolders"][dssf] = {"path": data_2sub}

    for e in eda_subfolders:
        eda_sub = f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_eda/{e}/"
        os.makedirs(eda_sub, exist_ok=True)
        metainfo["folder_paths"]["eda"][e] = {"path": eda_sub,
                                              "subfolders": {}}

        for esub in eda_sub_subfolders:
            if e != f"{ops_name}_shapiro_wilk_test":
                eda_2sub = f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_eda/{e}/{esub}_eda_plots/"
                os.makedirs(eda_2sub, exist_ok=True)
                metainfo["folder_paths"]["eda"][e]["subfolders"][esub] = {"path": eda_2sub}

    # Graph subfolders
    for gsf in rn_graph_subfolders:
        gsf_sub = f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_rn_graph/{gsf}/"
        os.makedirs(gsf_sub, exist_ok=True)
        metainfo["folder_paths"]["rn_graph"][gsf] = {"path": gsf_sub,
                                                     "subfolders": None}

    # Machine learning subfolders
    for mlsf in ml_subfolders:
        ml_sub = f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_ml/{ops_name}_{mlsf}/"
        os.makedirs(ml_sub, exist_ok=True)
        metainfo["folder_paths"]["ml"][mlsf] = {"path": ml_sub,
                                                "subfolders": {}}

        #Machine learning sub-subfolders
        for mlssf in ml_sub_subfolders:
            ml_2sub = f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_ml/{ops_name}_{mlsf}/{ops_name}_{mlssf}_{mlsf}/"
            os.makedirs(ml_2sub, exist_ok=True)
            metainfo["folder_paths"]["ml"][mlsf]["subfolders"][mlssf] = {"path": ml_2sub,
                                                                         "subfolders": {}}

            for mlsssf in ml_sub_sub_subfolders:
                ml_3sub = f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_ml/{ops_name}_{mlsf}/{ops_name}_{mlssf}_{mlsf}/{ops_name}_{mlsssf}_{mlssf}_{mlsf}/"
                os.makedirs(ml_3sub, exist_ok=True)
                metainfo["folder_paths"]["ml"][mlsf]["subfolders"][mlssf]["subfolders"][mlsssf] = {"path": ml_3sub}

    with open(f"{ops_folder}/{ops_name}/{metainfo_filename}.json", "w") as m: json.dump(metainfo, m, indent=4)


    return None


def del_ops_folder(ops_name: str) -> None:
    try:
        os.rmdir(ops_name)
        print(f"{ops_name} Operation Folder Deleted")
    except FileNotFoundError:
        print("\033[91mOperation Folder Not Found\033[0m")
    return None


def write_metainfo(ops_name: str) -> None:

    target_folder = f"{ops_folder}/{ops_name}/"
    assert os.path.isdir(target_folder) is True, f"{target_folder} folder not found. Have you created the operation first?"

    if os.path.isdir(target_folder) is True:
        metainfo = {
            "common": {
                "n_raw_traffic_volumes": None,
                "n_clean_traffic_volumes": None,
                "n_raw_average_speeds": None,
                "n_clean_average_speeds": None,
                "raw_volumes_size": None,
                "clean_volumes_size": None,
                "raw_average_speeds_size": None,
                "clean_average_speeds_size": None
            },
            "traffic_volumes": {
                "start_date_iso": None, #The start date which was inserted in the download section of the menu in ISO format
                "end_date_iso": None, #The end date which was inserted in the download section of the menu in ISO format
                "start_year": None, #The year obtained from the start_date_iso datetime
                "start_month": None, #The month obtained from the start_date_iso datetime
                "start_day": None, #The day obtained from the start_date_iso datetime
                "start_hour": None, #The hour obtained from the start_date_iso datetime
                "end_year": None, #The year obtained from the end_date_iso datetime
                "end_month": None, #The month obtained from the end_date_iso datetime
                "end_day": None, #The day obtained from the end_date_iso datetime
                "end_hour": None, #The hour obtained from the end_date_iso datetime
                "n_days": None, #The total number of days which we have data about
                "n_months": None, #The total number of months which we have data about
                "n_years:": None, #The total number of years which we have data about
                "n_weeks": None, #The total number of weeks which we have data about
                "raw_filenames": [], #The list of raw traffic volumes file names
                "raw_filepaths": [], #The list of file raw traffic volumes file path
                "clean_filenames": [], #The list of clean traffic volumes file names
                "clean_filepaths": [], #The list of file clean traffic volumes file path
                "n_rows": [], #The total number of records downloaded (clean volumes)
                "raw_volumes_start_date": None, #The first date available for raw volumes files
                "raw_volumes_end_date": None, #The last date available for raw volumes files
                "clean_volumes_start_date": None, #The first date available for clean volumes files
                "clean_volumes_end_date": None #The last date available for clean volumes files
            },
            "average_speeds": {
                "start_date_iso": None, #The start date which was inserted in the download section of the menu in ISO format
                "end_date_iso": None, #The end date which was inserted in the download section of the menu in ISO format
                "start_year": None, #The year obtained from the start_date_iso datetime
                "start_month": None, #The month obtained from the start_date_iso datetime
                "start_day": None, #The day obtained from the start_date_iso datetime
                "start_hour": None, #The hour obtained from the start_date_iso datetime
                "end_year": None, #The year obtained from the end_date_iso datetime
                "end_month": None, #The month obtained from the end_date_iso datetime
                "end_day": None, #The day obtained from the end_date_iso datetime
                "end_hour": None, #The hour obtained from the end_date_iso datetime
                "n_days": None, #The total number of days which we have data about
                "n_months": None, #The total number of months which we have data about
                "n_years:": None, #The total number of years which we have data about
                "n_weeks": None, #The total number of weeks which we have data about
                "raw_filenames": [], #The list of raw average speed file names
                "raw_filepaths": [], #The list of file raw average speed file path
                "clean_filenames": [], #The list of clean average speed file names
                "clean_filepaths": [], #The list of file clean average speed file path
                "n_rows": [], #The total number of records downloaded (clean average speeds)
                "raw_avg_speed_start_date": None, #The first date available for raw average speed files
                "raw_avg_speed_end_date": None, #The last date available for raw average speed files
                "clean_avg_speed_start_date": None, #The first date available for clean average speed files
                "clean_avg_speed_end_date": None #The last date available for clean average speed files
            },
            "metadata_filenames": [],
            "metadata_filepaths": [],
            "folder_paths": {},
            "forecasting": {
                "target_datetimes": {
                    "V": None,
                    "AS": None
                }
            },
            "by_trp_id": {
                "trp_ids" : {} #TODO ADD IF A RAW FILE HAS A CORRESPONDING CLEAN ONE (FOR BOTH TV AND AVG SPEEDS)
            }
        }

        with open(target_folder + metainfo_filename + ".json", "w") as tf: json.dump(metainfo, tf, indent=4)

    return None


def check_metainfo_file() -> bool:
    return os.path.isfile(f"{cwd}/{ops_folder}/{get_active_ops()}/metainfo.json") #Either True (if file exists) or False (in case the file doesn't exist)


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
        with open(metainfo_filepath, "r") as m: payload = json.load(m)
    else:
        raise FileNotFoundError(f'Metainfo file for "{get_active_ops()}" operation not found')

    #metainfo = payload has a specific reason to exist
    #This is how we preserve the whole original dictionary (loaded from the JSON file), but at the same time iterate over its keys and updating them
    #By doing to we'll assign the value (obtained from the value parameter of this method) to the right key, but preserving the rest of the dictionary
    metainfo = payload

    if mode == "equals":
        for key in keys_map[:-1]: metainfo = metainfo[key]
        metainfo[keys_map[-1]] = value #Updating the metainfo file key-value pair
        with open(metainfo_filepath, "w") as m: json.dump(payload, m, indent=4)
    elif mode == "append":
        for key in keys_map[:-1]: metainfo = metainfo[key]
        metainfo[keys_map[-1]].append(value) #Appending a new value to the list (which is the value of this key-value pair)
        with open(metainfo_filepath, "w") as m: json.dump(payload, m, indent=4)
    elif mode not in modes:
        print("\033[91mWrong mode\033[0m")
        exit(code=1)

    return None


def read_metainfo_key(keys_map: list) -> Any:
    """
    This function reads data from a specific key-value pair in the metainfo.json file of the active operation.

    Parameters:
        keys_map: the list which includes all the keys which bring to the key-value pair to read (the one to read included)
    """
    metainfo_filepath = f"{cwd}/{ops_folder}/{get_active_ops()}/metainfo.json"

    if check_metainfo_file() is True:
        with open(metainfo_filepath, "r") as m: payload = json.load(m)
    else:
        raise FileNotFoundError(f'Metainfo file for "{get_active_ops()}" operation not found')

    for key in keys_map[:-1]: payload = payload[key]

    return payload[keys_map[-1]] #Returning the metainfo key-value pair


# ==================== Auxiliary Utilities ====================

def check_datetime(dt: str):
    try:
        datetime.strptime(dt, dt_format)
        return True
    except ValueError:
        return False


def get_shapiro_wilk_plots_path() -> str:
    ops_name = get_active_ops()
    return f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_eda/{ops_name}_shapiro_wilk_test/"


def get_eda_plots_folder_path(sub: str = None) -> str:

    ops_name = get_active_ops()

    if sub == "volumes":
        return f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_eda/{ops_name}_plots/traffic_volumes_eda_plots/"

    elif sub == "avg_speeds":
        return f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_eda/{ops_name}_plots/avg_speeds_eda_plots/"

    elif sub is None:
        return f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_eda/{ops_name}_plots/"

    else:
        raise Exception("Wrong plots path")


def ZScore(df: dd.DataFrame, column: str) -> dd.DataFrame:

    df["z_score"] = (df[column] - df[column].mean()) / df[column].std()

    #print("Number of outliers: ", len(df[(df["z_score"] < -3) & (df["z_score"] > 3)]))
    #print("Length before: ", len(df))

    filtered_df = df[(df["z_score"] > -3) & (df["z_score"] < 3)]
    filtered_df = filtered_df.drop(columns="z_score")

    #print("Length after: ", len(filtered_df))

    return filtered_df.persist()


def retrieve_theoretical_hours_columns() -> list:
    return [f"{i:02}" for i in range(24)]


def clean_text(text: str) -> str:
    text = clean(text, no_emoji=True, no_currency_symbols=True)
    text = text.replace(" ", "_")
    text = text.lower()
    return text


# ==================== *** Road Network Utilities *** ====================

# ==================== Edges Utilities ====================

def retrieve_edges() -> gpd.GeoDataFrame:

    active_ops = get_active_ops()
    edges_folder = read_metainfo_key(["folder_paths", "rn_graph", f"{active_ops}_edges", "path"])
    edges_filepath = f"{edges_folder}/traffic-nodes-2024_2025-02-28.geojson"

    with open(edges_filepath, "r") as e: edges = geojson.load(e)["features"]
    #pprint.pprint(edges, indent=6)
    return edges









# ==================== Links Utilities ====================

def retrieve_arches() -> gpd.GeoDataFrame:

    active_ops = get_active_ops()
    arches_folder = read_metainfo_key(["folder_paths", "rn_graph", f"{active_ops}_arches", "path"])
    arches_filepath = f"{arches_folder}/traffic_links_2024_2025-02-27.geojson"

    with open(arches_filepath, "r") as a: arches = geojson.load(a)["features"]
    #pprint.pprint(arches, indent=6)
    return arches








# ==================== TrafficRegistrationPoints Utilities ====================

























