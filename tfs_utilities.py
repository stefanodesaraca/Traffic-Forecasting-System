from datetime import datetime
import os
import json
import pandas as pd
import dask.dataframe as dd
from tfs_ops_settings import *

cwd = os.getcwd()
ops_folder = "ops"


# ==================== Ops Utilities ====================

def get_active_ops_name() -> str:
    ops_name = read_active_ops_file()
    return ops_name


# ==================== TRP Utilities ====================

def import_TRPs_data():
    '''
    This function returns json data about all TRPs (downloaded previously)
    '''
    traffic_registration_points_path = get_traffic_registration_points_file_path()
    with open(traffic_registration_points_path, "r") as TRPs:
        trp_info = json.load(TRPs)

    return trp_info


def get_trp_id_list() -> list:

    traffic_registration_points_path = get_traffic_registration_points_file_path()
    with open(traffic_registration_points_path, "r") as TRPs:
        trp_info = json.load(TRPs)

    trp_id_list = [trp["id"] for trp in trp_info["trafficRegistrationPoints"]]

    return trp_id_list


def get_trp_road_category(trp_id: str) -> str:
    road_category = get_trp_metadata(trp_id)["road_category"]
    return road_category


def get_traffic_registration_points_file_path() -> str:
    '''
    This function returns the path to the traffic_measurement_points.json file which contains all TRPs' data (downloaded previously)
    '''
    ops_name = get_active_ops_name()
    traffic_registration_points_path = f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_data/traffic_registration_points.json"
    return traffic_registration_points_path


def get_trp_metadata(trp_id: str) -> dict:

    ops_name = get_active_ops_name()
    trp_metadata_file = f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_data/trp_metadata/{trp_id}_metadata"

    with open(trp_metadata_file, "r") as json_trp_metadata:
        trp_metadata = json.load(json_trp_metadata)

    return trp_metadata


def write_trp_metadata(trp_id: str) -> None:

    ops_name = get_active_ops_name()
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

    metadata = {"trp_id": trp_data["id"],
                "name": trp_data["name"],
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
                "number_of_data_nodes": len(volumes["trafficData"]["volume"]["byHour"]["edges"])}

    with open(trp_metadata_filepath + trp_metadata_filename + ".json", "w") as json_metadata:
        json.dump(metadata, json_metadata, indent=4)

    return None


def retrieve_trp_road_category(trp_id: str) -> str:
    trp_road_category = get_trp_metadata(trp_id)["road_category"]
    return trp_road_category


# ==================== Volumes Utilities ====================

def get_raw_traffic_volumes_folder_path() -> str:
    '''
    This function returns the path to the raw_traffic_volumes folder where all the raw traffic volume files are located
    '''
    ops_name = get_active_ops_name()
    raw_traffic_volumes_folder_path = f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_data/traffic_volumes/raw_traffic_volumes/"
    return raw_traffic_volumes_folder_path


def get_clean_traffic_volumes_folder_path() -> str:
    '''
    This function returns the path for the clean_traffic_volumes folder where all the cleaned traffic volumes data files are located
    '''
    ops_name = get_active_ops_name()
    clean_traffic_volumes_folder_path = f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_data/traffic_volumes/clean_traffic_volumes/"
    return clean_traffic_volumes_folder_path


def get_raw_traffic_volume_file_list() -> list:
    '''
    This function returns the name of every file contained in the raw_traffic_volumes folder, so every specific TRP's volumes
    '''

    ops_name = get_active_ops_name()
    traffic_volumes_folder_path = f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_data/traffic_volumes/raw_traffic_volumes/"

    # Identifying all the raw traffic volume files
    volume_files = os.listdir(traffic_volumes_folder_path)
    #print("Raw traffic volumes files: ", volume_files, "\n\n")

    return volume_files


def import_volumes_data(file):
    '''
    This function returns json traffic volumes data about a specific TRP
    '''
    with open(file, "r") as f:
        data = json.load(f)

    return data


def retrieve_theoretical_hours_columns() -> list:
    hours = [f"{i:02}" for i in range(24)]
    return hours


def get_clean_volume_files_list() -> list:

    clean_traffic_volumes_folder_path = get_clean_traffic_volumes_folder_path()

    clean_traffic_volumes = [clean_traffic_volumes_folder_path + vf for vf in os.listdir(get_clean_traffic_volumes_folder_path())]
    print("Clean traffic volumes files: ", clean_traffic_volumes)

    return clean_traffic_volumes


# ==================== Average Speed Utilities ====================

def get_raw_average_speed_folder_path() -> str:
    '''
    This function returns the path for the raw_average_speed folder where all the average speed files are located. Each file contains the average speeds for one TRP
    '''
    ops_name = get_active_ops_name()
    average_speed_folder_path = f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_data/average_speed/raw_average_speed/"

    return average_speed_folder_path


def get_clean_average_speed_folder_path() -> str:
    '''
    This function returns the path for the clean_average_speed folder where all the cleaned average speed data files are located
    '''
    ops_name = get_active_ops_name()
    clean_average_speed_folder_path = f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_data/average_speed/clean_average_speed/"
    return clean_average_speed_folder_path


def get_raw_avg_speed_file_list() -> list:
    '''
    This function returns the name of every file contained in the raw_average_speed folder
    '''
    ops_name = get_active_ops_name()
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
    files = [get_clean_average_speed_folder_path() + f for f in os.listdir(get_clean_average_speed_folder_path())]
    return files


# ==================== ML Related Utilities ====================

def get_ml_models_folder_path(target: str) -> str:

    ops_name = get_active_ops_name()

    if target == "volume":
        ml_folder_path = f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_ml/{ops_name}_models/{ops_name}_traffic_volumes_models/"
    elif target == "mean_speed":
        ml_folder_path = f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_ml/{ops_name}_models/{ops_name}_average_speed_models/"
    else:
        raise Exception("Wrong target variable in the get_ml_models_folder_path() function")

    return ml_folder_path


def get_ml_model_parameters_folder_path(target: str) -> str:

    ops_name = get_active_ops_name()

    if target == "volume":
        ml_parameters_folder_path = f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_ml/{ops_name}_models_parameters/{ops_name}_traffic_volumes_models_parameters/"
    elif target == "mean_speed":
        ml_parameters_folder_path = f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_ml/{ops_name}_models_parameters/{ops_name}_average_speed_models_parameters/"
    else:
        raise Exception("Wrong target variable in the get_ml_model_parameters_folder_path() function")

    return ml_parameters_folder_path


# ==================== Auxiliary Utilities ====================

def check_datetime(dt: str):

    try:
        datetime.strptime(dt, "%Y-%m-%dT%H:%M:%S")
        return True
    except ValueError:
        return False


def get_shapiro_wilk_plots_path() -> str:

    ops_name = get_active_ops_name()

    return f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_eda/{ops_name}_shapiro_wilk_test/"


def get_eda_plots_folder_path(sub: str = None) -> str:

    ops_name = get_active_ops_name()

    if sub == "volumes":
        return f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_eda/{ops_name}_plots/traffic_volumes_eda_plots/"

    elif sub == "avg_speeds":
        return f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_eda/{ops_name}_plots/avg_speeds_eda_plots/"

    elif sub is None:
        return f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_eda/{ops_name}_plots/"

    else:
        raise Exception("Wrong plots path")


def ZScore(df: [pd.DataFrame | dd.DataFrame], column: str) -> [pd.DataFrame | dd.DataFrame]:
    
    df["z_score"] = (df[column] - df[column].mean()) / df[column].std()

    #print("Number of outliers: ", len(df[(df["z_score"] < -3) & (df["z_score"] > 3)]))
    #print("Length before: ", len(df))

    filtered_df = df[(df["z_score"] > -3) & (df["z_score"] < 3)]
    filtered_df = filtered_df.drop(columns="z_score")

    #print("Length after: ", len(filtered_df))

    return filtered_df


def merge_volumes_data(trp_filepaths_list: list, return_pandas: bool) -> [dd.DataFrame | pd.DataFrame]:

    if return_pandas is False:
        dataframes_list = [dd.read_csv(trp) for trp in trp_filepaths_list]
        merged_data = dd.concat(dataframes_list, axis=0)
        merged_data = merged_data.sort_values(["year", "month", "day"], ascending=True)
    else:
        dataframes_list = [pd.read_csv(trp) for trp in trp_filepaths_list]
        merged_data = pd.concat(dataframes_list, axis=0)
        merged_data = merged_data.sort_values(["year", "month", "day"], ascending=True)

    return merged_data


def merge_avg_speed_data(trp_filepaths_list: list, return_pandas: bool) -> [dd.DataFrame | pd.DataFrame]:

    if return_pandas is False:
        dataframes_list = [dd.read_csv(trp) for trp in trp_filepaths_list]
        merged_data = dd.concat(dataframes_list, axis=0)
        merged_data = merged_data.sort_values(["year", "month", "day"], ascending=True)
    else:
        dataframes_list = [pd.read_csv(trp) for trp in trp_filepaths_list]
        merged_data = pd.concat(dataframes_list, axis=0)
        merged_data = merged_data.sort_values(["year", "month", "day"], ascending=True)

    return merged_data





