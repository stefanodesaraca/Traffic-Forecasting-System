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
    return get_trp_metadata(trp_id)["road_category"]


def get_traffic_registration_points_file_path() -> str:
    '''
    This function returns the path to the traffic_measurement_points.json file which contains all TRPs' data (downloaded previously)
    '''
    ops_name = get_active_ops_name()
    traffic_registration_points_path = f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_data/traffic_registration_points.json"
    return traffic_registration_points_path


def get_trp_metadata(trp_id: str) -> dict:

    ops_name = get_active_ops_name()
    trp_metadata = f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_data/trp_metadata/{trp_id}_metadata"

    return trp_metadata


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


def get_traffic_volume_file_list() -> list:
    '''
    This function returns the name of every file contained in the raw_traffic_volumes folder, so every specific TRP's volumes
    '''

    ops_name = get_active_ops_name()
    traffic_volumes_folder_path = f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_data/traffic_volumes/raw_traffic_volumes/"

    # Identifying all the raw traffic volume files
    volume_files = os.listdir(traffic_volumes_folder_path)
    print("Raw traffic volumes files: ", volume_files, "\n\n")

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


def get_clean_volume_files() -> list:

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


# ==================== ML Related Utilities ====================


def get_ml_models_folder_path() -> str:

    ops_name = get_active_ops_name()

    ml_folder_path = f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_ml/{ops_name}_models/"

    return ml_folder_path


def get_ml_model_parameters_folder_path() -> str:

    ops_name = get_active_ops_name()

    ml_parameters_folder_path = f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_ml/{ops_name}_model_parameters/"

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











