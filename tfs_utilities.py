from datetime import datetime
import os
import json
from tfs_ops_settings import *

cwd = os.getcwd()
ops_folder = "ops"


def get_active_ops_name():
    ops_name = read_active_ops_file()
    return ops_name


def import_TRPs_info():
    traffic_registration_points_path = get_traffic_registration_points_file_path()
    with open(traffic_registration_points_path, "r") as TRPs:
        trp_info = json.load(TRPs)

        return trp_info


def import_volumes_data(file):
    with open(file, "r") as f:
        data = json.load(f)

    return data


def get_traffic_registration_points_file_path():
    ops_name = get_active_ops_name()
    traffic_registration_points_path = f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_data/traffic_measurement_points.json"
    return traffic_registration_points_path


def get_raw_traffic_volumes_folder_path():
    ops_name = get_active_ops_name()
    raw_traffic_volumes_folder_path = f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_data/traffic_volumes/raw_traffic_volumes/"
    return raw_traffic_volumes_folder_path


def get_clean_traffic_volumes_folder_path():
    ops_name = get_active_ops_name()
    clean_traffic_volumes_folder_path = f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_data/traffic_volumes/clean_traffic_volumes/"
    return clean_traffic_volumes_folder_path


def get_traffic_volume_file_list():

    ops_name = get_active_ops_name()
    traffic_volumes_folder_path = f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_data/traffic_volumes/raw_traffic_volumes/"

    # Identifying all the raw traffic volume files
    volume_files = os.listdir(traffic_volumes_folder_path)
    print("Raw traffic volumes files: ", volume_files, "\n\n")

    return volume_files


def get_raw_average_speed_folder_path():
    ops_name = get_active_ops_name()
    average_speed_folder_path = f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_data/average_speed/raw_average_speed/"

    return average_speed_folder_path


def get_raw_avg_speed_file_list():

    ops_name = get_active_ops_name()
    average_speed_folder_path = f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_data/average_speed/raw_average_speed/"

    # Identifying all the raw average speed files
    average_speed_files = os.listdir(average_speed_folder_path)
    print("Average speed files: ", average_speed_files, "\n\n")

    return average_speed_files


def check_datetime(dt: str):

    try:
        datetime.strptime(dt, "%Y-%m-%dT%H:%M:%S")
        return True
    except ValueError:
        return False