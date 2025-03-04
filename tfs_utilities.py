from datetime import datetime
import os
import json
from tfs_ops_settings import *

cwd = os.getcwd()
ops_folder = "ops"
ops_name = read_active_ops_file()

traffic_registration_points_path = f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_data/traffic_measurement_points.json"
traffic_volumes_folder_path = f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_data/traffic_volumes/raw_traffic_volumes/"


def import_TRPs_info():
    with open(traffic_registration_points_path, "r") as TRPs:
        trp_info = json.load(TRPs)

        return trp_info


def import_volumes_data(file):
    with open(file, "r") as f:
        data = json.load(f)

    return data


def get_traffic_volume_file_list():

    # Identifying all the raw traffic volume files
    volume_files = os.listdir(traffic_volumes_folder_path)
    print("Raw traffic volumes files: ", volume_files, "\n\n")

    return volume_files


def check_datetime(dt: str):

    try:
        datetime.strptime(dt, "%Y-%m-%dT%H:%M:%S")
        return True
    except ValueError:
        return False