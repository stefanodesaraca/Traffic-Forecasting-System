import json
import os
from DataFetcher import *

ops_folder = "ops"
cwd = os.getcwd()

def traffic_measurement_points_to_json(ops_name: str):
    """
    The ops_name parameter is needed to identify the operation where the data needs to be downloaded.
    This implies that the same data can be downloaded multiple times, but downloaded into different operation folders,
    so reducing the risk of data loss or corruption in case of malfunctions.
    """

    client = start_client()

    TMPs = fetch_traffic_measurement_points(client)

    with open(f"{cwd}/{ops_folder}/{ops_name}_data/traffic_measurement_points.json", "w") as tmps_w:
        json.dump(TMPs, tmps_w, indent=6)

    return None

def traffic_data_to_json():

    client = start_client()

    fetch_traffic_volumes_for_tmp_id(client)






























