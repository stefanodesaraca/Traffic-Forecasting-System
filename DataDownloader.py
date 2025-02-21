import json
import os
from DataFetcher import *
from warnings import simplefilter

simplefilter("ignore")

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

    with open(f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_data/traffic_measurement_points.json", "w") as tmps_w:
        json.dump(TMPs, tmps_w, indent=4)

    return None


def traffic_volumes_data_to_json(ops_name: str, time_start: str, time_end: str):

    client = start_client()

    #Read traffic measurement points json file

    #TODO FETCH ALL IDs, FOR EVERY ID AN API REQUEST FOR TRAFFIC VOLUMES DATA
    #TODO GATHER ALL DATA POINTS AND RESPECTIVE DATA INTO A SINGLE JSON FILE

    with open(f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_data/traffic_measurement_points.json", "r") as tmps_r:
        tmps = json.load(tmps_r)


    ids = []

    trafficRegistrationPoints = tmps["trafficRegistrationPoints"]

    for trp in trafficRegistrationPoints:
        ids.append(trp["id"])

    #print(ids)


    for i in ids:
        fetch_traffic_volumes_for_tmp_id(client=client, traffic_measurement_point=i, time_start=time_start, time_end=time_end)


    print("\n\n")

    return None



























