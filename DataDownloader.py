import json
import os
from DataFetcher import *
from warnings import simplefilter
import math
import numpy as np
import time
from tqdm import tqdm

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
    with open(f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_data/traffic_measurement_points.json", "r") as tmps_r:
        tmps = json.load(tmps_r)


    ids = []

    trafficRegistrationPoints = tmps["trafficRegistrationPoints"]

    for trp in trafficRegistrationPoints:
        ids.append(trp["id"])

    #print(ids)


    def download_ids_chunk(chunk):
        try:
            id_volumes = {}
            for i in chunk:
                id_volumes.update({i: fetch_traffic_volumes_for_tmp_id(client=client, traffic_measurement_point=i, time_start=time_start, time_end=time_end)})
            return id_volumes
        except TimeoutError:
            print("\033[91mTimeout Error Raised. Safely Exited the Program\033[0m")
            exit()


    requestChunkSize = int(math.sqrt(len(ids))) #The chunk size of each request cycle will be equal to the square root of the total number of ids
    requestChunks = np.array_split(ids, requestChunkSize)

    #Checking for duplicates in the ids list
    #print(len(ids), "|", len(set(ids)))

    #print("Requests chunks: ", requestChunks)

    tv = {} #Traffic Volumes

    for ids_chunk in tqdm(requestChunks, total=len(requestChunks)):
        tv.update(download_ids_chunk(ids_chunk)) #The download_ids_chunk returns a dictionary with a set of ids and respective traffic volumes data

    #print(tv)

    time_start = time_start[:18].replace(":", "_") #Keeping only the characters that were inputted by the user
    time_end = time_end[:18].replace(":", "_")

    #Exporting traffic volumes to a json file
    with open(f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_data/traffic_volumes_S{time_start}_E{time_end}.json", "w") as tv_w:
        json.dump(tv, tv_w, indent=4)

    #TODO FIND A LIGHTER SOLUTION THAN JSON, SINCE JUST A MONTH OF DATA IS MORE OR LESS A GIGABYTE

    print("\n\n")

    return None



























