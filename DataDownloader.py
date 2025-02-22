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



    requestChunkSize = np.array_split(ids, int(math.sqrt(len(ids))))

    #Checking for duplicates in the ids list
    #print(len(ids), "|", len(set(ids)))

    print("Requests Block Size: ", requestChunkSize)

    tv = {} #Traffic Volumes

    for ids_chunk in requestChunkSize:
        tv.update(download_ids_chunk(ids_chunk)) #The download_ids_chunk returns a dictionary with a set of ids and respective traffic volumes data
        try:
            os.wait() #Waiting for the completion of the fetch of each single data chunk to avoid running into the TimeoutError
        except AttributeError: #To avoid raising an error when the last chunk of data will be downloaded and there will be no more processes to wait
            pass

    print(tv)

    print("\n\n")

    return None



























