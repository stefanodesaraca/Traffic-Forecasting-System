import OpsSettings
from OpsSettings import *
import numpy as np
import json
from datetime import datetime
import os
import dask.dataframe as dd



class Cleaner:

    def __init__(self):
        self._cwd = os.getcwd()
        self._ops_folder = "ops"
        self._ops_name = read_active_ops_file()


    @staticmethod
    def import_data(file):
        with open(file, "r") as f:
            data = json.load(f)

        return data


    #TODO USEFUL?
    @staticmethod
    def data_overview(data: dict, verbose: bool):
        return data



class TrafficVolumesCleaner(Cleaner):

    def __init__(self):
        super().__init__()


    #This function is only to give the user an overview of the data which we're currently cleaning, and some specific information about the TRP (Traffic Registration Point) which has collected it
    def data_overview(self, data: dict, verbose: bool):
        data = data["trafficRegistrationPoints"]

        if verbose is True:

            print("******** Traffic Registration Point Information ********")

            print("ID: ", data["id"])
            print("Name: ", data["name"])
            print("Road category: ", data["location"]["roadReference"]["roadCategory"]["id"])
            print("Coordinates -> ", "Lat: ", data["location"]["latLon"]["lat"], "Lon: ", data["location"]["latLon"]["lon"])
            print("County name: ", data["location"]["county"]["name"])
            print("County number: ", data["location"]["county"]["number"])
            print("Geographic number: ", data["location"]["county"]["geographicNumber"])
            print("Country part: ", data["location"]["county"]["countryPart"]["name"])
            print("Municipality ID: ", data["location"]["municipality"]["id"])
            print("Municipality name: ", data["location"]["municipality"]["name"])

            print("Traffic registration type: ", data["trafficRegistrationType"])
            print("Data time span: ")
            print(" - First data: ", data["dataTimeSpan"]["firstData"])
            print(" - First data with quality metrics: ", data["dataTimeSpan"]["firstDataWithQualityMetrics"])
            print(" - Latest data: ")
            print("   > Volume by day: ", data["dataTimeSpan"]["latestData"]["volumeByDay"])
            print("   > Volume by hour: ", data["dataTimeSpan"]["latestData"]["volumeByHour"])
            print("   > Volume average daily by year: ", data["dataTimeSpan"]["latestData"]["volumeAverageDailyByYear"])
            print("   > Volume average daily by season: ", data["dataTimeSpan"]["latestData"]["volumeAverageDailyBySeason"])
            print("   > Volume average daily by month: ", data["dataTimeSpan"]["latestData"]["volumeAverageDailyByMonth"])

            data = dd.DataFrame(data)

            print(data.describe())



        elif verbose is False:

            print("******** Traffic Registration Point Information ********")

            print("ID: ", data["id"])
            print("Name: ", data["name"])

        return None



    #TODO THIS WILL EXPORT ALL THE CLEANED DATA INTO SEPARATED FILES AND CLEAN EACH FILE INDIVIDUALLY THROUGH THE PIPELINE.
    # IN THE DATA EXPLORATION FILE WE'LL CREATE A FOR LOOP AND USE THE cleaning_pipeline() FUNCTION TO CLEAN EACH FILE.
    # THIS PROCESS WILL BE REPEATED BOTH FOR TRAFFIC VOLUMES AND AVERAGE SPEED, EACH ONE WITH THEIR OWN CUSTOM CLEANING PIPELINE
    def cleaning_pipeline(self, file: str):

        #Importing a single json file to be cleaned
        volumes = self.import_data(file)

        self.data_overview(volumes, verbose=True)





        return None


    def execute_cleaning(self):

        traffic_volumes_folder_path = f"{self._cwd}/{self._ops_folder}/{self._ops_name}/{self._ops_name}_data/traffic_volumes/raw_traffic_volumes/"

        #Identifying all the raw files
        volume_files = os.listdir(traffic_volumes_folder_path)

        print("Raw traffic volumes files: ", volume_files, "\n\n")

        self.cleaning_pipeline(traffic_volumes_folder_path + volume_files[0])


        return None
























