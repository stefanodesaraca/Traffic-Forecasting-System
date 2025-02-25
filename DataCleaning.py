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
    def import_TRPs(file):
        with open(file, "r") as TRPs:

            trp_info = json.load(TRPs)

            return trp_info


    @staticmethod
    def import_data(file):
        with open(file, "r") as f:
            data = json.load(f)

        return data


    #General definition of the data_overview() function. This will take two different forms: the traffic volumes one and the average speed one.
    #Thus, the generic "data" parameter will become the volumes_data or the avg_speed_data one
    @staticmethod
    def data_overview(trp_data: dict, data: dict, verbose: bool):
        return data



class TrafficVolumesCleaner(Cleaner):

    def __init__(self):
        super().__init__()


    #This function is only to give the user an overview of the data which we're currently cleaning, and some specific information about the TRP (Traffic Registration Point) which has collected it
    def data_overview(self, trp_data: dict, volumes_data: dict, verbose: bool):

        volumes_data = volumes_data["trafficData"]

        if verbose is True:

            print("******** Traffic Registration Point Information ********")

            print("ID: ", trp_data["id"])
            print("Name: ", trp_data["name"])
            print("Road category: ", trp_data["location"]["roadReference"]["roadCategory"]["id"])
            print("Coordinates -> ", "Lat: ", volumes_data["location"]["latLon"]["lat"], "Lon: ", trp_data["location"]["latLon"]["lon"])
            print("County name: ", trp_data["location"]["county"]["name"])
            print("County number: ", trp_data["location"]["county"]["number"])
            print("Geographic number: ", trp_data["location"]["county"]["geographicNumber"])
            print("Country part: ", trp_data["location"]["county"]["countryPart"]["name"])
            print("Municipality ID: ", trp_data["location"]["municipality"]["id"])
            print("Municipality name: ", trp_data["location"]["municipality"]["name"])

            print("Traffic registration type: ", trp_data["trafficRegistrationType"])
            print("Data time span: ")
            print(" - First data: ", trp_data["dataTimeSpan"]["firstData"])
            print(" - First data with quality metrics: ", trp_data["dataTimeSpan"]["firstDataWithQualityMetrics"])
            print(" - Latest data: ")
            print("   > Volume by day: ", trp_data["dataTimeSpan"]["latestData"]["volumeByDay"])
            print("   > Volume by hour: ", trp_data["dataTimeSpan"]["latestData"]["volumeByHour"])
            print("   > Volume average daily by year: ", trp_data["dataTimeSpan"]["latestData"]["volumeAverageDailyByYear"])
            print("   > Volume average daily by season: ", trp_data["dataTimeSpan"]["latestData"]["volumeAverageDailyBySeason"])
            print("   > Volume average daily by month: ", trp_data["dataTimeSpan"]["latestData"]["volumeAverageDailyByMonth"])

            volumes_data = dd.DataFrame(volumes_data)

            print(volumes_data.describe())



        elif verbose is False:

            print("******** Traffic Registration Point Information ********")

            print("ID: ", trp_data["id"])
            print("Name: ", trp_data["name"])

        return None



    #TODO THIS WILL EXPORT ALL THE CLEANED DATA INTO SEPARATED FILES AND CLEAN EACH FILE INDIVIDUALLY THROUGH THE PIPELINE.
    # IN THE DATA EXPLORATION FILE WE'LL CREATE A FOR LOOP AND USE THE cleaning_pipeline() FUNCTION TO CLEAN EACH FILE.
    # THIS PROCESS WILL BE REPEATED BOTH FOR TRAFFIC VOLUMES AND AVERAGE SPEED, EACH ONE WITH THEIR OWN CUSTOM CLEANING PIPELINE
    def cleaning_pipeline(self, file: str):

        #Importing a single json file to be cleaned
        volumes = self.import_data(file)

        print(volumes)

        self.data_overview(volumes, verbose=True)





        return None


    def execute_cleaning(self):

        traffic_volumes_folder_path = f"{self._cwd}/{self._ops_folder}/{self._ops_name}/{self._ops_name}_data/traffic_volumes/raw_traffic_volumes/"

        #Identifying all the raw files
        volume_files = os.listdir(traffic_volumes_folder_path)

        print("Raw traffic volumes files: ", volume_files, "\n\n")


        traffic_registration_points = self.import_TRPs(f"{self._cwd}/{self._ops_folder}/{self._ops_name}/{self._ops_name}_data/traffic_measurement_points.json")

        #TODO TEST HERE WITH [:2]
        for volume_f in volume_files[:2]:
            self.cleaning_pipeline(traffic_volumes_folder_path + volume_f) #String concatenation here


        return None
























