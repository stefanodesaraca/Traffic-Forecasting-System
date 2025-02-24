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
        self._ops_folder = read_active_ops_file()
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


        #TODO CONTINUE HERE WITH JUST A PD.DESCRIBE() OF THE DATA


        elif verbose is False:
            print("******** Traffic Registration Point Information ********")

            print("ID: ", data["id"])
            print("Name: ", data["name"])

        return None



    #TODO THIS WILL EXPORT ALL THE CLEANED DATA INTO SEPARATED FILES
    def cleaning_pipeline(self, file: str):

        #Importing a single json file to be cleaned
        volumes = self.import_data(file)







        return None


    def execute_cleaning(self):

        #Identifying all the raw files
        volume_files = os.listdir(f"{self._cwd}/{self._ops_folder}/{self._ops_name}/{self._ops_name}_data/traffic_volumes/raw_traffic_volumes")

        print("Raw traffic volumes files: ", volume_files)


        self.cleaning_pipeline()


        return None
























