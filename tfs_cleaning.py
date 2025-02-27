from tfs_ops_settings import *
import numpy as np
import json
import datetime
from datetime import datetime
import os
import dask.dataframe as dd
import pandas as pd



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
    def data_overview(trp_data, data: dict, verbose: bool):
        return data


#TODO THE ONLY PARAMETER WHICH WILL BE NEEDED BY THE TrafficVolumesCleaner CLASS IS THE trp id, ONE AT A TIME
class TrafficVolumesCleaner(Cleaner):

    def __init__(self):
        super().__init__()


    #This function is only to give the user an overview of the data which we're currently cleaning, and some specific information about the TRP (Traffic Registration Point) which has collected it
    def data_overview(self, trp_data, volumes_data: dict, verbose: bool):

        trp_data = trp_data["trafficRegistrationPoints"][0]

        if verbose is True:

            print("******** Traffic Registration Point Information ********")

            print("ID: ", trp_data["id"])
            print("Name: ", trp_data["name"])
            print("Road category: ", trp_data["location"]["roadReference"]["roadCategory"]["id"])
            print("Coordinates: ")
            print(" - Lat: ", trp_data["location"]["coordinates"]["latLon"]["lat"])
            print(" - Lon: ", trp_data["location"]["coordinates"]["latLon"]["lon"])
            print("County name: ", trp_data["location"]["county"]["name"])
            print("County number: ", trp_data["location"]["county"]["number"])
            print("Geographic number: ", trp_data["location"]["county"]["geographicNumber"])
            print("Country part: ", trp_data["location"]["county"]["countryPart"]["name"])
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

            print("\n")


            print(volumes_data)


            print("--------------------------------------------------------\n\n")



        elif verbose is False:

            print("******** Traffic Registration Point Information ********")

            print("ID: ", trp_data["id"])
            print("Name: ", trp_data["name"])

            print("--------------------------------------------------------\n\n")


        return None



    @staticmethod
    def clean_traffic_volumes_data(volumes_payload):

        # ------------------ Data payload extraction ------------------

        nodes = volumes_payload["trafficData"]["volume"]["byHour"]["edges"]


        # ------------------ Finding all unique days in which registrations took place ------------------

        registration_dates = [] #To find all the unique days

        for n in nodes:
            registration_dt = n["from"][-6:] #Only keeping the datetime without the +00:00 at the end
            print("Registration DT: ", registration_dt)
            registration_dates.append(registration_dt)

        registration_dates = set(registration_dates) #Removing duplicates
        print("Number of unique registration days: ", len(registration_dates))


        # ------------------ Extracting the data from JSON file and converting it into tabular format ------------------


        by_hour_structured = [] #This will later become a list of dictionaries to create the by_hour dataframe we're going to export and use in the future
        by_lane_structured = [] #This will later become a list of dictionaries to create the by_lane dataframe we're going to export and use in the future
        by_direction_structured = [] #This will later become a list of dictionaries to create the by_direction dataframe we're going to export and use in the future

        data_indexes = [] #This list will make every registration's day dictionary trackable to be able to insert the data into it

        #ud = unique day
        for idx, ud in enumerate(registration_dates):
            data_indexes.append({ud: idx})

            #Creating as many dictionaries as there are registration days, so each registration day will have its own dictionary with its specific data
            by_hour_structured.append({})
            by_lane_structured.append({})
            by_direction_structured.append({})

        print(data_indexes)


        for node in nodes:
            #This is the datetime which will be representative of a volume, specifically, there will be multiple datetimes with the same day
            # to address this fact we'll just re-format the data to keep track of the day, but also maintain the volume values for each hour
            registration_datetime = node["from"][-6:] #Only keeping the datetime without the +00:00 at the end

            registration_datetime = datetime.strptime(registration_datetime, "%Y-%m-%dT%H:%M:%S")
            day = registration_datetime.day
            hour = registration_datetime.hour


            # ----------------------- Total volumes section -----------------------
            total_volume = node["total"]["volumeNumbers"]["volume"]
            coverage_perc = node["total"]["coverage"]["percentage"]





            #   ----------------------- By lane section -----------------------

            lanes_data = node["byLane"]


            lanes = [] #Keeping track of all the lanes available to find the total number of lanes for each TRP (Traffic Registration Point)
            lanes_structured = {} #Structured data to create dataframe from a dict of dicts


            #Every lane's data is kept isolated from the other lanes' data, so a for cycle is needed
            for lane in lanes_data:
                road_link_lane_number = lane["lane"]["laneNumberAccordingToRoadLink"]
                lane_volume = lane["lane"]["total"]["volumeNumbers"]["volume"]
                lane_coverage = lane["lane"]["total"]["coverage"]["percentage"]

                lanes_structured.update({road_link_lane_number: {"volume": lane_volume,
                                                                 "lane_coverage": lane_coverage}
                                                                                                })

                lanes.append(road_link_lane_number)

            total_lanes = max(lanes) #To get the maximum number of lanes. This is needed in the future for the dataframes creation and to be inserted as additional data into the graph nodes
            #TODO WRITE A METADATA FILE FOR EACH TRP, SAVE THEM IN A SPECIFIC FOLDER IN THE GRAPH'S ONE (IN THE ops FOLDER)

            #   ----------------------- By direction section -----------------------

            by_direction_data = node["byDirection"]


            heading_directions = []  #Keeping track of all the directions available for each TRP (Traffic Registration Point)
            direction_structured = {}


            # Every direction's data is kept isolated from the other directions' data, so a for cycle is needed
            for direction in by_direction_data:
                heading = direction["heading"]
                direction_volume = direction["heading"]["total"]["volumeNumbers"]["volume"]
                direction_coverage = direction["heading"]["total"]["coverage"]["percentage"]

                direction_structured.update({heading: {"volume": direction_volume,
                                                       "direction_coverage": direction_coverage}
                                                                                                })
                #TODO THE SAME PRINCIPLE AS BEFORE APPLIES HERE, SAVE ALL THE AVAILABLE DIRECTIONS IN THE TRP'S METADATA FILE






        #TODO CREATE DATAFRAMES HERE (OUTSIDE THE MAIN for node in nodes FOR CYCLE), MEMORIZE DATA OUTSIDE THE CYCLE AND BE AWARE OF THE DATES PROBLEM


        return None #TODO RETURN CLEANED DATA IN THREE DIFFERENT DATAFRAMES AS DESCRIBED ON THE PAPER NOTEBOOK










    #TODO THIS WILL EXPORT ALL THE CLEANED DATA INTO SEPARATED FILES AND CLEAN EACH FILE INDIVIDUALLY THROUGH THE PIPELINE.
    # IN THE DATA EXPLORATION FILE WE'LL CREATE A FOR LOOP AND USE THE cleaning_pipeline() FUNCTION TO CLEAN EACH FILE.
    # THIS PROCESS WILL BE REPEATED BOTH FOR TRAFFIC VOLUMES AND AVERAGE SPEED, EACH ONE WITH THEIR OWN CUSTOM CLEANING PIPELINE
    def cleaning_pipeline(self, trp_file: str, file: str):

        #Importing a single json file to be cleaned
        volumes = self.import_data(file)

        #print(trp_file)
        #print(volumes)

        self.data_overview(trp_data=trp_file, volumes_data=volumes, verbose=True)


        return None


    def execute_cleaning(self):

        traffic_volumes_folder_path = f"{self._cwd}/{self._ops_folder}/{self._ops_name}/{self._ops_name}_data/traffic_volumes/raw_traffic_volumes/"
        traffic_registration_points_path = self.import_TRPs(f"{self._cwd}/{self._ops_folder}/{self._ops_name}/{self._ops_name}_data/traffic_measurement_points.json")


        #Identifying all the raw traffic volume files
        volume_files = os.listdir(traffic_volumes_folder_path)
        print("Raw traffic volumes files: ", volume_files, "\n\n")


        #TODO TEST HERE WITH [:2]
        for volume_f in volume_files[:2]:
            volumes_file_path = traffic_volumes_folder_path + volume_f
            self.cleaning_pipeline(trp_file=traffic_registration_points_path, file=volumes_file_path) #String concatenation here


        return None
























