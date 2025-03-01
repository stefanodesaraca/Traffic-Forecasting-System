from tfs_ops_settings import *
import numpy as np
import json
import datetime
from datetime import datetime
import os
import dask.dataframe as dd
import pandas as pd
import pprint


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

            #print(volumes_data)

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
        number_of_nodes = len(nodes)

        #print(nodes)


        # ------------------ Finding all unique days in which registrations took place ------------------

        registration_dates = [] #To find all the unique days

        for n in nodes:

            #print(n)

            registration_dt = n["node"]["from"][:-6] #Only keeping the datetime without the +00:00 at the end
            #print("Registration DT: ", registration_dt)
            registration_dates.append(registration_dt)

        registration_dates = set(registration_dates) #Removing duplicates and keeping the time as well. This will be needed to extract the hour too
        unique_registration_dates = set([r_date[:10] for r_date in registration_dates]) #Removing duplicates, this one will only keep the first 10 characters of the date, which comprehend just the date without the time. This is needed to know which are the unique days when data has been recorded
        print("Number of unique registration days: ", len(registration_dates))


        # ------------------ Extracting the data from JSON file and converting it into tabular format ------------------


        by_hour_structured = [] #This will later become a list of dictionaries to create the by_hour dataframe we're going to export and use in the future
        by_lane_structured = [] #This will later become a list of dictionaries to create the by_lane dataframe we're going to export and use in the future
        by_direction_structured = [] #This will later become a list of dictionaries to create the by_direction dataframe we're going to export and use in the future


        data_indexes = {} #This dictionary will make every registration's day dictionary trackable to be able to insert the data into it

        #ud = unique day
        for idx, ud in enumerate(unique_registration_dates):
            data_indexes.update({ud: idx}) #Every key-value pair represents {unique date: by_hour/lane/direction_structured list cell index}

            #Creating as many dictionaries as there are registration days, so each registration day will have its own dictionary with its specific data
            by_hour_structured.append({})

        print("Data indexes: ", data_indexes)
        print("By lane structured: ", by_lane_structured)


        #Every lane has its own volumes and coverage for the same node, hence the same datetime.
        #Since a dataframe is bidimensional we must find a solution to keep track of the lanes individual volumes and coverage, but still changing the data structure to tabular
        # the solution is to have every row of the dataframe be one lane's data, and have one field which indicates the date
        # so multiple rows will have the same date, but different lanes
        by_lane_index = 0

        #The same principle as by_lane_index applies here. Since there are multiple headings we'll have to create an index for each row in the dataframe
        #Again, multiple rows will have the same date, but different headings
        by_direction_index = 0


        for node in nodes:

            # ---------------------- Fetching registration datetime ----------------------

            #This is the datetime which will be representative of a volume, specifically, there will be multiple datetimes with the same day
            # to address this fact we'll just re-format the data to keep track of the day, but also maintain the volume values for each hour
            registration_datetime = node["node"]["from"][:-6] #Only keeping the datetime without the +00:00 at the end

            registration_datetime = datetime.strptime(registration_datetime, "%Y-%m-%dT%H:%M:%S")
            day = registration_datetime.strftime("%Y-%m-%d")
            hour = registration_datetime.strftime("%H")

            #print(registration_datetime)

            print(day)
            print(hour)


            # ---------------------- Finding the by_hour_structured list cell where the data for the current node will be inserted ----------------------

            by_hour_idx = data_indexes[day] #We'll obtain the index of the list cell where the dictionary for this specific date lies


            # ----------------------- Total volumes section -----------------------
            total_volume = node["node"]["total"]["volumeNumbers"]["volume"]
            coverage_perc = node["node"]["total"]["coverage"]["percentage"]

            by_hour_structured[by_hour_idx].update({f"v{hour}": total_volume}) # <-- Inserting the total volumes (for the specific hour) data into the dictionary previously created in the by_hour_structured list
            by_hour_structured[by_hour_idx].update({f"cvg{hour}": coverage_perc}) # <-- Inserting the coverage data (for the specific hour) into the dictionary previously created in the by_hour_structured list


            #   ----------------------- By lane section -----------------------

            lanes_data = node["node"]["byLane"] #Extracting byLane data
            lanes = [] #Storing the lane numbers to find the maximum number of lanes for the specific TRP

            #Every lane's data is kept isolated from the other lanes' data, so a for cycle is needed to extract all the data from each lane's section
            for lane in lanes_data:
                road_link_lane_number = lane["lane"]["laneNumberAccordingToRoadLink"]
                lane_volume = lane["total"]["volumeNumbers"]["volume"]
                lane_coverage = lane["total"]["coverage"]["percentage"]

                lanes.append(road_link_lane_number)


                by_lane_structured.append({by_lane_index: {"date": day,
                                                          f"lane": f"l{road_link_lane_number}",
                                                          f"v{hour}": lane_volume,
                                                          f"lane_cvg{hour}": lane_coverage}
                                                                                                            })

                by_lane_index += 1

                #print(by_lane_structured)
                #print(by_lane_index)


            #   ----------------------- By direction section -----------------------

            by_direction_data = node["byDirection"]
            heading_directions = []  #Keeping track of the directions available for the specific TRP (Traffic Registration Point)

            # Every direction's data is kept isolated from the other directions' data, so a for cycle is needed
            for direction in by_direction_data:
                heading = direction["heading"]
                direction_volume = direction["heading"]["total"]["volumeNumbers"]["volume"]
                direction_coverage = direction["heading"]["total"]["coverage"]["percentage"]

                by_direction_structured.append({by_direction_index: {"heading": heading,
                                                                     f"v{hour}": direction_volume,
                                                                     f"direction_cvg{hour}": direction_coverage}
                                                                                                                 })

                by_direction_index += 1

                #TODO THE SAME PRINCIPLE AS BEFORE APPLIES HERE, SAVE ALL THE AVAILABLE DIRECTIONS IN THE TRP'S METADATA FILE


        total_lanes = max(by_lane_structured[0]["lane"]) #Finding the number of lanes for the specific TRP (we'll just check the first element since every data node has all lanes' data registered for every hour. In case there's no traffic the values will just be null)
        #This could be inserted as additional data into the graph nodes
        #TODO WRITE A METADATA FILE FOR EACH TRP, SAVE THEM IN A SPECIFIC FOLDER IN THE GRAPH'S ONE (IN THE ops FOLDER)

        print("Total lanes: ", total_lanes)
        print("Theoretical number of registrations (total_lanes * number of nodes): ", total_lanes*number_of_nodes)
        print("Real number of registrations (length of the by_lane_structured) list: ", len(by_lane_structured))

        #We'll have an overview on all the available directions for the specific TRP
        print("Available directions (headings): ", set(by_direction_structured[0]["heading"])) #Same principle as for total_lanes applies here


        print("----------------- By Hour Structured -----------------")
        pprint.pp(by_hour_structured)

        print("----------------- By Lane Structured -----------------")

        pprint.pp(by_lane_structured)

        print("----------------- By Direction Structured -----------------")

        pprint.pp(by_direction_structured)





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

        self.clean_traffic_volumes_data(volumes)


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
























