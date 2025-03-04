from tfs_ops_settings import *
from tfs_utilities import *
import numpy as np
import json
import datetime
from datetime import datetime
import os
import pandas as pd
import pprint


class Cleaner:

    def __init__(self):
        self._cwd = os.getcwd()
        self._ops_folder = "ops"
        self._ops_name = read_active_ops_file()

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

        #print(volumes_data)

        trp_id = volumes_data["trafficData"]["trafficRegistrationPoint"]["id"]
        trp_data = [i for i in trp_data["trafficRegistrationPoints"] if i["id"] == trp_id] #Finding the data for the specific TRP taken in consideration by iterating on all the TRPs available in the trp_file
        trp_data = trp_data[0]

        print(trp_data)

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
            print("Number of data nodes: ", len(volumes_data["trafficData"]["volume"]["byHour"]["edges"]))

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

        by_hour_structured = [] #This will later become a list of dictionaries to create the by_hour dataframe we're going to export and use in the future
        by_lane_structured = [] #This will later become a list of dictionaries to create the by_lane dataframe we're going to export and use in the future
        by_direction_structured = [] #This will later become a list of dictionaries to create the by_direction dataframe we're going to export and use in the future


        # ------------------ Data payload extraction ------------------

        nodes = volumes_payload["trafficData"]["volume"]["byHour"]["edges"]
        number_of_nodes = len(nodes)

        #print(nodes)

        if number_of_nodes == 0:
            print(f"\033[91mNo data found for TRP: {volumes_payload["trafficData"]["trafficRegistrationPoint"]["id"]}\033[0m\n\n")

            return None

        else:

            # ------------------ Finding the number of lanes available for the TRP taken into consideration ------------------

            lane_sample_node = nodes[0]["node"]["byLane"]
            n_lanes = max([ln["lane"]["laneNumberAccordingToRoadLink"] for ln in lane_sample_node]) #Determining the total number of lanes for the TRP taken into consideration
            print("Number of lanes: ", n_lanes)

            #The number of lanes is calculated because, as opposed to by_hour_structured, where the list index will be the row index in the dataframe,
            # in the by_lane and by_direction dataframes dates and lane numbers could be repeated, thus there isn't a unique dict key which could be used to
            # identify the right dictionary where to write volumes and coverage data
            # So, we'll create afterward a unique identifier which will be made like: date + "l" + lane number. This will make possible to identify each single dictionary in the list of dicts (by_lane_structured)
            # and consequently put the right data in it.
            # This is also made to address the fact that a node could contain data from slightly more than one day

            #This could be inserted as additional data into the graph nodes
            #TODO WRITE A METADATA FILE FOR EACH TRP, SAVE THEM IN A SPECIFIC FOLDER IN THE GRAPH'S ONE (IN THE ops FOLDER)


            # ------------------ Finding all the available directions for the TRP ------------------

            direction_sample_node = nodes[0]["node"]["byDirection"]
            directions = [d["heading"] for d in direction_sample_node]


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

            by_hour_data_indexes = {} #This dictionary will make every registration's day dictionary trackable to be able to insert the data into it
            by_lane_data_indexes = {}
            by_direction_data_indexes = {}

            l_idx_cnt = 0 #Lane index counter
            d_idx_cnt = 0 #Direction index counter

            #ud = unique day
            for idx, ud in enumerate(unique_registration_dates):
                by_hour_data_indexes.update({ud: idx}) #Every key-value pair represents {unique date: by_hour/lane/direction_structured list cell index}

                #Creating as many dictionaries as there are registration days, so each registration day will have its own dictionary with its specific data
                by_hour_structured.append({})

                # It's necessary to execute this step here because two kinds of data are necessary to ensure that the correct number of dictionary and the correct keys are created
                # 1. A node could contain data from slightly more than one day (for example 1 or 2 hours from the prior one and the rest for the specific day)
                # 2. Knowing the number of lanes before lets us define specific indexes for the by_lane_structured list of dict so that we can ensure that the data for a specific date-lane combination goes into its dedicated dictionary
                for l_number in range(1, n_lanes+1): #It's necessary to start from 1 since the lanes are numbered from 1 to n (l_number = lane number)
                    by_lane_structured.append({f"{ud}l{l_number}": {}}) #Appending a dict to fill out with data later
                    by_lane_data_indexes.update({f"{ud}l{l_number}": l_idx_cnt})

                    l_idx_cnt += 1

                for dr in directions:
                    by_direction_structured.append({f"{ud}h{dr}": {}})
                    by_direction_data_indexes.update({f"{ud}h{dr}": d_idx_cnt})

                    d_idx_cnt += 1

            #print("By hour data indexes: ", by_hour_data_indexes)
            #print("By hour structured: ", by_hour_structured) #It's normal that the list contains only empty dictionaries since they're later going to be filled with data
            #print("By lane data indexes: ", by_lane_data_indexes)
            #print("By lane structured: ", by_lane_structured)
            #print("By direction structured: ", by_direction_structured)
            #print("By direction data indexes: ", by_direction_data_indexes)


            for node in nodes:

                # ---------------------- Fetching registration datetime ----------------------

                #This is the datetime which will be representative of a volume, specifically, there will be multiple datetimes with the same day
                # to address this fact we'll just re-format the data to keep track of the day, but also maintain the volume values for each hour
                registration_datetime = node["node"]["from"][:-6] #Only keeping the datetime without the +00:00 at the end

                registration_datetime = datetime.strptime(registration_datetime, "%Y-%m-%dT%H:%M:%S")
                day = registration_datetime.strftime("%Y-%m-%d")
                hour = registration_datetime.strftime("%H")

                #print(registration_datetime)

                #print(day)
                #print(hour)


                # ---------------------- Finding the by_hour_structured list cell where the data for the current node will be inserted ----------------------

                by_hour_idx = by_hour_data_indexes[day] #We'll obtain the index of the list cell where the dictionary for this specific date lies


                # ----------------------- Total volumes section -----------------------
                total_volume = node["node"]["total"]["volumeNumbers"]["volume"] if node["node"]["total"]["volumeNumbers"] is not None else None #In some cases the volumeNumbers key could have null as value, so the "volume" key won't be present. In that case we'll directly insert None as value with an if statement
                coverage_perc = node["node"]["total"]["coverage"]["percentage"]

                by_hour_structured[by_hour_idx]["date"] = day #Adding or updating the "date" key for each row

                by_hour_structured[by_hour_idx].update({f"v{hour}": total_volume, # <-- Inserting the total volumes (for the specific hour) data into the dictionary previously created in the by_hour_structured list
                                                        f"cvg{hour}": coverage_perc}) # <-- Inserting the coverage data (for the specific hour) into the dictionary previously created in the by_hour_structured list


                #   ----------------------- By lane section -----------------------

                lanes_data = node["node"]["byLane"] #Extracting byLane data
                lanes = [] #Storing the lane numbers to find the maximum number of lanes for the specific TRP


                #Every lane's data is kept isolated from the other lanes' data, so a for cycle is needed to extract all the data from each lane's section
                for lane in lanes_data:
                    #print(lane)
                    road_link_lane_number = lane["lane"]["laneNumberAccordingToRoadLink"]
                    lane_volume = lane["total"]["volumeNumbers"]["volume"] if lane["total"]["volumeNumbers"] is not None else None #In some cases the volumeNumbers key could have null as value, so the "volume" key won't be present. In that case we'll directly insert None as value with an if statement
                    lane_coverage = lane["total"]["coverage"]["percentage"]

                    lanes.append(road_link_lane_number)

                    date_lane_index = str(day) + "l" + str(road_link_lane_number) #Combination of day and lane number
                    by_lane_index = by_lane_data_indexes[date_lane_index]

                    # ------- Creating or updating new keys for the dictionary which contains the ith-day and jth-lane data -------

                    by_lane_structured[by_lane_index][date_lane_index]["date"] = day
                    by_lane_structured[by_lane_index][date_lane_index]["lane"] = f"l{road_link_lane_number}"
                    by_lane_structured[by_lane_index][date_lane_index][f"v{hour}"] = lane_volume
                    by_lane_structured[by_lane_index][date_lane_index][f"lane_cvg{hour}"] = lane_coverage


                #   ----------------------- By direction section -----------------------

                by_direction_data = node["node"]["byDirection"]

                # Every direction's data is kept isolated from the other directions' data, so a for cycle is needed
                for direction_section in by_direction_data:
                    heading = direction_section["heading"]
                    direction_volume = direction_section["total"]["volumeNumbers"]["volume"] if direction_section["total"]["volumeNumbers"] is not None else None #In some cases the volumeNumbers key could have null as value, so the "volume" key won't be present. In that case we'll directly insert None as value with an if statement
                    direction_coverage = direction_section["total"]["coverage"]["percentage"]

                    date_direction_index = str(day) + "h" + str(heading) #Combination of day and direction (heading)
                    by_direction_index = by_direction_data_indexes[date_direction_index]

                    by_direction_structured[by_direction_index][date_direction_index]["date"] = day
                    by_direction_structured[by_direction_index][date_direction_index]["heading"] = heading
                    by_direction_structured[by_direction_index][date_direction_index][f"v{hour}"] = direction_volume
                    by_direction_structured[by_direction_index][date_direction_index][f"direction_cvg{hour}"] = direction_coverage


                    #TODO THE SAME PRINCIPLE AS BEFORE APPLIES HERE, SAVE ALL THE AVAILABLE DIRECTIONS IN THE TRP'S METADATA FILE



            # ------------------ Ensuring that every dictionary in by_hour/by_lane/by_direction_structured has the same number of key-value pairs ------------------


            hours = [f"{i:02}" for i in range(24)] #Generating 24 elements starting from 00 to 23
            by_x_structured_volume_keys = [f"v{i:02}" for i in range(24)] #These can be used both for by_hour_structured, by_lane_structured and for by_direction_structured

            #print(by_x_structured_volume_keys)

            by_hour_coverage_keys = [f"cvg{i:02}" for i in range(24)]
            by_lane_coverage_keys = [f"lane_cvg{i:02}" for i in range(24)]
            by_direction_coverage_keys = [f"direction_cvg{i:02}" for i in range(24)]


            # ------------------ by_hour_structured check ------------------

            #bh_dict = by hour dictionary
            #Creating a dictionary made by the list indexes of the by_hour_structured list and verifying which keys are missing in each element of the list (which is a dictionary) and thus, their values as well
            by_hour_structured_keys = {by_hour_structured.index(list_element): list(list_element.keys()) for list_element in by_hour_structured}

            by_hour_volume_keys_to_add = {element: [to_add for to_add in by_x_structured_volume_keys if to_add not in keys] for element, keys in by_hour_structured_keys.items()}
            by_hour_cvg_keys_to_add = {element: [to_add for to_add in by_hour_coverage_keys if to_add not in keys] for element, keys in by_hour_structured_keys.items()}

            by_hour_volume_keys_to_add = {bh_idx: bh_to_add_element for bh_idx, bh_to_add_element in by_hour_volume_keys_to_add.items() if len(bh_to_add_element) != 0} #Only keeping non-empty key-value pairs. So basically we're only keeping the dictionary pairs which actually contain elements to add
            by_hour_cvg_keys_to_add = {bh_idx: bh_to_add_element for bh_idx, bh_to_add_element in by_hour_cvg_keys_to_add.items() if len(bh_to_add_element) != 0} #Only keeping non-empty key-value pairs. So basically we're only keeping the dictionary pairs which actually contain elements to add

            #print("By hour volume keys to add: ")
            #print(by_hour_volume_keys_to_add)

            #print("By hour coverage keys to add: ")
            #print(by_hour_cvg_keys_to_add, "\n\n")


            # ------------------ Adding missing volume key-value pairs to by_hour_structured ------------------

            #Adding missing volume keys
            for list_idx, bh_to_add_elements in by_hour_volume_keys_to_add.items():
                bh_to_add_elements = {el: None for el in bh_to_add_elements} #Adding the missing keys as keys and None as value
                by_hour_structured[list_idx].update(bh_to_add_elements)

            #Adding missing coverage keys
            for list_idx, bh_to_add_elements in by_hour_cvg_keys_to_add.items():
                bh_to_add_elements = {el: None for el in bh_to_add_elements} #Adding the missing keys as keys and None as value
                by_hour_structured[list_idx].update(bh_to_add_elements)


            # ------------------ by_lane_structured check ------------------

            by_lane_structured_keys = {by_lane_structured.index(list_element): list(list_element.values()) for list_element in by_lane_structured}
            by_lane_structured_keys = {e: list(k[0].keys()) for e, k in by_lane_structured_keys.items()}

            by_lane_volume_keys_to_add = {element: [to_add for to_add in by_x_structured_volume_keys if to_add not in keys] for element, keys in by_lane_structured_keys.items()}
            by_lane_cvg_keys_to_add = {element: [to_add for to_add in by_lane_coverage_keys if to_add not in keys] for element, keys in by_lane_structured_keys.items()}

            by_lane_volume_keys_to_add = {bl_idx: bl_to_add_element for bl_idx, bl_to_add_element in by_lane_volume_keys_to_add.items() if len(bl_to_add_element) != 0} #Only keeping non-empty key-value pairs. So basically we're only keeping the dictionary pairs which actually contain elements to add
            by_lane_cvg_keys_to_add = {bl_idx: bl_to_add_element for bl_idx, bl_to_add_element in by_lane_cvg_keys_to_add.items() if len(bl_to_add_element) != 0} #Only keeping non-empty key-value pairs. So basically we're only keeping the dictionary pairs which actually contain elements to add

            #print("By lane volume keys to add: ")
            #print(by_lane_volume_keys_to_add)

            #print("By lane coverage keys to add: ")
            #print(by_lane_cvg_keys_to_add, "\n\n")


            # ------------------ Adding missing key-value pairs to by_lane_structured ------------------

            #Adding missing volume keys
            for list_idx, bl_to_add_elements in by_lane_volume_keys_to_add.items():
                bl_to_add_elements = {el: None for el in bl_to_add_elements} #Adding the missing keys as keys and None as value

                for el in by_lane_structured:
                    if by_lane_structured.index(el) == list_idx:
                        bl_loc_key = list(el.keys())[0]

                        by_lane_structured[list_idx][bl_loc_key].update(bl_to_add_elements)

            #Adding missing coverage keys
            for list_idx, bl_to_add_elements in by_lane_cvg_keys_to_add.items():
                bl_to_add_elements = {el: None for el in bl_to_add_elements} #Adding the missing keys as keys and None as value

                for el in by_lane_structured:
                    if by_lane_structured.index(el) == list_idx:
                        bl_loc_key = list(el.keys())[0]

                        by_lane_structured[list_idx][bl_loc_key].update(bl_to_add_elements)


            # ------------------ by_direction_structured check ------------------

            by_direction_structured_keys = {by_direction_structured.index(list_element): list(list_element.values()) for list_element in by_direction_structured}
            by_direction_structured_keys = {e: list(k[0].keys()) for e, k in by_direction_structured_keys.items()}

            by_direction_volume_keys_to_add = {element: [to_add for to_add in by_x_structured_volume_keys if to_add not in keys] for element, keys in by_direction_structured_keys.items()}
            by_direction_cvg_keys_to_add = {element: [to_add for to_add in by_direction_coverage_keys if to_add not in keys] for element, keys in by_direction_structured_keys.items()}

            #print("By direction volume keys to add: ")
            #print(by_direction_volume_keys_to_add)

            #print("By direction coverage keys to add: ")
            #print(by_direction_cvg_keys_to_add, "\n\n")


            # ------------------ Adding missing key-value pairs to by_direction_structured ------------------

            #Adding missing volume keys
            for list_idx, bd_to_add_elements in by_direction_volume_keys_to_add.items():
                bd_to_add_elements = {el: None for el in bd_to_add_elements}  # Adding the missing keys as keys and None as value

                for el in by_direction_structured:
                    if by_direction_structured.index(el) == list_idx:
                        bd_loc_key = list(el.keys())[0]

                        by_direction_structured[list_idx][bd_loc_key].update(bd_to_add_elements)

            #Adding missing coverage keys
            for list_idx, bd_to_add_elements in by_direction_cvg_keys_to_add.items():
                bd_to_add_elements = {el: None for el in bd_to_add_elements}  # Adding the missing keys as keys and None as value

                for el in by_direction_structured:
                    if by_direction_structured.index(el) == list_idx:
                        bd_loc_key = list(el.keys())[0]

                        by_direction_structured[list_idx][bd_loc_key].update(bd_to_add_elements)



            # ------------------ Inserting additional indexes of by_lane_structured and by_direction_structured as key removing them ------------------
            #Doing so we'll transform by_lane_structured and by_direction_structured from lists of dict of dict to a simpler lists of dict

            # ------------------ Inserting the record_indexes as key-vale pairs into the dictionaries ------------------

            for by_lane_element in by_lane_structured:
                by_lane_element_list_idx = by_lane_structured.index(by_lane_element)
                by_lane_record_idx = list(by_lane_element.keys())[0] #Obtaining the "record_index", which is the string that represents the dictionary which contains the data for a specific day and lane

                by_lane_structured[by_lane_element_list_idx][by_lane_record_idx]["by_lane_record_index"] = by_lane_record_idx


            for by_direction_element in by_direction_structured:
                by_direction_element_list_idx = by_direction_structured.index(by_direction_element)
                by_direction_record_idx = list(by_direction_element.keys())[0] #Obtaining the "record_index", which is the string that represents the dictionary which contains the data for a specific day and lane

                by_direction_structured[by_direction_element_list_idx][by_direction_record_idx]["by_direction_record_index"] = by_direction_record_idx


            # ------------------ Removing the record_indexes and transforming the old lists of dict of dict to simpler lists of dict ------------------


            for element in by_lane_structured:
                by_lane_element_list_idx = by_lane_structured.index(element)
                by_lane_structured[by_lane_element_list_idx] = list(element.values())[0]

            for element in by_direction_structured:
                by_direction_element_list_idx = by_direction_structured.index(element)
                by_direction_structured[by_direction_element_list_idx] = list(element.values())[0]



            # ------------------ Dataframes creation and printing ------------------


            print("\n\n----------------- By Hour Structured -----------------")
            #pprint.pp(by_hour_structured)
            #print(by_hour_structured)

            by_hour_df = pd.DataFrame(by_hour_structured)
            print(by_hour_df)


            print("\n\n----------------- By Lane Structured -----------------")
            #pprint.pp(by_lane_structured)
            #print(by_lane_structured)

            by_lane_df = pd.DataFrame(by_lane_structured)
            print(by_lane_df)


            print("\n\n----------------- By Direction Structured -----------------")
            #pprint.pp(by_direction_structured)
            #print(by_direction_structured)

            by_direction_df = pd.DataFrame(by_direction_structured)
            print(by_direction_df)


            #print("\n\n")

            print("\n\n")






            #TODO SORTING THE DICTIONARY KEYS ISN'T IMPORTANT, WE CAN JUST LET PANDAS DO IT WITH THE "columns" ATTRIBUTE WHEN CREATING THE DF. https://stackoverflow.com/questions/75441918/dataframe-from-list-of-dicts-with-relative-order-of-keys-maintained-in-columns




            return None #TODO RETURN CLEANED DATA IN THREE DIFFERENT DATAFRAMES AS DESCRIBED ON THE PAPER NOTEBOOK










    #TODO THIS WILL EXPORT ALL THE CLEANED DATA INTO SEPARATED FILES AND CLEAN EACH FILE INDIVIDUALLY THROUGH THE PIPELINE.
    # IN THE DATA EXPLORATION FILE WE'LL CREATE A FOR LOOP AND USE THE cleaning_pipeline() FUNCTION TO CLEAN EACH FILE.
    # THIS PROCESS WILL BE REPEATED BOTH FOR TRAFFIC VOLUMES AND AVERAGE SPEED, EACH ONE WITH THEIR OWN CUSTOM CLEANING PIPELINE
    def cleaning_pipeline(self, trp_file_path: str, volumes_file_path: str):

        volumes = import_volumes_data(volumes_file_path)
        trp_data = import_TRPs_info()

        self.data_overview(trp_data=trp_data, volumes_data=volumes, verbose=True)
        self.clean_traffic_volumes_data(volumes)

        return None


    def execute_cleaning(self, volumes_file_path: str):

        self.cleaning_pipeline(trp_file_path=traffic_registration_points_path, volumes_file_path=volumes_file_path)

        return None
























