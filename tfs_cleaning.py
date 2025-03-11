from tfs_ops_settings import *
from tfs_utilities import *
import numpy as np
import json
import datetime
from datetime import datetime
import os
import pandas as pd
import pprint

from sklearn.linear_model import Lasso
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


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


    #Executing multiple imputation to get rid of NaNs using the MICE method (Multiple Imputation by Chained Equations)
    @staticmethod
    def impute_missing_values(data: pd.DataFrame):
        '''
        This function should only be supplied with numerical columns-only dataframes
        '''

        lr = Lasso(random_state=100) #Using Lasso regression (L1 Penalization) to get better results in case of non-informative columns present in the data (coverage data, because their values all the same)

        mice_imputer = IterativeImputer(estimator=lr, random_state=100, verbose=0, imputation_order="arabic", initial_strategy="median") #Imputation order is set to arabic so that the imputations start from the right (so from the traffic volume columns)
        data = pd.DataFrame(mice_imputer.fit_transform(data), columns=data.columns) #Fitting the imputer and processing all the data columns except the date one

        return data



class TrafficVolumesCleaner(Cleaner):

    def __init__(self):
        super().__init__()


    @staticmethod
    def retrieve_trp_data_from_volumes_file(trp_data, volumes_data: dict):

        trp_id = volumes_data["trafficData"]["trafficRegistrationPoint"]["id"]
        trp_data = [i for i in trp_data["trafficRegistrationPoints"] if i["id"] == trp_id]  # Finding the data for the specific TRP taken in consideration by iterating on all the TRPs available in the trp_file
        trp_data = trp_data[0]

        return trp_data


    #This function is only to give the user an overview of the data which we're currently cleaning, and some specific information about the TRP (Traffic Registration Point) which has collected it
    def data_overview(self, trp_data, volumes_data: dict, verbose: bool):

        #print(volumes_data)

        trp_data = self.retrieve_trp_data_from_volumes_file(trp_data, volumes_data)

        #print(trp_data)

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

        elif verbose is False:

            print("******** Traffic Registration Point Information ********")

            print("ID: ", trp_data["id"])
            print("Name: ", trp_data["name"])

        return None


    def restructure_traffic_volumes_data(self, volumes_payload):

        by_hour_structured = {"trp_id": [],
                              "volume": [],
                              "coverage": [],
                              "year": [],
                              "month": [],
                              "day": [],
                              "hour": []}

        by_lane_structured = {"trp_id": [],
                              "volume": [],
                              "coverage": [],
                              "year": [],
                              "month": [],
                              "day": [],
                              "hour": [],
                              "lane": []}

        by_direction_structured = {"trp_id": [],
                                   "volume": [],
                                   "coverage": [],
                                   "year": [],
                                   "month": [],
                                   "day": [],
                                   "hour": [],
                                   "direction": []}


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

            print("Directions available: ", directions)


            # ------------------ Finding all unique days in which registrations took place ------------------

            registration_dates = [] #To find all the unique days

            for n in nodes:

                #print(n)

                registration_dt = n["node"]["from"][:-6] #Only keeping the datetime without the +00:00 at the end
                #print("Registration DT: ", registration_dt)
                registration_dates.append(registration_dt)

            registration_dates = set(registration_dates) #Removing duplicates and keeping the time as well. This will be needed to extract the hour too
            unique_registration_dates = set([r_date[:10] for r_date in registration_dates]) #Removing duplicates, this one will only keep the first 10 characters of the date, which comprehend just the date without the time. This is needed to know which are the unique days when data has been recorded
            print("Number of unique registration dates: ", len(unique_registration_dates))


            # ------------------ Extracting the data from JSON file and converting it into tabular format ------------------

            for node in nodes:

                # ---------------------- Fetching registration datetime ----------------------

                #This is the datetime which will be representative of a volume, specifically, there will be multiple datetimes with the same day
                # to address this fact we'll just re-format the data to keep track of the day, but also maintain the volume values for each hour
                registration_datetime = node["node"]["from"][:-6] #Only keeping the datetime without the +00:00 at the end

                registration_datetime = datetime.strptime(registration_datetime, "%Y-%m-%dT%H:%M:%S")
                year = registration_datetime.strftime("%Y")
                month = registration_datetime.strftime("%m")
                day = registration_datetime.strftime("%Y-%m-%d")
                hour = registration_datetime.strftime("%H")

                #print(registration_datetime)

                #print(day)
                #print(hour)


                #TODO CHECK THERE ARE NO MISSING DATES, HOURS AND SO ON
                #TODO GENERATE DATE RANGE WITH DATETIME AND LIST COMPREHENSION
                #TODO RETRIEVE ALL THEORETICAL HOURS WITH THE SPECIFIC FUNCTION CREATED
                #TODO EXECUTE A CHECK ON ALL NODES OF THE TRP'S VOLUME DATA (VOLUMES FILE), CHECK WHICH DATES, HOURS, ETC. ARE MISSING AND CREATE THE MISSING ROWS (WITH MULTIPLE LISTS (ONE FOR EACH VARIABLE)) TO ADD BEFORE(!) THE START OF THE FOR CYCLE BELOW!
                #TODO WHEN ALL THE ROWS HAVE BEEN CREATED AND INSERTED IN THE FOR CYCLE BELOW, SORT THE ROWS BY YEAR, MONTH, DAY, HOUR IN DESCENDING ORDER


                # ----------------------- Total volumes section -----------------------

                total_volume = node["node"]["total"]["volumeNumbers"]["volume"] if node["node"]["total"]["volumeNumbers"] is not None else None #In some cases the volumeNumbers key could have null as value, so the "volume" key won't be present. In that case we'll directly insert None as value with an if statement
                coverage_perc = node["node"]["total"]["coverage"]["percentage"]

                by_hour_structured["year"].append(year)
                by_hour_structured["month"].append(month)
                by_hour_structured["day"].append(day)
                by_hour_structured["hour"].append(hour)
                by_hour_structured["volume"].append(total_volume)
                by_hour_structured["coverage"].append(coverage_perc)


                #   ----------------------- By lane section -----------------------

                lanes_data = node["node"]["byLane"] #Extracting byLane data

                #Every lane's data is kept isolated from the other lanes' data, so a for cycle is needed to extract all the data from each lane's section
                for lane in lanes_data:

                    road_link_lane_number = lane["lane"]["laneNumberAccordingToRoadLink"]
                    lane_volume = lane["total"]["volumeNumbers"]["volume"] if lane["total"]["volumeNumbers"] is not None else None #In some cases the volumeNumbers key could have null as value, so the "volume" key won't be present. In that case we'll directly insert None as value with an if statement
                    lane_coverage = lane["total"]["coverage"]["percentage"]

                    # ------- XXXXXXX -------

                    by_lane_structured["year"].append(year)
                    by_lane_structured["month"].append(month)
                    by_lane_structured["day"].append(day)
                    by_lane_structured["hour"].append(hour)
                    by_lane_structured["volume"].append(lane_volume)
                    by_lane_structured["coverage"].append(lane_coverage)
                    by_lane_structured["lane"].append(road_link_lane_number)


                #   ----------------------- By direction section -----------------------

                by_direction_data = node["node"]["byDirection"]

                # Every direction's data is kept isolated from the other directions' data, so a for cycle is needed
                for direction_section in by_direction_data:
                    heading = direction_section["heading"]
                    direction_volume = direction_section["total"]["volumeNumbers"]["volume"] if direction_section["total"]["volumeNumbers"] is not None else None #In some cases the volumeNumbers key could have null as value, so the "volume" key won't be present. In that case we'll directly insert None as value with an if statement
                    direction_coverage = direction_section["total"]["coverage"]["percentage"]

                    by_direction_structured["year"].append(year)
                    by_direction_structured["month"].append(month)
                    by_direction_structured["day"].append(day)
                    by_direction_structured["hour"].append(hour)
                    by_direction_structured["volume"].append(direction_volume)
                    by_direction_structured["coverage"].append(direction_coverage)
                    by_direction_structured["direction"].append(heading)


                    #TODO THE SAME PRINCIPLE AS BEFORE APPLIES HERE, SAVE ALL THE AVAILABLE DIRECTIONS IN THE TRP'S METADATA FILE



            # ------------------ Ensuring that every dictionary in by_hour/by_lane/by_direction_structured has the same number of key-value pairs ------------------


            hours = [f"{i:02}" for i in range(24)] #Generating 24 elements starting from 00 to 23
            by_x_structured_volume_keys = retrieve_volume_columns() #These can be used both for by_hour_structured, by_lane_structured and for by_direction_structured

            #print(by_x_structured_volume_keys)

            #These are all the coverage keys that should exist when a measurement point has coverage data for all hours
            by_hour_coverage_keys = [f"cvg{i:02}" for i in range(24)]
            by_lane_coverage_keys = [f"lane_cvg{i:02}" for i in range(24)]
            by_direction_coverage_keys = [f"direction_cvg{i:02}" for i in range(24)]

            #To better understand what a volume or a coverage key look like let's make some example:
            # - Volume keys: v00, v01, v02, ..., v22, v23, v24
            # - Coverage keys: cvg00, cvg01, cvg02, ..., cvg22, cvg23, cvg24

            #There are variations of key names for coverage, so:
            # - by_lane coverage keys: lane_cvg00, lane_cvg01, lane_cvg02, ..., lane_cvg22, lane_cvg23, lane_cvg24
            # - by_direction coverage keys: direction_cvg00, direction_cvg01, direction_cvg02, ..., direction_cvg22, direction_cvg23, direction_cvg24



            # ------------------ by_hour_structured check ------------------

            #by_hour_structured is a list of dict
            #Each dict has keys and values, which correspond to either volume or coverage for a certain hour
            #In some cases data for specific hours could be missing, thus we want to ensure that, even if data is missing, we'll have those keys anyways
            #This is because when creating the dataframes we must ensure that the number of keys (and so of columns in the future df) is the same for each dictionary (every dictionary will become a record of the df)
            #Every dictionary in by_hour_structured is an element of the list (since by_hour_structured is a list of dict)
            #To ensure we'll have the same keys and number of keys for each dictionary in by_hour_structured we'll create another dict with the list index as key and a list of the keys available as values
            by_hour_structured_keys = {by_hour_structured.index(list_element): list(list_element.keys()) for list_element in by_hour_structured}


            #The next step in tidying up the by_hours_structured list of dict is to determine which keys are missing in each element of the list
            #We can do so with a dict comprehension coupled with a list comprehension (both for better performances)
            #The idea is: we'll create a dict which takes the by_hour_structured list index of each element as key and we'll insert the keys which aren't included in by_x_structured_volume_keys as values
            #This way we'll have the list index of every element and all missing keys (if any)
            #We'll execute this process both for volume and coverage keys
            by_hour_volume_keys_to_add = {element: [to_add for to_add in by_x_structured_volume_keys if to_add not in keys] for element, keys in by_hour_structured_keys.items()}
            by_hour_cvg_keys_to_add = {element: [to_add for to_add in by_hour_coverage_keys if to_add not in keys] for element, keys in by_hour_structured_keys.items()}


            #During the process of identifying the missing keys we create a list for each element, but since not all of them have missing keys there could be empty lists as well
            #Thus, we can address this problem by simply removing all key-value pairs where the value is an empty list
            #Again, we'll repeat this process both for the volume and coverage missing keys dictionaries
            by_hour_volume_keys_to_add = {bh_idx: bh_to_add_element for bh_idx, bh_to_add_element in by_hour_volume_keys_to_add.items() if len(bh_to_add_element) != 0} #Only keeping non-empty key-value pairs. So basically we're only keeping the dictionary pairs which actually contain elements to add
            by_hour_cvg_keys_to_add = {bh_idx: bh_to_add_element for bh_idx, bh_to_add_element in by_hour_cvg_keys_to_add.items() if len(bh_to_add_element) != 0} #Only keeping non-empty key-value pairs. So basically we're only keeping the dictionary pairs which actually contain elements to add

            #print("By hour volume keys to add: ")
            #print(by_hour_volume_keys_to_add)

            #print("By hour coverage keys to add: ")
            #print(by_hour_cvg_keys_to_add, "\n\n")


            # ------------------ Adding missing volume key-value pairs to by_hour_structured ------------------

            # - Adding missing volume keys -
            #Now that we have the missing keys for each list element in a specific dictionary (by_hour_volume_keys_to_add) we can just iterate over it
            # with a for cycle with the .items() method
            #For each element we'll create a dictionary with all the missing keys as keys and None as values
            #Then we'll just use the .update() method and insert it in the correct list element in by_hour_structured using the list_idx
            for list_idx, to_add_elements in by_hour_volume_keys_to_add.items():
                to_add_elements = {el: None for el in to_add_elements} #Adding the missing keys as keys and None as value
                by_hour_structured[list_idx].update(to_add_elements)

            # - Adding missing coverage keys -
            #Same process as before, but with coverage keys
            for list_idx, to_add_elements in by_hour_cvg_keys_to_add.items():
                to_add_elements = {el: None for el in to_add_elements} #Adding the missing keys as keys and None as value
                by_hour_structured[list_idx].update(to_add_elements)


            #Keep in mind that the processes to identify missing keys until now were applied only for the by_hour_structured list of dict, now we'll have to repeat them
            # with some tweaks for the by_lane_structured and by_direction_structured lists



            # ------------------ by_lane_structured check ------------------

            #by_lane_structured is a list of dict of dict, this means that every element of the list contains one key-value pair where the
            # key is a unique identifier (record index) of the list element (formatted like YYYY-MM-DDlX) and the value is a dictionary which is made by
            # many key-value pairs where the keys are either traffic volumes or coverage and the values contain the corresponding data


            #Since as we already said by_lane_structured is a list of dict of dict we must first of all able to access every part of the data structure
            #To determine the relationship between the list element, record index (YYYY-MM-DDlX) and the keys and values which actually represent the volumes data by lane
            # we'll create a dictionary which is composed by the list index of each element and the record index
            #Then we'll replace each value (so, the record index) with the keys available for the sub-dictionary
            #Essentially what we're doing is: we're taking the record index to access the keys of the sub-dictionary and replace the record index with the fetched keys afterwards
            #By doing so, we'll have a dictionary with the by_lane_structured list indexes as keys and the available data keys (as a list, one for each by_lane_structured list element) as values
            by_lane_structured_keys = {by_lane_structured.index(list_element): list(list_element.values()) for list_element in by_lane_structured}
            by_lane_structured_keys = {e: list(k[0].keys()) for e, k in by_lane_structured_keys.items()} #Replacing record indexes with the available keys for the by_lane_structured sub-dicts

            #Once determined the keys available we'll execute a process identical to the one described previously in the by_hour_section
            #We'll determine, again, the keys to add (the missing keys, both volume and coverage ones)
            by_lane_volume_keys_to_add = {element: [to_add for to_add in by_x_structured_volume_keys if to_add not in keys] for element, keys in by_lane_structured_keys.items()}
            by_lane_cvg_keys_to_add = {element: [to_add for to_add in by_lane_coverage_keys if to_add not in keys] for element, keys in by_lane_structured_keys.items()}

            #Once again we'll remove all the elements of the dictionary which have empty missing keys lists
            by_lane_volume_keys_to_add = {bl_idx: bl_to_add_element for bl_idx, bl_to_add_element in by_lane_volume_keys_to_add.items() if len(bl_to_add_element) != 0} #Only keeping non-empty key-value pairs. So basically we're only keeping the dictionary pairs which actually contain elements to add
            by_lane_cvg_keys_to_add = {bl_idx: bl_to_add_element for bl_idx, bl_to_add_element in by_lane_cvg_keys_to_add.items() if len(bl_to_add_element) != 0} #Only keeping non-empty key-value pairs. So basically we're only keeping the dictionary pairs which actually contain elements to add

            #print("By lane volume keys to add: ")
            #print(by_lane_volume_keys_to_add)

            #print("By lane coverage keys to add: ")
            #print(by_lane_cvg_keys_to_add, "\n\n")


            # ------------------ Adding missing key-value pairs to by_lane_structured ------------------

            # - Adding missing volume keys -
            #This is a particularly difficult step of the whole process, so we'll describe it step by step in the code
            #1. We'll create as for by_hour_structured a dict with the missing keys and None as their values
            for list_idx, to_add_elements in by_lane_volume_keys_to_add.items(): # <- VOLUME MISSING KEYS
                to_add_elements = {el: None for el in to_add_elements} #Adding the missing keys as keys and None as value

                #2. We'll iterate over every element of the by_lane_structured, verify if it's in the list of the by_lane_structured elements which have missing keys
                # if so, we'll fetch the location of the sub-dictionary which has missing keys and insert them (with None as values) with the .update method
                for el in by_lane_structured:
                    if by_lane_structured.index(el) == list_idx: #If this if statement is true then the element el of by_lane_structured has missing keys
                        bl_loc_key = list(el.keys())[0] #Finding the record index for the dictionary with missing keys

                        by_lane_structured[list_idx][bl_loc_key].update(to_add_elements)

            #- Adding missing coverage keys -
            #2.1 Same principle as above applies here, but for coverage keys
            for list_idx, to_add_elements in by_lane_cvg_keys_to_add.items(): # <- COVERAGE MISSING KEYS
                to_add_elements = {el: None for el in to_add_elements} #Adding the missing keys as keys and None as value

                for el in by_lane_structured:
                    if by_lane_structured.index(el) == list_idx:
                        bl_loc_key = list(el.keys())[0]

                        by_lane_structured[list_idx][bl_loc_key].update(to_add_elements)


            # ------------------ by_direction_structured check ------------------

            #The same process described in the by_lane_structured section applies here since by_direction_structured is a list of dict of dict too
            #The logic is the same, the only things that change are the variables name

            by_direction_structured_keys = {by_direction_structured.index(list_element): list(list_element.values()) for list_element in by_direction_structured}
            by_direction_structured_keys = {e: list(k[0].keys()) for e, k in by_direction_structured_keys.items()}

            by_direction_volume_keys_to_add = {element: [to_add for to_add in by_x_structured_volume_keys if to_add not in keys] for element, keys in by_direction_structured_keys.items()}
            by_direction_cvg_keys_to_add = {element: [to_add for to_add in by_direction_coverage_keys if to_add not in keys] for element, keys in by_direction_structured_keys.items()}

            #print("By direction volume keys to add: ")
            #print(by_direction_volume_keys_to_add)

            #print("By direction coverage keys to add: ")
            #print(by_direction_cvg_keys_to_add, "\n\n")


            # ------------------ Adding missing key-value pairs to by_direction_structured ------------------

            # - Adding missing volume keys -
            #Same principles as before apply here
            for list_idx, to_add_elements in by_direction_volume_keys_to_add.items():
                to_add_elements = {el: None for el in to_add_elements}  # Adding the missing keys as keys and None as value

                for el in by_direction_structured:
                    if by_direction_structured.index(el) == list_idx:
                        bd_loc_key = list(el.keys())[0]

                        by_direction_structured[list_idx][bd_loc_key].update(to_add_elements)

            # - Adding missing coverage keys -
            for list_idx, to_add_elements in by_direction_cvg_keys_to_add.items():
                to_add_elements = {el: None for el in to_add_elements}  # Adding the missing keys as keys and None as value

                for el in by_direction_structured:
                    if by_direction_structured.index(el) == list_idx:
                        bd_loc_key = list(el.keys())[0]

                        by_direction_structured[list_idx][bd_loc_key].update(to_add_elements)



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


            #print("\n\n----------------- By Hour Structured -----------------")
            #pprint.pp(by_hour_structured)
            #print(by_hour_structured)

            by_hour_df = pd.DataFrame(by_hour_structured)
            by_hour_df = by_hour_df.reindex(sorted(by_hour_df.columns), axis=1)
            #print(by_hour_df)


            #print("\n\n----------------- By Lane Structured -----------------")
            #pprint.pp(by_lane_structured)
            #print(by_lane_structured)

            #by_lane_df = pd.DataFrame(by_lane_structured)
            #by_lane_df = by_lane_df.reindex(sorted(by_lane_df.columns), axis=1)
            #print(by_lane_df)


            #print("\n\n----------------- By Direction Structured -----------------")
            #pprint.pp(by_direction_structured)
            #print(by_direction_structured)

            #by_direction_df = pd.DataFrame(by_direction_structured)
            #by_direction_df = by_direction_df.reindex(sorted(by_direction_df.columns), axis=1)
            #print(by_direction_df)

            #print("\n\n")

            return by_hour_df #TODO IN THE FUTURE SOME ANALYSES COULD BE EXECUTED WITH THE by_lane_df OR by_direction_df, BUT FOR NOW IT'S BETTER TO SAVE PERFORMANCES AND MEMORY BY JUST RETURNING TWO STRINGS AND NOT EVEN CREATING THE DFs


    #This function is design only to clean by_hour data since that's the data we're going to use for the main purposes of this project
    def clean_traffic_volumes_data(self, by_hour_df: pd.DataFrame):

        volume_columns = retrieve_volume_columns()

        #Short dataframe overview
        #print("Short overview on the dataframe: \n", by_hour_df.describe())

        #Checking dataframe columns
        #print("Dataframe columns: \n", by_hour_df.columns, "\n")

        #If all values aren't 0 then execute multiple imputation to fill NaNs:

        # ------------------ Execute multiple imputation with MICE (Multiple Imputation by Chain of Equations) ------------------

        dates = by_hour_df["date"]  # Setting apart the dates column to execute MICE (multiple imputation) only on numerical columns and then merging that back to the df

        try:
            cleaner = Cleaner()
            by_hour_df = cleaner.impute_missing_values(by_hour_df.drop("date", axis=1))

            by_hour_df["date"] = dates

        except ValueError:
            print("\03391mValue error raised. Continuing with the cleaning\0330m")
            return None


        # ------------------ Data types transformation ------------------

        for col in volume_columns:
            by_hour_df[col] = by_hour_df[col].astype("int")

        # print(by_hour_df, "\n")

        # print("Data types: ")
        # print(by_hour_df.dtypes)

        print("\n\n")

        return by_hour_df


    def export_traffic_volumes_data(self, by_hour: pd.DataFrame, volumes_file_path, trp_data):

        file_name = volumes_file_path.split("/")[-1].replace(".json", "C.csv")

        clean_traffic_volumes_folder_path = get_clean_traffic_volumes_folder_path()
        trp_id = trp_data["id"]

        file_path = clean_traffic_volumes_folder_path + file_name #C stands for "cleaned"
        #print(file_path)

        try:
            by_hour.to_csv(file_path)
            print(f"TRP: {trp_id} data exported correctly\n\n")
        except AttributeError:
            print(f"\033[91mCouldn't export {trp_id} TRP volumes data\033[0m")

        return None


    def cleaning_pipeline(self, volumes_file_path: str):

        volumes = import_volumes_data(volumes_file_path)
        trp_data = import_TRPs_data()

        self.data_overview(trp_data=trp_data, volumes_data=volumes, verbose=False)
        by_hour_df = self.restructure_traffic_volumes_data(volumes) #TODO IN THE FUTURE SOME ANALYSES COULD BE EXECUTED WITH THE by_lane_df OR by_direction_df, IN THAT CASE WE'LL REPLACE THE _, _ WITH by_lane_df, by_direction_df

        #TODO TO-NOTE TRPs WITH NO DATA WON'T BE INCLUDED IN THE ROAD NETWORK CREATION SINCE THEY WON'T RETURN A DATAFRAME (BECAUSE THEY DON'T HAVE DATA TO STORE IN A DF)

        if by_hour_df is not None:
            by_hour_df = self.clean_traffic_volumes_data(by_hour_df)

            if by_hour_df is not None:
                self.export_traffic_volumes_data(by_hour_df, volumes_file_path, self.retrieve_trp_data_from_volumes_file(trp_data, volumes))

        elif by_hour_df is None:
            pass #A warning for empty nodes is given during the restructuring section


        print("--------------------------------------------------------\n\n")

        return None


    def execute_cleaning(self, volumes_file_path: str):

        self.cleaning_pipeline(volumes_file_path=volumes_file_path)

        return None



class AverageSpeedCleaner(Cleaner):

    def __init__(self):
        super().__init__()


    @staticmethod
    def retrieve_trp_id_from_avg_speed_file(filename: str):
        trp_id = filename.split("_")[0]
        return trp_id


    def retrieve_trp_data_from_avg_speed_file(self, avg_speed_filename):

        avg_speed_folder_path = get_raw_average_speed_folder_path() #Getting the raw average speed folder path
        trp_id = self.retrieve_trp_id_from_avg_speed_file(avg_speed_filename) #Combining the raw average speed folder path with the filename of the file we want to check out
        trp_data = import_TRPs_data()  #All TRPs data retrieval

        trp_data = [i for i in trp_data["trafficRegistrationPoints"] if i["id"] == trp_id]  #Finding the data for the specific TRP taken in consideration by iterating on all the TRPs available in the trp_file
        trp_data = trp_data[0]

        return trp_data


    def data_overview(self, trp_data, verbose: bool):
        # Since the prints below are all the same (except for one) we could technically create a workaround to avoid having to repeat these lines, but it would complicate a lot something way simpler (just prints).
        # Thus, for readability purposes we're gonna repeat the same prints (except for one) as in the TrafficVolumeCleaner class

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


        elif verbose is False:

            print("******** Traffic Registration Point Information ********")

            print("ID: ", trp_data["id"])
            print("Name: ", trp_data["name"])

        print("\n\n")

        return None


    def clean_avg_speed_data(self, avg_speed_data):

        avg_speed_data = avg_speed_data.drop(columns=["traffic_volume", "lane"], axis=1)

        avg_speed_data["coverage"] = avg_speed_data["coverage"].apply(lambda x: x.replace(",", ".")) #Replacing commas with dots
        avg_speed_data["coverage"] = avg_speed_data["coverage"].astype("float") #Converting the coverage column to float data type
        avg_speed_data["coverage"] = avg_speed_data["coverage"]*100 #Transforming the coverage values from 0.0 to 1.0 to 0 to 100 (percent)

        avg_speed_data["mean_speed"] = avg_speed_data["mean_speed"].replace(",", ".", regex=True) #The regex=True parameter is necessary, otherwise the function, for some reason, won't be able to perform the replacement
        avg_speed_data["mean_speed"] = avg_speed_data["mean_speed"].astype("float") #Converting the mean_speed column to float data type

        avg_speed_data["percentile_85"] = avg_speed_data["percentile_85"].replace(",", ".", regex=True) #The regex=True parameter is necessary, otherwise the function, for some reason, won't be able to perform the replacement
        avg_speed_data["percentile_85"] = avg_speed_data["percentile_85"].astype("float")  # Converting the percentile_85 column to float data type

        avg_speed_data["hour_start"] = avg_speed_data["hour_start"].apply(lambda x: x[:2]) #Keeping only the first two characters (which represent only the hour data)

        #print(avg_speed_data.isna().sum())
        #print(avg_speed_data.dtypes)

        # ------------------ Initial data types transformation ------------------

        avg_speed_data["trp_id"] = avg_speed_data["trp_id"].astype("str")
        avg_speed_data["date"] = pd.to_datetime(avg_speed_data["date"])
        avg_speed_data["hour_start"] = avg_speed_data["hour_start"].astype("str") #This is done to then extract hour values only from the string using string splitting


        # ------------------ Multiple imputation to fill NaN values ------------------

        non_mice_cols = ["trp_id", "date", "hour_start"]
        df_non_mice_cols = avg_speed_data[non_mice_cols] #To merge them later into the NaN filled dataframe

        avg_speed_data = avg_speed_data.drop(columns=non_mice_cols, axis=1) #Columns to not include for Multiple Imputation By Chained Equations (MICE)

        try:
            cleaner = Cleaner()
            avg_speed_data = cleaner.impute_missing_values(avg_speed_data)

            print("Multiple imputation on average speed data executed successfully\n\n")

            #print(avg_speed_data.isna().sum())

        except ValueError:
            print("\03391mValue error raised. Continuing with the cleaning\0330m")
            return None

        #Merging not numerical columns back into the MICEed dataframe
        avg_speed_data["trp_id"] = df_non_mice_cols["trp_id"]
        avg_speed_data["date"] = df_non_mice_cols["date"]
        avg_speed_data["hour_start"] = df_non_mice_cols["hour_start"]

        print("Dataframe overview: \n", avg_speed_data.head(15), "\n")
        print("Basic descriptive statistics on the dataframe: \n", avg_speed_data.describe(), "\n")


        #Determining the days range of the data
        t_min = avg_speed_data["date"].min()
        t_max = avg_speed_data["date"].max()


        print("Registrations time-range: ")
        print("First day of data registration: ", t_min)
        print("Last day of data registration: ", t_max, "\n\n")


        # ------------------ Restructuring the data ------------------
        #Here we'll restructure the data into a more convenient, efficient and readable format
        #The mean_speed will be defined by hour and not by hour AND lane.
        #This is because the purpose of this project is to create a predictive ml/statistical model for each type of road
        #Also, generalizing the lanes data wouldn't make much sense because the lanes in each street may have completely different data, not because of the traffic behaviour, but because of the location of the TRP itself
        #If we were to create a model for each TRP, then it could make some sense, but that's not the goal of this project

        #agg_data will be a list of dict which we'll use to create a dataframe afterward
        agg_data = []

        #TODO REMOVE THE [:2] AFTER TESTING
        for ud in avg_speed_data["date"].unique()[:2]:

            day_data = avg_speed_data.query(f"date == '{ud}'")
            #print(day_data)

            #hourly_data will be a temporary dictionary which contains the data for the ud (unique day) taken in consideration during the execution of for loop
            #The data will be added during the nested loop below
            hourly_data = {"date": ud}

            for h in day_data["hour_start"].unique():
                #Using the median to have a more robust indicator which won't be influenced by outliers as much as the mean
                hourly_data[f"mean_speed{h}"] = np.round(np.median(day_data[["mean_speed", "hour_start"]].query(f"hour_start == '{h}'")["mean_speed"]), decimals=2)
                hourly_data[f"p85_{h}"] = np.round(np.median(day_data[["percentile_85", "hour_start"]].query(f"hour_start == '{h}'")["percentile_85"]), decimals=2)
                hourly_data[f"cvg{h}"] = np.round(np.median(day_data[["coverage", "hour_start"]].query(f"hour_start == '{h}'")["coverage"]), decimals=2)

            agg_data.append(hourly_data)

        #print(agg_data)

        #The old avg_data dataframe will be overwritten by this new one which will have all the previous data, but with a new structure
        avg_speed_data = pd.DataFrame(agg_data)
        avg_speed_data = avg_speed_data.reindex(sorted(avg_speed_data.columns), axis=1)

        print(avg_speed_data.head(15))










        return None



    def execute_cleaning(self, file_path, file_name):
        '''
        The avg_speed_file_path parameter is the path to the average speed file the user wants to analyze
        The avg_speed_file_name parameter is just the name of the file, needed for secondary purposes or functionalities

        '''

        trp_data = self.retrieve_trp_data_from_avg_speed_file(avg_speed_filename=file_name)
        average_speed_data = import_avg_speed_data(file_path=file_path)

        self.data_overview(trp_data, verbose=True)

        self.clean_avg_speed_data(average_speed_data)

        #TODO CHECK VERY CAREFULLY THE FIRST AND LAST DATES FOR EVERY FILE



        return None












