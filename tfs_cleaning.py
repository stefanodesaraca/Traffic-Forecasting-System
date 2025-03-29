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


    #This function is only to give the user an overview of the data which we're currently cleaning, and some specific information about the TRP (Traffic Registration Point) which has collected it
    def data_overview(self, volumes_data: dict, verbose: bool = True) -> None:

        #print(volumes_data)

        trp_id = volumes_data["trafficData"]["trafficRegistrationPoint"]["id"]
        trp_metadata = get_trp_metadata(trp_id)


        if verbose is True:

            print("******** Traffic Registration Point Information ********")

            print("ID: ", trp_metadata["trp_id"])
            print("Name: ", trp_metadata["name"])
            print("Road category: ", trp_metadata["road_category"])
            print("Coordinates: ")
            print(" - Lat: ", trp_metadata["lat"])
            print(" - Lon: ", trp_metadata["lon"])
            print("County name: ", trp_metadata["county_name"])
            print("County number: ", trp_metadata["county_number"])
            print("Geographic number: ", trp_metadata["geographic_number"])
            print("Country part: ", trp_metadata["country_part"])
            print("Municipality name: ", trp_metadata["municipality_name"])

            print("Traffic registration type: ", trp_metadata["traffic_registration_type"])
            print("Data time span: ")
            print(" - First data: ", trp_metadata["first_data"])
            print(" - First data with quality metrics: ", trp_metadata["first_data_with_quality_metrics"])
            print(" - Latest data: ")
            print("   > Volume by day: ", trp_metadata["latest_volume_by_day"])
            print("   > Volume by hour: ", trp_metadata["latest_volume_byh_hour"])
            print("   > Volume average daily by year: ", trp_metadata["latest_volume_average_daily_by_year"])
            print("   > Volume average daily by season: ", trp_metadata["latest_volume_average_daily_by_season"])
            print("   > Volume average daily by month: ", trp_metadata["latest_volume_average_daily_by_month"])
            print("Number of data nodes: ", trp_metadata["number_of_data_nodes"])

            print("\n")

            #print(volumes_data)

        elif verbose is False:

            print("******** Traffic Registration Point Information ********")

            print("ID: ", trp_metadata["trp_id"])
            print("Name: ", trp_metadata["name"])

        return None


    def restructure_traffic_volumes_data(self, volumes_payload):

        by_hour_structured = {"trp_id": [],
                              "volume": [],
                              "coverage": [],
                              "year": [],
                              "month": [],
                              "week": [],
                              "day": [],
                              "hour": []}

        by_lane_structured = {"trp_id": [],
                              "volume": [],
                              "coverage": [],
                              "year": [],
                              "month": [],
                              "week": [],
                              "day": [],
                              "hour": [],
                              "lane": []}

        by_direction_structured = {"trp_id": [],
                                   "volume": [],
                                   "coverage": [],
                                   "year": [],
                                   "month": [],
                                   "week": [],
                                   "day": [],
                                   "hour": [],
                                   "direction": []}


        # ------------------ Data payload extraction ------------------

        nodes = volumes_payload["trafficData"]["volume"]["byHour"]["edges"]
        number_of_nodes = len(nodes)

        trp_id = volumes_payload["trafficData"]["trafficRegistrationPoint"]["id"]

        #print(nodes)

        if number_of_nodes == 0:
            print(f"\033[91mNo data found for TRP: {volumes_payload['trafficData']['trafficRegistrationPoint']['id']}\033[0m\n\n")

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


            registration_dates = [n["node"]["from"][:-6] for n in nodes] #Only keeping the datetime without the +00:00 at the end
            #print(registration_dates)

            registration_dates = set(registration_dates) #Removing duplicates and keeping the time as well. This will be needed to extract the hour too
            unique_registration_dates = set([r_date[:10] for r_date in registration_dates]) #Removing duplicates, this one will only keep the first 10 characters of the date, which comprehend just the date without the time. This is needed to know which are the unique days when data has been recorded
            print("Number of unique registration dates: ", len(unique_registration_dates))


            # TODO EXECUTE A CHECK ON ALL NODES OF THE TRP'S VOLUME DATA (VOLUMES FILE), CHECK WHICH DATES, HOURS, ETC. ARE MISSING AND CREATE THE MISSING ROWS (WITH MULTIPLE LISTS (ONE FOR EACH VARIABLE)) TO ADD BEFORE(!) THE START OF THE FOR CYCLE BELOW!
            # TODO WHEN ALL THE ROWS HAVE BEEN CREATED AND INSERTED IN THE FOR CYCLE BELOW, SORT THE ROWS BY YEAR, MONTH, DAY, HOUR IN DESCENDING ORDER


            available_day_hours = {d: [] for d in unique_registration_dates}

            for rd in registration_dates:
                available_day_hours[rd[:10]].append(datetime.strptime(rd, "%Y-%m-%dT%H:%M:%S").strftime("%H"))

            #print(available_day_hours)

            theoretical_hours = retrieve_theoretical_hours_columns()

            missing_hours_by_day = {d: [h for h in theoretical_hours if h not in available_day_hours[d]] for d in available_day_hours.keys()}
            missing_hours_by_day = {d: l for d, l in missing_hours_by_day.items() if len(l) != 0} #Removing elements with empty lists
            print("Missing hours for each day: ", missing_hours_by_day)


            first_registration_date = min(unique_registration_dates)
            last_registration_date = max(unique_registration_dates)

            print("First registration day available: ", first_registration_date)
            print("Last registration day available: ", last_registration_date)

            theoretical_days_available = pd.date_range(start=first_registration_date, end=last_registration_date, freq="d")
            missing_days = [d for d in theoretical_days_available if str(d)[:10] not in unique_registration_dates]

            #print("Theoretical days available: ", [str(d)[:10] for d in theoretical_days_available])
            print("Missing days: ", missing_days)


            for d, mh in missing_hours_by_day.items():
                for h in mh:
                    by_hour_structured["trp_id"].append(trp_id)
                    by_hour_structured["volume"].append(None)
                    by_hour_structured["coverage"].append(None)
                    by_hour_structured["year"].append(datetime.strptime(d, "%Y-%m-%d").strftime("%Y"))
                    by_hour_structured["month"].append(datetime.strptime(d, "%Y-%m-%d").strftime("%m"))
                    by_hour_structured["week"].append(datetime.strptime(d, "%Y-%m-%d").strftime("%V"))
                    by_hour_structured["day"].append(datetime.strptime(d, "%Y-%m-%d").strftime("%d"))
                    by_hour_structured["hour"].append(h)

            #pprint.pprint(by_hour_structured)

            #TODO ADDRESS FOR MISSING DAYS PROBLEM, CREATE ALL ATTRIBUTES WITH NONE VALUES FOR EACH MISSING DAY AND ADD IT TO BY_HOUR_STRUCTURED, BY_LANE_STRUCTURED AND BY_DIRECTION_STRUCTURED


            # ------------------ Extracting the data from JSON file and converting it into tabular format ------------------

            for node in nodes:

                # ---------------------- Fetching registration datetime ----------------------

                #This is the datetime which will be representative of a volume, specifically, there will be multiple datetimes with the same day
                # to address this fact we'll just re-format the data to keep track of the day, but also maintain the volume values for each hour
                registration_datetime = node["node"]["from"][:-6] #Only keeping the datetime without the +00:00 at the end

                registration_datetime = datetime.strptime(registration_datetime, "%Y-%m-%dT%H:%M:%S")
                year = registration_datetime.strftime("%Y")
                month = registration_datetime.strftime("%m")
                week = registration_datetime.strftime("%V")
                day = registration_datetime.strftime("%d")
                hour = registration_datetime.strftime("%H")

                #print(registration_datetime)

                #print(day)
                #print(hour)

                # ----------------------- Total volumes section -----------------------

                total_volume = node["node"]["total"]["volumeNumbers"]["volume"] if node["node"]["total"]["volumeNumbers"] is not None else None #In some cases the volumeNumbers key could have null as value, so the "volume" key won't be present. In that case we'll directly insert None as value with an if statement
                coverage_perc = node["node"]["total"]["coverage"]["percentage"] if node["node"]["total"]["coverage"] is not None else None #For less recent data it's possible that sometimes coverage can be null, so we'll address this problem like so

                by_hour_structured["trp_id"].append(trp_id)
                by_hour_structured["year"].append(year)
                by_hour_structured["month"].append(month)
                by_hour_structured["week"].append(week)
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
                    lane_coverage = lane["total"]["coverage"]["percentage"] if lane["total"]["coverage"] is not None else None

                    # ------- XXXXXXX ------- TODO WRITE MISSING TITLES

                    by_lane_structured["trp_id"].append(trp_id)
                    by_lane_structured["year"].append(year)
                    by_lane_structured["month"].append(month)
                    by_lane_structured["week"].append(week)
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
                    direction_coverage = direction_section["total"]["coverage"]["percentage"] if direction_section["total"]["coverage"] is not None else None

                    by_direction_structured["trp_id"].append(trp_id)
                    by_direction_structured["year"].append(year)
                    by_direction_structured["month"].append(month)
                    by_direction_structured["week"].append(week)
                    by_direction_structured["day"].append(day)
                    by_direction_structured["hour"].append(hour)
                    by_direction_structured["volume"].append(direction_volume)
                    by_direction_structured["coverage"].append(direction_coverage)
                    by_direction_structured["direction"].append(heading)


                    #TODO THE SAME PRINCIPLE AS BEFORE APPLIES HERE, SAVE ALL THE AVAILABLE DIRECTIONS IN THE TRP'S METADATA FILE



            # ------------------ Ensuring that XXXXXXXXXXXXXXX ------------------


            #for k in by_hour_structured.keys():
                #print(f"List length for key: {k} = ", len(by_hour_structured[k]))



            # ------------------ Dataframes creation and printing ------------------


            #print("\n\n----------------- By Hour Structured -----------------")
            #pprint.pp(by_hour_structured)
            #print(by_hour_structured)

            by_hour_df = pd.DataFrame(by_hour_structured)
            by_hour_df = by_hour_df.sort_values(by=["hour", "day", "month", "year"], ascending=True)
            #by_hour_df = by_hour_df.reindex(sorted(by_hour_df.columns), axis=1)
            #print(by_hour_df.head(15))


            #print("\n\n----------------- By Lane Structured -----------------")
            #pprint.pp(by_lane_structured)
            #print(by_lane_structured)

            #by_lane_df = pd.DataFrame(by_lane_structured)
            #by_lane_df = by_lane_df.reindex(sorted(by_lane_df.columns), axis=1)
            #print(by_lane_df.head(15))


            #print("\n\n----------------- By Direction Structured -----------------")
            #pprint.pp(by_direction_structured)
            #print(by_direction_structured)

            #by_direction_df = pd.DataFrame(by_direction_structured)
            #by_direction_df = by_direction_df.reindex(sorted(by_direction_df.columns), axis=1)
            #print(by_direction_df.head(15))

            #print("\n\n")

            return by_hour_df #TODO IN THE FUTURE SOME ANALYSES COULD BE EXECUTED WITH THE by_lane_df OR by_direction_df, BUT FOR NOW IT'S BETTER TO SAVE PERFORMANCES AND MEMORY BY JUST RETURNING TWO STRINGS AND NOT EVEN CREATING THE DFs


    #This function is design only to clean by_hour data since that's the data we're going to use for the main purposes of this project
    def clean_traffic_volumes_data(self, by_hour_df: pd.DataFrame) -> [pd.DataFrame | None]:

        #Short dataframe overview
        #print("Short overview on the dataframe: \n", by_hour_df.describe())

        #Checking dataframe columns
        #print("Dataframe columns: \n", by_hour_df.columns, "\n")

        #If all values aren't 0 then execute multiple imputation to fill NaNs:


        # ------------------ Execute multiple imputation with MICE (Multiple Imputation by Chain of Equations) ------------------

        non_mice_columns = by_hour_df[["trp_id"]]  # Setting apart the dates column to execute MICE (multiple imputation) only on numerical columns and then merging that back to the df

        try:
            cleaner = Cleaner()
            by_hour_df = cleaner.impute_missing_values(by_hour_df.drop(non_mice_columns.columns, axis=1))

            for nm_col in non_mice_columns.columns:
                by_hour_df[nm_col] = non_mice_columns[nm_col]

        except ValueError:
            print("\03391mValue error raised. Continuing with the cleaning\0330m")
            return None


        # ------------------ Data types transformation ------------------

        by_hour_df["year"] = by_hour_df["year"].astype("int")
        by_hour_df["month"] = by_hour_df["month"].astype("int")
        by_hour_df["week"] = by_hour_df["week"].astype("int")
        by_hour_df["day"] = by_hour_df["day"].astype("int")
        by_hour_df["hour"] = by_hour_df["hour"].astype("int")
        by_hour_df["volume"] = by_hour_df["volume"].astype("int")


        #print("By hour dataframe overview: \n", by_hour_df.head(15), "\n")

        #print("Data types: ")
        #print(by_hour_df.dtypes, "\n")

        #print("NaN sum: \n", by_hour_df.isna().sum())

        print("\n\n")

        return by_hour_df


    def export_traffic_volumes_data(self, by_hour: pd.DataFrame, volumes_file_path, trp_id: str) -> None:

        file_name = volumes_file_path.split("/")[-1].replace(".json", "C.csv")

        clean_traffic_volumes_folder_path = get_clean_traffic_volumes_folder_path()

        file_path = clean_traffic_volumes_folder_path + file_name #C stands for "cleaned"
        #print(file_path)

        try:
            by_hour.to_csv(file_path, index=False)
            print(f"TRP: {trp_id} data exported correctly\n\n")
        except AttributeError:
            print(f"\033[91mCouldn't export {trp_id} TRP volumes data\033[0m")

        return None


    def cleaning_pipeline(self, volumes_file_path: str) -> None:

        volumes = import_volumes_data(volumes_file_path)
        trp_id = volumes["trafficData"]["trafficRegistrationPoint"]["id"]

        self.data_overview(volumes_data=volumes, verbose=True)
        by_hour_df = self.restructure_traffic_volumes_data(volumes) #TODO IN THE FUTURE SOME ANALYSES COULD BE EXECUTED WITH THE by_lane_df OR by_direction_df, IN THAT CASE WE'LL REPLACE THE _, _ WITH by_lane_df, by_direction_df

        #TODO TO-NOTE TRPs WITH NO DATA WON'T BE INCLUDED IN THE ROAD NETWORK CREATION SINCE THEY WON'T RETURN A DATAFRAME (BECAUSE THEY DON'T HAVE DATA TO STORE IN A DF)

        if by_hour_df is not None:
            by_hour_df = self.clean_traffic_volumes_data(by_hour_df)

            if by_hour_df is not None:
                self.export_traffic_volumes_data(by_hour_df, volumes_file_path, trp_id=trp_id)

        elif by_hour_df is None:
            pass #A warning for empty nodes is given during the restructuring section


        print("--------------------------------------------------------\n\n")

        return None


    def execute_cleaning(self, volumes_file_path: str) -> None:

        self.cleaning_pipeline(volumes_file_path=volumes_file_path)

        return None



class AverageSpeedCleaner(Cleaner):

    def __init__(self):
        super().__init__()


    @staticmethod
    def retrieve_trp_id_from_avg_speed_file(filename: str) -> str:
        trp_id = filename.split("_")[0]
        return trp_id


    def retrieve_trp_data_from_avg_speed_file(self, avg_speed_filename):

        avg_speed_folder_path = get_raw_average_speed_folder_path() #Getting the raw average speed folder path
        trp_id = self.retrieve_trp_id_from_avg_speed_file(avg_speed_filename) #Combining the raw average speed folder path with the filename of the file we want to check out
        trp_data = import_TRPs_data()  #All TRPs data retrieval

        trp_data = [i for i in trp_data["trafficRegistrationPoints"] if i["id"] == trp_id]  #Finding the data for the specific TRP taken in consideration by iterating on all the TRPs available in the trp_file
        trp_data = trp_data[0]

        return trp_data


    def data_overview(self, trp_id, verbose: bool = True) -> None:
        # Since the prints below are all the same (except for one) we could technically create a workaround to avoid having to repeat these lines, but it would complicate a lot something that's way simpler (just prints).
        # Thus, for readability purposes we're going to repeat the same prints (except for one) as in the TrafficVolumeCleaner class

        trp_metadata = get_trp_metadata(trp_id)

        if verbose is True:

            print("******** Traffic Registration Point Information ********")

            print("ID: ", trp_metadata["trp_id"])
            print("Name: ", trp_metadata["name"])
            print("Road category: ", trp_metadata["road_category"])
            print("Coordinates: ")
            print(" - Lat: ", trp_metadata["lat"])
            print(" - Lon: ", trp_metadata["lon"])
            print("County name: ", trp_metadata["county_name"])
            print("County number: ", trp_metadata["county_number"])
            print("Geographic number: ", trp_metadata["geographic_number"])
            print("Country part: ", trp_metadata["country_part"])
            print("Municipality name: ", trp_metadata["municipality_name"])

            print("Traffic registration type: ", trp_metadata["traffic_registration_type"])
            print("Data time span: ")
            print(" - First data: ", trp_metadata["first_data"])
            print(" - First data with quality metrics: ", trp_metadata["first_data_with_quality_metrics"])
            print(" - Latest data: ")
            print("   > Volume by day: ", trp_metadata["latest_volume_by_day"])
            print("   > Volume by hour: ", trp_metadata["latest_volume_byh_hour"])
            print("   > Volume average daily by year: ", trp_metadata["latest_volume_average_daily_by_year"])
            print("   > Volume average daily by season: ", trp_metadata["latest_volume_average_daily_by_season"])
            print("   > Volume average daily by month: ", trp_metadata["latest_volume_average_daily_by_month"])
            print("Number of data nodes: ", trp_metadata["number_of_data_nodes"])

            print("\n")


        elif verbose is False:

            print("******** Traffic Registration Point Information ********")

            print("ID: ", trp_metadata["trp_id"])
            print("Name: ", trp_metadata["name"])

        print("\n\n")

        return None


    def clean_avg_speed_data(self, avg_speed_data) -> [tuple[pd.DataFrame, str, str, str] | None]:


        #TODO ADDRESS FOR TOTALLY EMPTY FILES CASE (MICE)

        trp_id = str(avg_speed_data["trp_id"].unique()[0]) #There'll be only one value since each file only represents the data for one TRP only

        # Determining the days range of the data
        t_min = pd.to_datetime(avg_speed_data["date"]).min()
        t_max = pd.to_datetime(avg_speed_data["date"].max())

        print("Registrations time-range: ")
        print("First day of data registration: ", t_min)
        print("Last day of data registration: ", t_max, "\n\n")


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

        avg_speed_data["year"] = avg_speed_data["date"].dt.year
        avg_speed_data["month"] = avg_speed_data["date"].dt.month
        avg_speed_data["week"] = avg_speed_data["date"].dt.isocalendar().week
        avg_speed_data["day"] = avg_speed_data["date"].dt.day

        avg_speed_data["year"] = avg_speed_data["year"].astype("int")
        avg_speed_data["month"] = avg_speed_data["month"].astype("int")
        avg_speed_data["week"] = avg_speed_data["week"].astype("int")
        avg_speed_data["day"] = avg_speed_data["day"].astype("int")

        avg_speed_data["hour_start"] = avg_speed_data["hour_start"].astype("int")

        avg_speed_data = avg_speed_data.drop(columns=["traffic_volume", "lane"], axis=1)


        # ------------------ Multiple imputation to fill NaN values ------------------

        non_mice_cols = ["trp_id", "date"]
        df_non_mice_cols = avg_speed_data[non_mice_cols] #To merge them later into the NaN filled dataframe

        avg_speed_data = avg_speed_data.drop(columns=non_mice_cols, axis=1) #Columns to not include for Multiple Imputation By Chained Equations (MICE)

        try:
            cleaner = Cleaner()
            avg_speed_data = cleaner.impute_missing_values(avg_speed_data)

            print("Multiple imputation on average speed data executed successfully\n\n")

            #print(avg_speed_data.isna().sum())

            #These transformations here are necessary since after the multiple imputation every column's type becomes float
            avg_speed_data["year"] = avg_speed_data["year"].astype("int")
            avg_speed_data["month"] = avg_speed_data["month"].astype("int")
            avg_speed_data["week"] = avg_speed_data["week"].astype("int")
            avg_speed_data["day"] = avg_speed_data["day"].astype("int")

            avg_speed_data["hour_start"] = avg_speed_data["hour_start"].astype("int")

        except ValueError:
            print("\03391mValue error raised. Continuing with the cleaning\0330m")
            return None

        #Merging non MICE columns back into the MICEed dataframe
        for nm_col in non_mice_cols:
            avg_speed_data[nm_col] = df_non_mice_cols[nm_col]

        print("Dataframe overview: \n", avg_speed_data.head(15), "\n")
        print("Basic descriptive statistics on the dataframe: \n", avg_speed_data.drop(columns=["year", "month", "week", "day", "hour_start"], axis=1).describe(), "\n")


        # ------------------ Restructuring the data ------------------
        #Here we'll restructure the data into a more convenient, efficient and readable format
        #The mean_speed will be defined by hour and not by hour AND lane.
        #This is because the purpose of this project is to create a predictive ml/statistical model for each type of road
        #Also, generalizing the lanes data wouldn't make much sense because the lanes in each street may have completely different data, not because of the traffic behaviour, but because of the location of the TRP itself
        #If we were to create a model for each TRP, then it could make some sense, but that's not the goal of this project

        #agg_data will be a dict of lists which we'll use to create a dataframe afterward
        agg_data = {"trp_id": [],
                    "date": [],
                    "year": [],
                    "month": [],
                    "week": [],
                    "day": [],
                    "hour_start": [],
                    "mean_speed": [],
                    "percentile_85": [],
                    "coverage": []}


        for ud in avg_speed_data["date"].unique():

            day_data = avg_speed_data.query(f"date == '{ud}'")
            #print(day_data)

            for h in day_data["hour_start"].unique():
                #print(day_data[["mean_speed", "hour_start"]].query(f"hour_start == {h}")["mean_speed"])
                #Using the median to have a more robust indicator which won't be influenced by outliers as much as the mean
                agg_data["mean_speed"].append(np.round(np.median(day_data[["mean_speed", "hour_start"]].query(f"hour_start == {h}")["mean_speed"]), decimals=2))
                agg_data["percentile_85"].append(np.round(np.median(day_data[["percentile_85", "hour_start"]].query(f"hour_start == {h}")["percentile_85"]), decimals=2))
                agg_data["coverage"].append(np.round(np.median(day_data[["coverage", "hour_start"]].query(f"hour_start == {h}")["coverage"]), decimals=2))
                agg_data["hour_start"].append(h)
                agg_data["year"].append(int(datetime.strptime(str(ud)[:10], "%Y-%m-%d").strftime("%Y")))
                agg_data["month"].append(int(datetime.strptime(str(ud)[:10], "%Y-%m-%d").strftime("%m")))
                agg_data["week"].append(int(datetime.strptime(str(ud)[:10], "%Y-%m-%d").strftime("%V")))
                agg_data["day"].append(int(datetime.strptime(str(ud)[:10], "%Y-%m-%d").strftime("%d")))
                agg_data["date"].append(ud)
                agg_data["trp_id"].append(trp_id)


        #print(agg_data)

        #The old avg_data dataframe will be overwritten by this new one which will have all the previous data, but with a new structure
        avg_speed_data = pd.DataFrame(agg_data)
        avg_speed_data = avg_speed_data.reindex(sorted(avg_speed_data.columns), axis=1)

        #print(avg_speed_data.head(15))
        #print(avg_speed_data.dtypes)

        return avg_speed_data, trp_id, str(t_max)[:10], str(t_min)[:10]


    def export_clean_avg_speed_data(self, avg_speed_data: pd.DataFrame, trp_id: str, t_max: str, t_min: str) -> None:

        clean_avg_data_folder_path = get_clean_average_speed_folder_path()

        try:
            avg_speed_data.to_csv(clean_avg_data_folder_path + trp_id + f"_S{t_min}_E{t_max}C.csv", index=False) #S stands for Start (registration starting date), E stands for End and C for Clean
            print(f"Average speed data for TRP: {trp_id} saved successfully\n\n")
        except Exception as e:
            print(f"\033[91mCouldn't export TRP: {trp_id} volumes data. Error: {e}\033[0m")


        return None


    def execute_cleaning(self, file_path, file_name) -> None:
        '''
        The avg_speed_file_path parameter is the path to the average speed file the user wants to analyze
        The avg_speed_file_name parameter is just the name of the file, needed for secondary purposes or functionalities

        '''

        try:
            average_speed_data = import_avg_speed_data(file_path=file_path)

            trp_id = file_name.split("_")[0]
            self.data_overview(trp_id, verbose=True)

            average_speed_data, trp_id, t_max, t_min = self.clean_avg_speed_data(average_speed_data)

            self.export_clean_avg_speed_data(average_speed_data, trp_id, t_max, t_min)

        except IndexError:
            print(f"\033091mNo data available for file: {file_name}")


        return None












