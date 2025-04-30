from tfs_utils import *
import numpy as np
import json
import datetime
from datetime import datetime
import os
import pandas as pd
import pprint

from sklearn.linear_model import Lasso, GammaRegressor, QuantileRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklego.meta import ZeroInflatedRegressor

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

dt_format = "%Y-%m-%dT%H"



class BaseCleaner:

    def __init__(self):
        self._cwd = os.getcwd()
        self._ops_folder = "ops"
        self._ops_name = get_active_ops()
        self._regressor_types = ["lasso", "gamma", "quantile"]

    #General definition of the data_overview() function. This will take two different forms: the traffic volumes one and the average speed one.
    #Thus, the generic "data" parameter will become the volumes_data or the avg_speed_data one
    @staticmethod
    def data_overview(trp_data, data: dict, verbose: bool):
        return data


    #Executing multiple imputation to get rid of NaNs using the MICE method (Multiple Imputation by Chained Equations)
    def impute_missing_values(self, data: pd.DataFrame, r: str = "lasso") -> pd.DataFrame: #TODO TO CHANGE INTO dd.DataFrame
        """
        This function should only be supplied with numerical columns-only dataframes

        Parameters:
            data: the data with missing values
            r: the regressor kind. Has to be within a specific list or regressors available
        """
        if r not in self._regressor_types: raise ValueError(f"Regressor type '{r}' is not supported. Must be one of: {self._regressor_types}")

        reg = None
        if r in self._regressor_types:
            if r == "lasso": reg = ZeroInflatedRegressor(regressor=Lasso(random_state=100, fit_intercept=True), classifier=DecisionTreeClassifier(random_state=100)) #Using Lasso regression (L1 Penalization) to get better results in case of non-informative columns present in the data (coverage data, because their values all the same)
            elif r == "gamma": reg = ZeroInflatedRegressor(regressor=GammaRegressor(fit_intercept=True, verbose=0), classifier=DecisionTreeClassifier(random_state=100)) #Using Gamma regression to address for the zeros present in the data (which will need to be predicted as well)
            elif r == "quantile": reg = ZeroInflatedRegressor(regressor=QuantileRegressor(fit_intercept=True), classifier=DecisionTreeClassifier(random_state=100))

        mice_imputer = IterativeImputer(estimator=reg, random_state=100, verbose=0, imputation_order="roman", initial_strategy="mean") #Imputation order is set to arabic so that the imputations start from the right (so from the traffic volume columns)
        data = pd.DataFrame(mice_imputer.fit_transform(data), columns=data.columns) #Fitting the imputer and processing all the data columns except the date one

        return data


class TrafficVolumesCleaner(BaseCleaner):

    def __init__(self):
        super().__init__()


    #This function is only to give the user an overview of the data which we're currently cleaning, and some specific information about the TRP (Traffic Registration Point) which has collected it
    def data_overview(self, volumes_data: dict, verbose: bool = True) -> None:

        trp_id = volumes_data["trafficData"]["trafficRegistrationPoint"]["id"]
        try:
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

        except FileNotFoundError as e: print(f"\033[91mMetadata file not found. Error: {e}\033[0m"); return None


    def restructure_traffic_volumes_data(self, volumes_payload):

        by_hour_structured = {"trp_id": [],
                              "volume": [],
                              "coverage": [],
                              "year": [],
                              "month": [],
                              "week": [],
                              "day": [],
                              "hour": [],
                              "date": []}

        by_lane_structured = {"trp_id": [],
                              "volume": [],
                              "coverage": [],
                              "year": [],
                              "month": [],
                              "week": [],
                              "day": [],
                              "hour": [],
                              "date": [],
                              "lane": []}

        by_direction_structured = {"trp_id": [],
                                   "volume": [],
                                   "coverage": [],
                                   "year": [],
                                   "month": [],
                                   "week": [],
                                   "day": [],
                                   "hour": [],
                                   "date": [],
                                   "direction": []}


        # ------------------ Data payload extraction ------------------

        nodes = volumes_payload["trafficData"]["volume"]["byHour"]["edges"]
        number_of_nodes = len(nodes)
        trp_id = volumes_payload["trafficData"]["trafficRegistrationPoint"]["id"]

        if number_of_nodes == 0:
            print(f"\033[91mNo data found for TRP: {volumes_payload['trafficData']['trafficRegistrationPoint']['id']}\033[0m\n\n")
            return None

        else:

            update_metainfo(trp_id, ["common", "non_empty_volumes_trps"], mode="append")

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

            # ------------------ Finding all the available directions for the TRP ------------------

            direction_sample_node = nodes[0]["node"]["byDirection"]
            directions = [d["heading"] for d in direction_sample_node]
            print("Directions available: ", directions)


            # ------------------ Finding all unique days in which registrations took place ------------------

            reg_datetimes = set([datetime.fromisoformat(n["node"]["from"]).replace(tzinfo=None).isoformat() for n in nodes]) #Only keeping the datetime without the +00:00 at the end #TODO TO TEST THIS, BEFORE IT WAS: n["node"]["from"][:-6]
            #Removing duplicates and keeping the time as well. This will be needed to extract the hour too
            #print(reg_datetimes)

            reg_dates = set([str(datetime.fromisoformat(dt).date().isoformat()) for dt in reg_datetimes])
            #datetime.fromisoformat() converts a string to a datetime object. Then it only keeps the date and converts it into iso format again.
            #The final step is converting the datetime object to a string with str()
            #Removing duplicates with set(). With the line above we'll get the unique days where data registration took place. #TODO TO TEST THIS. BEFORE IT WAS dt[:10]
            print("Number of days where registrations took place: ", len(reg_dates))

            first_registration_date = min(reg_dates)
            last_registration_date = max(reg_dates)

            print("First registration day available: ", first_registration_date)
            print("Last registration day available: ", last_registration_date)

            #The available_day_hours dict will have as key-value pairs: the day and a list with all hours which do have registrations (so that have data)
            available_day_hours = {d: [] for d in reg_dates} #These dict will have a dictionary for each day with an empty list
            for rd in reg_datetimes: available_day_hours[rd[:10]].append(datetime.strptime(rd, "%Y-%m-%dT%H:%M:%S").strftime("%H"))

            # ------------------ Addressing missing days and hours ------------------

            missing_days = [str(d.date().isoformat()) for d in get_theoretical_days(first_registration_date, last_registration_date) if str(d.date().isoformat()) not in reg_dates]
            print("Missing days: ", missing_days)
            print("Number of missing days: ", len(missing_days))

            missing_hours_by_day = {d: [h for h in get_theoretical_hours() if h not in available_day_hours[d]] for d in available_day_hours.keys()} #This dictionary comprehension goes like this: we'll create a day key with a list of hours for each day in the available days.
            #Each day's list will only include registration hours (h) which SHOULD exist, but are missing in the available dates in the data
            for md in missing_days: missing_hours_by_day[md] = get_theoretical_hours() #If a whole day is missing we'll just create it and say that all hours of that day are missing
            missing_hours_by_day = {d: l for d, l in missing_hours_by_day.items() if len(l) != 0} #Removing elements with empty lists (the days which don't have missing hours)
            print("Missing hours by day: ", missing_hours_by_day)

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
                    by_hour_structured["date"].append(d)

            #pprint.pprint(by_hour_structured)

            #TODO ADDRESS FOR MISSING DAYS PROBLEM, CREATE ALL ATTRIBUTES WITH NONE VALUES FOR EACH MISSING DAY AND ADD IT TO BY_HOUR_STRUCTURED, BY_LANE_STRUCTURED AND BY_DIRECTION_STRUCTURED


            # ------------------ Extracting the data from JSON file and converting it into tabular format ------------------

            for node in nodes:
                # ---------------------- Fetching registration datetime ----------------------

                #This is the datetime which will be representative of a volume, specifically, there will be multiple datetimes with the same day
                # to address this fact we'll just re-format the data to keep track of the day, but also maintain the volume values for each hour
                reg_datetime = datetime.strptime(datetime.fromisoformat(node["node"]["from"]).replace(tzinfo=None).isoformat(), "%Y-%m-%dT%H:%M:%S").strftime("%Y-%m-%dT%H")  #Only keeping the datetime without the +00:00 at the end
                year = datetime.strptime(reg_datetime, "%Y-%m-%dT%H").strftime("%Y")
                month = datetime.strptime(reg_datetime, "%Y-%m-%dT%H").strftime("%m")
                week = datetime.strptime(reg_datetime, "%Y-%m-%dT%H").strftime("%V")
                day = datetime.strptime(reg_datetime, "%Y-%m-%dT%H").strftime("%d")
                hour = datetime.strptime(reg_datetime, "%Y-%m-%dT%H").strftime("%H")
                date = datetime.strptime(reg_datetime, "%Y-%m-%dT%H").date().isoformat()

                # ----------------------- Total volumes section -----------------------

                volume = node["node"]["total"]["volumeNumbers"]["volume"] if node["node"]["total"]["volumeNumbers"] is not None else None #In some cases the volumeNumbers key could have null as value, so the "volume" key won't be present. In that case we'll directly insert None as value with an if statement
                coverage = node["node"]["total"]["coverage"]["percentage"] if node["node"]["total"]["coverage"] is not None else None #For less recent data it's possible that sometimes coverage can be null, so we'll address this problem like so

                by_hour_structured["trp_id"].append(trp_id)
                by_hour_structured["year"].append(year)
                by_hour_structured["month"].append(month)
                by_hour_structured["week"].append(week)
                by_hour_structured["day"].append(day)
                by_hour_structured["hour"].append(hour)
                by_hour_structured["volume"].append(volume)
                by_hour_structured["coverage"].append(coverage)
                by_hour_structured["date"].append(date)


                #   ----------------------- By lane section -----------------------

                lanes_data = node["node"]["byLane"] #Extracting byLane data

                #Every lane's data is kept isolated from the other lanes' data, so a for cycle is needed to extract all the data from each lane's section
                for lane in lanes_data:

                    road_link_lane_number = lane["lane"]["laneNumberAccordingToRoadLink"]
                    lane_volume = lane["total"]["volumeNumbers"]["volume"] if lane["total"]["volumeNumbers"] is not None else None #In some cases the volumeNumbers key could have null as value, so the "volume" key won't be present. In that case we'll directly insert None as value with an if statement
                    lane_coverage = lane["total"]["coverage"]["percentage"] if lane["total"]["coverage"] is not None else None

                    # ------- Extracting data from the dictionary and appending it into by_lane_structured -------

                    by_lane_structured["trp_id"].append(trp_id)
                    by_lane_structured["year"].append(year)
                    by_lane_structured["month"].append(month)
                    by_lane_structured["week"].append(week)
                    by_lane_structured["day"].append(day)
                    by_lane_structured["hour"].append(hour)
                    by_lane_structured["volume"].append(lane_volume)
                    by_lane_structured["coverage"].append(lane_coverage)
                    by_lane_structured["lane"].append(road_link_lane_number)
                    by_lane_structured["date"].append(date)


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
                    by_direction_structured["date"].append(date)

                    #TODO THE SAME PRINCIPLE AS BEFORE APPLIES HERE, SAVE ALL THE AVAILABLE DIRECTIONS IN THE TRP'S METADATA FILE

            # ------------------ Ensuring that XXXXXXXXXXXXXXX ------------------

            #for k in by_hour_structured.keys():
                #print(f"List length for key: {k} = ", len(by_hour_structured[k]))

            # ------------------ Dataframes creation and printing ------------------


            #print("\n\n----------------- By Hour Structured -----------------")
            #pprint.pp(by_hour_structured)
            #print(by_hour_structured)

            by_hour_df = pd.DataFrame(by_hour_structured) #TODO CHANGE INTO dd.DataFrame
            by_hour_df = by_hour_df.sort_values(by=["date", "hour"], ascending=True) #TODO ADD THEN .persist()
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
    def clean_traffic_volumes_data(self, by_hour_df: pd.DataFrame) -> pd.DataFrame | None:

        #Short dataframe overview
        #print("Short overview on the dataframe: \n", by_hour_df.describe())

        #Checking dataframe columns
        #print("Dataframe columns: \n", by_hour_df.columns, "\n")

        #If all values aren't 0 then execute multiple imputation to fill NaNs:

        # ------------------ Execute multiple imputation with MICE (Multiple Imputation by Chain of Equations) ------------------

        non_mice_columns = by_hour_df[["trp_id", "date", "year", "month", "day", "week"]]
        #Setting apart the dates column to execute MICE (multiple imputation) only on numerical columns and then merging that back to the df
        #Still, we're leaving the hour variable to address for the variability of the traffic volumes during the day

        print("Shape before MICE: ", by_hour_df.shape)
        print("Number of zeros before MICE: ", len(by_hour_df[by_hour_df["volume"] == 0]))

        try:
            cleaner = BaseCleaner()
            by_hour_df = cleaner.impute_missing_values(by_hour_df.drop(non_mice_columns.columns, axis=1), r="gamma")  #Don't use gamma regression since, apparently it can't handle zeros

            for nm_col in non_mice_columns.columns:
                by_hour_df[nm_col] = non_mice_columns[nm_col]

            print("Shape after MICE: ", by_hour_df.shape)
            print("Number of zeros after MICE: ", len(by_hour_df[by_hour_df["volume"] == 0]))
            print("Number of negative values (after MICE): ", len(by_hour_df[by_hour_df["volume"] < 0]))

        except ValueError as e:
            print(f"\033[91mValue error raised. Error: {e} Continuing with the cleaning.\033[0m")
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


    #TODO CHANGE by_hour TO dd.DatFrame
    def export_traffic_volumes_data(self, by_hour: pd.DataFrame, volumes_file_path, trp_id: str) -> None:

        file_name = volumes_file_path.split("/")[-1].replace(".json", "C.csv") #TODO IMPROVE THIS THROUGH A SIMPLER FILE NAME AND A PARSER OR GETTER FUNCTION IN tfs_utils.py

        clean_traffic_volumes_folder_path = get_clean_traffic_volumes_folder_path()
        file_path = clean_traffic_volumes_folder_path + file_name #C stands for "cleaned"

        try:
            by_hour.to_csv(file_path, index=False)
            update_metainfo(file_name, ["traffic_volumes", "clean_filenames"], mode="append")
            update_metainfo(file_path, ["traffic_volumes", "clean_filepaths"], mode="append")
            print(f"TRP: {trp_id} data exported correctly\n\n")
            return None
        except AttributeError:
            print(f"\033[91mCouldn't export {trp_id} TRP volumes data\033[0m")
            return None


    def cleaning_pipeline(self, volumes_file_path: str) -> None:
        print(volumes_file_path)
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



class AverageSpeedCleaner(BaseCleaner):

    def __init__(self):
        super().__init__()
        self.min_date: datetime.datetime | None = None
        self.max_date: datetime.datetime | None = None

        self.min_date = datetime.strptime(read_metainfo_key(keys_map=["average_speeds", "start_date_iso"]), dt_format)
        self.max_date = datetime.strptime(read_metainfo_key(keys_map=["average_speeds", "end_date_iso"]), dt_format)


    def clean_avg_speed_data(self, avg_speed_data: pd.DataFrame) -> tuple[pd.DataFrame, str, str, str] | None: #TODO TO CHANGE IN dd.DataFrame

        trp_id = str(avg_speed_data["trp_id"].unique()[0]) #There'll be only one value since each file only represents the data for one TRP only

        if len(avg_speed_data) > 0: update_metainfo(trp_id, ["common", "non_empty_avg_speed_trps"], mode="append") #And continue with the code execution...
        else: return None #In case a file is completely empty we'll just return None and not insert its TRP into the "non_empty_avg_speed_trps" key of the metainfo file

        # Determining the days range of the data
        t_min = pd.to_datetime(avg_speed_data["date"]).min()
        t_max = pd.to_datetime(avg_speed_data["date"]).max()

        print("Registrations time-range: ")
        print("First day of data registration: ", t_min)
        print("Last day of data registration: ", t_max, "\n\n")

        # TODO THE SAME WITH MAX DATE
        if self.min_date is not None and self.min_date > t_min: pass
        elif self.min_date is None: update_metainfo(t_min, keys_map=["average_speeds", "start_date_iso"], mode="equals")
        elif self.min_date is not None and self.min_date < t_min: update_metainfo(t_min, keys_map=["average_speeds", "start_date_iso"], mode="equals") #TODO RIGHT NOW WE'RE TAKING THE LATEST MNINUM DATE AVAILABLE
        else: update_metainfo(t_min, keys_map=["average_speeds", "start_date_iso"], mode="equals")


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

        non_mice_df = avg_speed_data[["trp_id", "date", "year", "month", "day", "week"]]
        avg_speed_data = avg_speed_data.drop(columns=non_mice_df.columns, axis=1) #Columns to not include for Multiple Imputation By Chained Equations (MICE)

        print("Shape before MICE: ", avg_speed_data.shape)
        print("Number of zeros before MICE: ", len(avg_speed_data[avg_speed_data["volume"] == 0]))

        try:
            cleaner = BaseCleaner()
            avg_speed_data = cleaner.impute_missing_values(avg_speed_data, r="gamma")
            #print("Multiple imputation on average speed data executed successfully\n\n")

            #print(avg_speed_data.isna().sum())

            print("Shape after MICE: ", avg_speed_data.shape, "\n\n")
            print("Number of zeros after MICE: ", len(avg_speed_data[avg_speed_data["volume"] == 0]))
            print("Number of negative values (after MICE): ", len(avg_speed_data[avg_speed_data["volume"] < 0]))

        except ValueError as e:
            print(f"\033[91mValue error raised. Error: {e} Continuing with the cleaning\033[0m")
            return None

        #Merging non MICE columns back into the MICEed dataframe
        for nm_col in non_mice_df.columns: avg_speed_data[nm_col] = non_mice_df[nm_col]

        #These transformations here are necessary since after the multiple imputation every column's type becomes float
        avg_speed_data["year"] = avg_speed_data["year"].astype("int")
        avg_speed_data["month"] = avg_speed_data["month"].astype("int")
        avg_speed_data["week"] = avg_speed_data["week"].astype("int")
        avg_speed_data["day"] = avg_speed_data["day"].astype("int")
        avg_speed_data["hour_start"] = avg_speed_data["hour_start"].astype("int")

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
        filepath = clean_avg_data_folder_path + trp_id + f"_S{t_min}_E{t_max}C.csv"
        filename = trp_id + f"_S{t_min}_E{t_max}C.csv"

        try:
            avg_speed_data.to_csv(filepath, index=False) #S stands for Start (registration starting date), E stands for End and C for Clean
            update_metainfo(filename, ["average_speeds", "clean_filenames"], mode="append")
            update_metainfo(filepath, ["average_speeds", "clean_filepaths"], mode="append")
            print(f"Average speed data for TRP: {trp_id} saved successfully\n\n")
            return None
        except Exception as e:
            print(f"\033[91mCouldn't export TRP: {trp_id} volumes data. Error: {e}\033[0m")
            return None


    def execute_cleaning(self, file_path, file_name) -> None:
        """
        The avg_speed_file_path parameter is the path to the average speed file the user wants to analyze
        The avg_speed_file_name parameter is just the name of the file, needed for secondary purposes or functionalities
        """

        try:
            average_speed_data = import_avg_speed_data(file_path=file_path)
            #NOTE REMOVED trp_id = get_trp_id_from_filename(file_name) HERE
            #TODO IN THE FUTURE ADD THE UNIFIED data_overview() FUNCTION WHICH WILL PRINT A COMMON DATA OVERVIEW FOR BOTH TRAFFIC VOLUMES DATA AND AVERAGE SPEED ONE

            #Addressing for the empty files problem
            if len(average_speed_data) > 0:
                average_speed_data, trp_id, t_max, t_min = self.clean_avg_speed_data(average_speed_data)
                self.export_clean_avg_speed_data(average_speed_data, trp_id, t_max, t_min)
                return None
            else:
                return None
        except IndexError as e:
            print(f"\033[91mNo data available for file: {file_name}. Error: {e}\033[0m\n")
            return None












