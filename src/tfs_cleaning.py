import numpy as np
import json
import datetime
from datetime import datetime
import os
import pandas as pd
import pprint
import traceback
import logging
import dask.dataframe as dd

from sklearn.linear_model import Lasso, GammaRegressor, QuantileRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklego.meta import ZeroInflatedRegressor

from tfs_utils import *


pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

dt_format = "%Y-%m-%dT%H"


class BaseCleaner:
    def __init__(self):
        self._cwd = os.getcwd()
        self._ops_folder = "ops"
        self._ops_name = get_active_ops()
        self._regressor_types = ["lasso", "gamma", "quantile"]


    # Executing multiple imputation to get rid of NaNs using the MICE method (Multiple Imputation by Chained Equations)
    def _impute_missing_values(self, data: pd.DataFrame | dd.DataFrame, r: str = "gamma") -> pd.DataFrame:
        """
        This function should only be supplied with numerical columns-only dataframes

        Parameters:
            data: the data with missing values
            r: the regressor kind. Has to be within a specific list or regressors available
        """
        if r not in self._regressor_types:
            raise ValueError(f"Regressor type '{r}' is not supported. Must be one of: {self._regressor_types}")

        reg = None
        if r == "lasso":
            reg = ZeroInflatedRegressor(
                regressor=Lasso(random_state=100, fit_intercept=True),
                classifier=DecisionTreeClassifier(random_state=100)
            )  # Using Lasso regression (L1 Penalization) to get better results in case of non-informative columns present in the data (coverage data, because their values all the same)
        elif r == "gamma":
            reg = ZeroInflatedRegressor(
                regressor=GammaRegressor(fit_intercept=True, verbose=0),
                classifier=DecisionTreeClassifier(random_state=100)
            )  # Using Gamma regression to address for the zeros present in the data (which will need to be predicted as well)
        elif r == "quantile":
            reg = ZeroInflatedRegressor(
                regressor=QuantileRegressor(fit_intercept=True),
                classifier=DecisionTreeClassifier(random_state=100)
            )

        mice_imputer = IterativeImputer(
            estimator=reg,
            random_state=100,
            verbose=0,
            imputation_order="roman",
            initial_strategy="mean"
        )  # Imputation order is set to arabic so that the imputations start from the right (so from the traffic volume columns)

        return pd.DataFrame(mice_imputer.fit_transform(data), columns=data.columns) # Fitting the imputer and processing all the data columns except the date one #TODO BOTTLENECK


    @staticmethod
    def _is_empty(data: dict[Any, Any]) -> bool:
        return True if len(data) == 0 else False



class TrafficVolumesCleaner(BaseCleaner):
    def __init__(self):
        super().__init__()


    @staticmethod
    def _get_lanes_number(data: dict[Any, Any]) -> int:
        return max((ln["lane"]["laneNumberAccordingToRoadLink"] for ln in data[0]["node"]["byLane"]))  # Determining the total number of lanes for the TRP taken into consideration. Using a generator comprehension to improve performances


    @staticmethod
    def _get_directions(data: dict[Any, Any]) -> Generator[int]:
        return (d["heading"] for d in data[0]["node"]["byDirection"])


    @staticmethod
    def _get_registration_datetimes(edges: dict[Any, Any]) -> set[str]:
        return set(datetime.fromisoformat(node["node"]["from"]).replace(tzinfo=None).isoformat() for node in edges) # Only keeping the datetime without the +00:00 at the end
               # Removing duplicates with set() at the end


    def _get_missing_data(self, edges: dict[Any, Any]) -> dict[str, list[str]]:

        reg_datetimes = self._get_registration_datetimes(edges)

        # The available_day_hours dict will have as key-value pairs: the day and a list with all hours which do have registrations (so that have data)
        available_day_hours = {str(datetime.fromisoformat(dt).date().isoformat()): [datetime.strptime(rd, "%Y-%m-%dT%H:%M:%S").strftime("%H") for rd in reg_datetimes]
                               for dt in (str(datetime.fromisoformat(dt).date().isoformat()) for dt in reg_datetimes)}  # These dict will have a dictionary for each day with an empty list

        # ------------------ Addressing missing days and hours problem ------------------

        missing_hours_by_day = {
            d: [h for h in (f"{i:02}" for i in range(24)) if h not in available_day_hours[d]]
            for d in available_day_hours.keys()
        }  # This dictionary comprehension goes like this: we'll create a day key with a list of hours for each day in the available days.
        # Each day's list will only include registration hours (h) which SHOULD exist, but are missing in the available dates in the data

        #Determining the missing days in the first generator: for missing_day in (1° generator)
        #Then simply creating the key-value pair in the missing_hours_by_day with all 24 hours as values of the list which includes missing hours for that day
        for missing_day in (
            str(d.date().isoformat())
            for d in pd.date_range(start=min(reg_datetimes), end=max(reg_datetimes), freq="d")
            if str(d.date().isoformat()) not in available_day_hours.keys()
        ):
            missing_hours_by_day[missing_day] = [f"{i:02}" for i in range(24)]  # If a whole day is missing we'll just create it and say that all hours of that day are missing

        return {d: l for d, l in missing_hours_by_day.items() if len(l) != 0}  # Removing elements with empty lists (the days which don't have missing hours)


    def _parse_by_hour(self, payload: dict[Any, Any]) -> pd.DataFrame | dd.DataFrame | None:

        trp_id = payload["trafficData"]["trafficRegistrationPoint"]["id"]
        payload = payload["trafficData"]["volume"]["byHour"]["edges"]

        if self._is_empty(payload):
            print(f"\033[91mNo data found for TRP: {trp_id}\033[0m\n\n")
            return None

        update_metainfo(trp_id, ["common", "non_empty_volumes_trps"], mode="append")

        n_lanes = self._get_lanes_number(payload)
        print("Number of lanes: ", n_lanes)
        directions = list(self._get_directions(payload))
        print("Directions available: ", directions)

        #Just for demonstration purposes
        registration_dates = self._get_registration_datetimes(payload)
        print("Number of days where registrations took place: ", len(registration_dates))
        print("First registration day available: ", min(registration_dates))
        print("Last registration day available: ", max(registration_dates))


        by_hour_structured = {
            "trp_id": [],
            "volume": [],
            "coverage": [],
            "year": [],
            "month": [],
            "week": [],
            "day": [],
            "hour": [],
            "date": [],
        }

        missing_days_cnt = 0
        for d, mh in self._get_missing_data(payload).items():
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

            missing_days_cnt += 1

        print("Number of missing days: ", missing_days_cnt)

        # ------------------ Extracting the data from JSON file and converting it into tabular format ------------------

        for edge in payload:
            # ---------------------- Fetching registration datetime ----------------------

            # This is the datetime which will be representative of a volume, specifically, there will be multiple datetimes with the same day
            # to address this fact we'll just re-format the data to keep track of the day, but also maintain the volume values for each hour
            reg_datetime = datetime.strptime(datetime.fromisoformat(edge["node"]["from"]).replace(tzinfo=None).isoformat(), "%Y-%m-%dT%H:%M:%S").strftime("%Y-%m-%dT%H")  # Only keeping the datetime without the +00:00 at the end

            # ----------------------- Total volumes section -----------------------

            volume = edge["node"]["total"]["volumeNumbers"]["volume"] if edge["node"]["total"]["volumeNumbers"] is not None else None  # In some cases the volumeNumbers key could have null as value, so the "volume" key won't be present. In that case we'll directly insert None as value with an if statement
            coverage = edge["node"]["total"]["coverage"]["percentage"] if edge["node"]["total"]["coverage"] is not None else None  # For less recent data it's possible that sometimes coverage can be null, so we'll address this problem like so

            by_hour_structured["trp_id"].append(trp_id)
            by_hour_structured["year"].append(datetime.strptime(reg_datetime, "%Y-%m-%dT%H").strftime("%Y"))
            by_hour_structured["month"].append(datetime.strptime(reg_datetime, "%Y-%m-%dT%H").strftime("%m"))
            by_hour_structured["week"].append(datetime.strptime(reg_datetime, "%Y-%m-%dT%H").strftime("%V"))
            by_hour_structured["day"].append(datetime.strptime(reg_datetime, "%Y-%m-%dT%H").strftime("%d"))
            by_hour_structured["hour"].append(datetime.strptime(reg_datetime, "%Y-%m-%dT%H").strftime("%H"))
            by_hour_structured["volume"].append(volume)
            by_hour_structured["coverage"].append(coverage)
            by_hour_structured["date"].append(datetime.strptime(reg_datetime, "%Y-%m-%dT%H").date().isoformat())


        return pd.DataFrame(by_hour_structured).sort_values(by=["date", "hour"], ascending=True)


    def _parse_by_lane(self, payload: dict[Any, Any]) -> pd.DataFrame | dd.DataFrame | None:

        trp_id = payload["trafficData"]["trafficRegistrationPoint"]["id"]
        payload = payload["trafficData"]["volume"]["byHour"]["edges"]

        if self._is_empty(payload):
            print(f"\033[91mNo data found for TRP: {trp_id}\033[0m\n\n")
            return None

        update_metainfo(trp_id, ["common", "non_empty_volumes_trps"], mode="append")

        n_lanes = self._get_lanes_number(payload)
        print("Number of lanes: ", n_lanes)
        directions = list(self._get_directions(payload))
        print("Directions available: ", directions)

        # Just for demonstration purposes
        registration_dates = self._get_registration_datetimes(payload)
        print("Number of days where registrations took place: ", len(registration_dates))
        print("First registration day available: ", min(registration_dates))
        print("Last registration day available: ", max(registration_dates))

        by_lane_structured = {
            "trp_id": [],
            "volume": [],
            "coverage": [],
            "year": [],
            "month": [],
            "week": [],
            "day": [],
            "hour": [],
            "date": [],
            "lane": [],
        }

        missing_days_cnt = 0
        for d, mh in self._get_missing_data(payload).items():
            for h in mh:
                by_lane_structured["trp_id"].append(trp_id)
                by_lane_structured["volume"].append(None)
                by_lane_structured["coverage"].append(None)
                by_lane_structured["year"].append(datetime.strptime(d, "%Y-%m-%d").strftime("%Y"))
                by_lane_structured["month"].append(datetime.strptime(d, "%Y-%m-%d").strftime("%m"))
                by_lane_structured["week"].append(datetime.strptime(d, "%Y-%m-%d").strftime("%V"))
                by_lane_structured["day"].append(datetime.strptime(d, "%Y-%m-%d").strftime("%d"))
                # TODO ADD None FOR THE LANE MISSING DATA (FOR EACH LANE)
                by_lane_structured["hour"].append(h)
                by_lane_structured["date"].append(d)

            missing_days_cnt += 1

        print("Number of missing days: ", missing_days_cnt)

        for edge in payload:

            # This is the datetime which will be representative of a volume, specifically, there will be multiple datetimes with the same day
            # to address this fact we'll just re-format the data to keep track of the day, but also maintain the volume values for each hour
            reg_datetime = datetime.strptime(datetime.fromisoformat(edge["node"]["from"]).replace(tzinfo=None).isoformat(), "%Y-%m-%dT%H:%M:%S").strftime("%Y-%m-%dT%H")  # Only keeping the datetime without the +00:00 at the end

            # Every lane's data is kept isolated from the other lanes' data, so a for cycle is needed to extract all the data from each lane's section
            for lane in edge["node"]["byLane"]:

                # ------- Extracting data from the dictionary and appending it into by_lane_structured -------

                by_lane_structured["trp_id"].append(trp_id)
                by_lane_structured["year"].append(datetime.strptime(reg_datetime, "%Y-%m-%dT%H").strftime("%Y"))
                by_lane_structured["month"].append(datetime.strptime(reg_datetime, "%Y-%m-%dT%H").strftime("%m"))
                by_lane_structured["week"].append(datetime.strptime(reg_datetime, "%Y-%m-%dT%H").strftime("%V"))
                by_lane_structured["day"].append(datetime.strptime(reg_datetime, "%Y-%m-%dT%H").strftime("%d"))
                by_lane_structured["hour"].append(datetime.strptime(reg_datetime, "%Y-%m-%dT%H").strftime("%H"))
                by_lane_structured["volume"].append(lane["total"]["volumeNumbers"]["volume"] if lane["total"]["volumeNumbers"] is not None else None) # In some cases the volumeNumbers key could have null as value, so the "volume" key won't be present. In that case we'll directly insert None as value with an if statement
                by_lane_structured["coverage"].append(lane["total"]["coverage"]["percentage"] if lane["total"]["coverage"] is not None else None)
                by_lane_structured["lane"].append(lane["lane"]["laneNumberAccordingToRoadLink"])
                by_lane_structured["date"].append(datetime.strptime(reg_datetime, "%Y-%m-%dT%H").date().isoformat())

        return pd.DataFrame(by_lane_structured).sort_values(by=["date", "hour"], ascending=True)


    def _parse_by_direction(self, payload: dict[Any, Any]) -> pd.DataFrame | dd.DataFrame | None:

        trp_id = payload["trafficData"]["trafficRegistrationPoint"]["id"]
        payload = payload["trafficData"]["volume"]["byHour"]["edges"]

        if self._is_empty(payload):
            print(f"\033[91mNo data found for TRP: {trp_id}\033[0m\n\n")
            return None

        update_metainfo(trp_id, ["common", "non_empty_volumes_trps"], mode="append")

        n_lanes = self._get_lanes_number(payload)
        print("Number of lanes: ", n_lanes)
        directions = list(self._get_directions(payload))
        print("Directions available: ", directions)

        # Just for demonstration purposes
        registration_dates = self._get_registration_datetimes(payload)
        print("Number of days where registrations took place: ", len(registration_dates))
        print("First registration day available: ", min(registration_dates))
        print("Last registration day available: ", max(registration_dates))

        by_direction_structured = {
            "trp_id": [],
            "volume": [],
            "coverage": [],
            "year": [],
            "month": [],
            "week": [],
            "day": [],
            "hour": [],
            "date": [],
            "direction": [],
        }

        missing_days_cnt = 0
        for d, mh in self._get_missing_data(payload).items():
            for h in mh:
                by_direction_structured["trp_id"].append(trp_id)
                by_direction_structured["volume"].append(None)
                by_direction_structured["coverage"].append(None)
                by_direction_structured["year"].append(datetime.strptime(d, "%Y-%m-%d").strftime("%Y"))
                by_direction_structured["month"].append(datetime.strptime(d, "%Y-%m-%d").strftime("%m"))
                by_direction_structured["week"].append(datetime.strptime(d, "%Y-%m-%d").strftime("%V"))
                by_direction_structured["day"].append(datetime.strptime(d, "%Y-%m-%d").strftime("%d"))
                # TODO ADD None FOR THE DIRECTION'S MISSING DATA (FOR EACH LANE)
                by_direction_structured["hour"].append(h)
                by_direction_structured["date"].append(d)

            missing_days_cnt += 1

        print("Number of missing days: ", missing_days_cnt)

        for edge in payload:

            # This is the datetime which will be representative of a volume, specifically, there will be multiple datetimes with the same day
            # to address this fact we'll just re-format the data to keep track of the day, but also maintain the volume values for each hour
            reg_datetime = datetime.strptime(datetime.fromisoformat(edge["node"]["from"]).replace(tzinfo=None).isoformat(), "%Y-%m-%dT%H:%M:%S").strftime("%Y-%m-%dT%H")  # Only keeping the datetime without the +00:00 at the end

            # Every direction's data is kept isolated from the other directions' data, so a for cycle is needed
            for direction_section in edge["node"]["byDirection"]:

                by_direction_structured["trp_id"].append(trp_id)
                by_direction_structured["year"].append(datetime.strptime(reg_datetime, "%Y-%m-%dT%H").strftime("%Y"))
                by_direction_structured["month"].append(datetime.strptime(reg_datetime, "%Y-%m-%dT%H").strftime("%m"))
                by_direction_structured["week"].append(datetime.strptime(reg_datetime, "%Y-%m-%dT%H").strftime("%V"))
                by_direction_structured["day"].append(datetime.strptime(reg_datetime, "%Y-%m-%dT%H").strftime("%d"))
                by_direction_structured["hour"].append(datetime.strptime(reg_datetime, "%Y-%m-%dT%H").strftime("%H"))
                by_direction_structured["volume"].append(direction_section["total"]["volumeNumbers"]["volume"] if direction_section["total"]["volumeNumbers"] is not None else None) # In some cases the volumeNumbers key could have null as value, so the "volume" key won't be present. In that case we'll directly insert None as value with an if statement
                by_direction_structured["coverage"].append(direction_section["total"]["coverage"]["percentage"] if direction_section["total"]["coverage"] is not None else None)
                by_direction_structured["direction"].append(direction_section["heading"])
                by_direction_structured["date"].append(datetime.strptime(reg_datetime, "%Y-%m-%dT%H").date().isoformat())

        return pd.DataFrame(by_direction_structured).sort_values(by=["date", "hour"], ascending=True)

    # TODO IN THE FUTURE SOME ANALYSES COULD BE EXECUTED WITH THE by_lane_df OR by_direction_df, IN THAT CASE WE'LL REPLACE THE _, _ WITH by_lane_df, by_direction_df


    def clean(self, volumes_file_path: str, export: bool = True) -> None:
        with open(volumes_file_path, "r", encoding="utf-8") as f:
            by_hour_df = self._parse_by_hour(json.load(f))

        trp_id = by_hour_df["trp_id"].unique()

        #Checking if the dataframe obtained as result of parsing isn't empty
        if by_hour_df.shape[0] > 0:

            # ------------------ Execute multiple imputation with MICE (Multiple Imputation by Chain of Equations) ------------------

            try:
                print("Shape before MICE: ", len(by_hour_df), len(by_hour_df.columns))
                print("Number of zeros before MICE: ", len(by_hour_df[by_hour_df["volume"] == 0]))

                by_hour_df = pd.concat([by_hour_df[["trp_id", "date", "year", "month", "day", "week"]], BaseCleaner()._impute_missing_values(by_hour_df.drop(columns=["trp_id", "date", "year", "month", "day", "week"], axis=1), r="gamma")], axis=1)  # Don't use gamma regression since, apparently it can't handle zeros

                print("Shape after MICE: ", len(by_hour_df), len(by_hour_df.columns))
                print("Number of zeros after MICE: ", len(by_hour_df[by_hour_df["volume"] == 0]))
                print("Number of negative values (after MICE): ", len(by_hour_df[by_hour_df["volume"] < 0]))

            except ValueError as e:
                print(f"\033[91mValue error raised. Error: {e} Continuing with the cleaning.\033[0m")
                return None

            # ------------------ Data types transformation ------------------

            for col in ("year", "month", "week", "day", "hour", "volume"):
                by_hour_df[col] = by_hour_df[col].astype("int")

            # ------------------ Export section ------------------

        if export is True:
            try:
                by_hour_df.to_csv(read_metainfo_key(keys_map=["folder_paths", "data", "traffic_volumes", "subfolders", "clean", "path"]) + volumes_file_path.replace(".json", "C.csv"), index=False, encoding="utf-8")  # C stands for "cleaned"
                print(f"TRP: {trp_id} data exported correctly\n")

                update_trp_metadata(trp_id=trp_id, value=volumes_file_path.replace(".json", "C.csv"), metadata_keys_map=["files", "volumes", "clean"], mode="equals")
                update_trp_metadata(trp_id=trp_id, value=by_hour_df["date"].min(), metadata_keys_map=["data_info", "volumes", "start_date"], mode="equals")
                update_trp_metadata(trp_id=trp_id, value=by_hour_df["date"].max(), metadata_keys_map=["data_info", "volumes", "end_date"], mode="equals")

                print(f"TRP: {trp_id} metadata updated correctly\n\n")

                return None
            except AttributeError:
                print(f"\033[91mCouldn't export {trp_id} TRP volumes data\033[0m")
                return None

        print("--------------------------------------------------------\n\n")

        return None



class AverageSpeedCleaner(BaseCleaner):
    def __init__(self):
        super().__init__()


    def clean(self, speeds: pd.DataFrame) -> tuple[pd.DataFrame | dd.DataFrame, str, str, str] | None:

        if len(speeds) == 0:
            return None  # In case a file is completely empty we'll just return None and not insert its TRP into the "non_empty_avg_speed_trps" key of the metainfo file

        trp_id = str(speeds["trp_id"].unique()[0])  # There'll be only one value since each file only represents the data for one TRP only
        update_metainfo(trp_id, ["common", "non_empty_avg_speed_trps"], mode="append")


        # ------------------ Mean speed, percentile_85 and coverage cleaning ------------------

        speeds["coverage"] = speeds["coverage"].replace(",", ".", regex=True).astype("float") * 100 # Replacing commas with dots and then converting the coverage column to float data type. Finally, transforming the coverage values from 0.0 to 1.0 to 0 to 100 (percent)

        speeds["mean_speed"] = speeds["mean_speed"].replace(",", ".", regex=True).astype("float")  # The regex=True parameter is necessary, otherwise the function, for some reason, won't be able to perform the replacement. Then converting the mean_speed column to float data type
        speeds["percentile_85"] = speeds["percentile_85"].replace(",", ".", regex=True).astype("float")  # The regex=True parameter is necessary, otherwise the function, for some reason, won't be able to perform the replacement. Then converting the percentile_85 column to float data type


        # ------------------ Mean speed and coverage cleaning ------------------

        speeds["hour_start"] = speeds["hour_start"].str[:2] # Keeping only the first two characters (which represent only the hour data)


        # ------------------ Initial data types transformation ------------------

        speeds["trp_id"] = speeds["trp_id"].astype("str")
        speeds["date"] = pd.to_datetime(speeds["date"])

        # Determining the days range of the data
        t_min = pd.to_datetime(speeds["date"]).min()
        t_max = pd.to_datetime(speeds["date"]).max()

        print("Registrations time-range: ")
        print("First day of data registration: ", t_min)
        print("Last day of data registration: ", t_max, "\n\n")


        speeds["year"] = speeds["date"].dt.year
        speeds["month"] = speeds["date"].dt.month
        speeds["week"] = speeds["date"].dt.isocalendar().week
        speeds["day"] = speeds["date"].dt.day

        speeds["year"] = speeds["year"].astype("int")
        speeds["month"] = speeds["month"].astype("int")
        speeds["week"] = speeds["week"].astype("int")
        speeds["day"] = speeds["day"].astype("int")

        speeds["hour_start"] = speeds["hour_start"].astype("int")

        speeds = speeds.drop(columns=["traffic_volume", "lane"], axis=1)

        # ------------------ Multiple imputation to fill NaN values ------------------

        non_mice_df = speeds[["trp_id", "date", "year", "month", "day", "week"]]
        speeds = speeds.drop(columns=non_mice_df.columns, axis=1)  # Columns to not include for Multiple Imputation By Chained Equations (MICE)

        print("Shape before MICE: ", speeds.shape)
        print("Number of zeros before MICE: ", len(speeds[speeds["volume"] == 0]))

        try:
            cleaner = BaseCleaner()
            speeds = cleaner._impute_missing_values(speeds, r="gamma")
            # print("Multiple imputation on average speed data executed successfully\n\n")

            # print(avg_speed_data.isna().sum())

            print("Shape after MICE: ", speeds.shape, "\n\n")
            print("Number of zeros after MICE: ", len(speeds[speeds["volume"] == 0]))
            print("Number of negative values (after MICE): ", len(speeds[speeds["volume"] < 0]))

        except ValueError as e:
            print(f"\033[91mValue error raised. Error: {e} Continuing with the cleaning\033[0m")
            return None

        # Merging non MICE columns back into the MICEed dataframe
        for nm_col in non_mice_df.columns:
            speeds[nm_col] = non_mice_df[nm_col]

        # These transformations here are necessary since after the multiple imputation every column's type becomes float
        speeds["year"] = speeds["year"].astype("int")
        speeds["month"] = speeds["month"].astype("int")
        speeds["week"] = speeds["week"].astype("int")
        speeds["day"] = speeds["day"].astype("int")
        speeds["hour_start"] = speeds["hour_start"].astype("int")

        print("Dataframe overview: \n", speeds.head(15), "\n")
        print("Basic descriptive statistics on the dataframe: \n", speeds.drop(columns=["year", "month", "week", "day", "hour_start"], axis=1).describe(), "\n")

        # ------------------ Restructuring the data ------------------
        # Here we'll restructure the data into a more convenient, efficient and readable format
        # The mean_speed will be defined by hour and not by hour AND lane.
        # This is because the purpose of this project is to create a predictive ml/statistical model for each type of road
        # Also, generalizing the lanes data wouldn't make much sense because the lanes in each street may have completely different data, not because of the traffic behaviour, but because of the location of the TRP itself
        # If we were to create a model for each TRP, then it could make some sense, but that's not the goal of this project

        # agg_data will be a dict of lists which we'll use to create a dataframe afterward
        agg_data = {
            "trp_id": [],
            "date": [],
            "year": [],
            "month": [],
            "week": [],
            "day": [],
            "hour_start": [],
            "mean_speed": [],
            "percentile_85": [],
            "coverage": [],
        }

        for ud in speeds["date"].unique():
            day_data = speeds.query(f"date == '{ud}'")
            # print(day_data)

            for h in day_data["hour_start"].unique():
                # print(day_data[["mean_speed", "hour_start"]].query(f"hour_start == {h}")["mean_speed"])
                # Using the median to have a more robust indicator which won't be influenced by outliers as much as the mean
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


        # The old avg_data dataframe will be overwritten by this new one which will have all the previous data, but with a new structure
        speeds = pd.DataFrame(agg_data)
        speeds = speeds.reindex(sorted(speeds.columns), axis=1)

        # print(avg_speed_data.head(15))
        # print(avg_speed_data.dtypes)

        return speeds, trp_id, str(t_max)[:10], str(t_min)[:10]


    def export_clean_avg_speed_data(self, avg_speed_data: pd.DataFrame, trp_id: str, t_max: str, t_min: str) -> None:
        filepath = read_metainfo_key(keys_map=["folder_paths", "data", "average_speed", "subfolders", "clean", "path"]) + trp_id + f"_S{t_min}_E{t_max}C.csv"
        filename = trp_id + f"_S{t_min}_E{t_max}C.csv"

        try:
            avg_speed_data.to_csv(filepath, index=False)  # S stands for Start (registration starting date), E stands for End and C for Clean
            update_metainfo(filename, ["average_speeds", "clean_filenames"], mode="append")
            print(f"Average speed data for TRP: {trp_id} saved successfully\n\n")
            return None
        except Exception as e:
            logging.error(traceback.format_exc())
            print(f"\033[91mCouldn't export TRP: {trp_id} volumes data. Error: {e}\033[0m")
            return None


    def execute_cleaning(self, file_path: str, file_name: str) -> None:
        """
        The avg_speed_file_path parameter is the path to the average speed file the user wants to analyze
        The avg_speed_file_name parameter is just the name of the file, needed for secondary purposes or functionalities
        """
        try:
            average_speed_data = pd.read_csv(file_path, sep=";", engine="c")
            #TODO IN THE FUTURE ADD THE UNIFIED data_overview() FUNCTION WHICH WILL PRINT A COMMON DATA OVERVIEW FOR BOTH TRAFFIC VOLUMES DATA AND AVERAGE SPEED ONE

            # Addressing for the empty files problem
            if len(average_speed_data) > 0:
                average_speed_data, trp_id, t_max, t_min = self.clean(average_speed_data)
                self.export_clean_avg_speed_data(average_speed_data, trp_id, t_max, t_min)
            else:
                pass
            return None
        except IndexError as e:
            print(f"\033[91mNo data available for file: {file_name}. Error: {e}\033[0m\n")
            return None
