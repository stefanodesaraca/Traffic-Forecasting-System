import numpy as np
import json
import datetime
from datetime import datetime
import os
import pandas as pd
import pprint
import dask.dataframe as dd
import asyncio
import aiofiles
from pathlib import Path

from sklearn.linear_model import Lasso, GammaRegressor, QuantileRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklego.meta import ZeroInflatedRegressor

from tfs_utils import *


pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)


global_metadata_manager = GlobalMetadataManager()
project_metadata_manager = ProjectMetadataManager()

gp_toolbox = GeneralPurposeToolbox()

dm = DirectoryManager(project_dir=, gp_toolbox=gp_toolbox, global_metadata_manager=global_metadata_manager, project_metadata_manager=project_metadata_manager)


class BaseCleaner:
    def __init__(self):
        self._cwd: str | Path = dm.cwd
        self._ops_folder: str = "ops"
        self._ops_name: str | None = dm.get_current_project()
        self._regressor_types: list = ["lasso", "gamma", "quantile"]


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
    def _get_directions(data: dict[Any, Any]) -> Generator[int, None, None]:
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

        n_lanes = self._get_lanes_number(payload)
        print("Number of lanes: ", n_lanes)
        directions = list(self._get_directions(payload))
        print("Directions available: ", directions)

        #Just for demonstration purposes
        registration_dates = set(map(lambda x: x[:10], self._get_registration_datetimes(payload))) #Conveniently keeping only the date with [:10]
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

            by_hour_structured["trp_id"].append(trp_id)
            by_hour_structured["year"].append(datetime.strptime(reg_datetime, "%Y-%m-%dT%H").strftime("%Y"))
            by_hour_structured["month"].append(datetime.strptime(reg_datetime, "%Y-%m-%dT%H").strftime("%m"))
            by_hour_structured["week"].append(datetime.strptime(reg_datetime, "%Y-%m-%dT%H").strftime("%V"))
            by_hour_structured["day"].append(datetime.strptime(reg_datetime, "%Y-%m-%dT%H").strftime("%d"))
            by_hour_structured["hour"].append(datetime.strptime(reg_datetime, "%Y-%m-%dT%H").strftime("%H"))
            by_hour_structured["volume"].append(edge["node"]["total"]["volumeNumbers"]["volume"] if edge["node"]["total"]["volumeNumbers"] is not None else None) # In some cases the volumeNumbers key could have null as value, so the "volume" key won't be present. In that case we'll directly insert None as value with an if statement
            by_hour_structured["coverage"].append(edge["node"]["total"]["coverage"]["percentage"] or None) # For less recent data it's possible that sometimes coverage can be null, so we'll address this problem like so
            by_hour_structured["date"].append(datetime.strptime(reg_datetime, "%Y-%m-%dT%H").date().isoformat())


        return pd.DataFrame(by_hour_structured).sort_values(by=["date", "hour"], ascending=True)


    def _parse_by_lane(self, payload: dict[Any, Any]) -> pd.DataFrame | dd.DataFrame | None:

        trp_id = payload["trafficData"]["trafficRegistrationPoint"]["id"]
        payload = payload["trafficData"]["volume"]["byHour"]["edges"]

        if self._is_empty(payload):
            print(f"\033[91mNo data found for TRP: {trp_id}\033[0m\n\n")
            return None

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
                by_lane_structured["volume"].append(lane["total"]["volumeNumbers"]["volume"] if lane["node"]["total"]["volumeNumbers"] is not None else None) # In some cases the volumeNumbers key could have null as value, so the "volume" key won't be present. In that case we'll directly insert None as value with an if statement
                by_lane_structured["coverage"].append(lane["total"]["coverage"]["percentage"] or None)
                by_lane_structured["lane"].append(lane["lane"]["laneNumberAccordingToRoadLink"])
                by_lane_structured["date"].append(datetime.strptime(reg_datetime, "%Y-%m-%dT%H").date().isoformat())

        return pd.DataFrame(by_lane_structured).sort_values(by=["date", "hour"], ascending=True)


    def _parse_by_direction(self, payload: dict[Any, Any]) -> pd.DataFrame | dd.DataFrame | None:

        trp_id = payload["trafficData"]["trafficRegistrationPoint"]["id"]
        payload = payload["trafficData"]["volume"]["byHour"]["edges"]

        if self._is_empty(payload):
            print(f"\033[91mNo data found for TRP: {trp_id}\033[0m\n\n")
            return None

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
                by_direction_structured["volume"].append(direction_section["total"]["volumeNumbers"]["volume"] if direction_section["node"]["total"]["volumeNumbers"] is not None else None) # In some cases the volumeNumbers key could have null as value, so the "volume" key won't be present. In that case we'll directly insert None as value with an if statement
                by_direction_structured["coverage"].append(direction_section["total"]["coverage"]["percentage"] or None)
                by_direction_structured["direction"].append(direction_section["heading"])
                by_direction_structured["date"].append(datetime.strptime(reg_datetime, "%Y-%m-%dT%H").date().isoformat())

        return pd.DataFrame(by_direction_structured).sort_values(by=["date", "hour"], ascending=True)


    async def clean_async(self, trp_id: str, export: bool = True) -> None:
        async with aiofiles.open(await read_metainfo_key_async(
                keys_map=["folder_paths", "data", GlobalDefinitions.VOLUME.value, "subfolders", "raw", "path"]) + trp_id + GlobalDefinitions.RAW_VOLUME_FILENAME_ENDING.value + ".json", "r", encoding="utf-8") as m:
            by_hour_df = await asyncio.to_thread(self._parse_by_hour, json.loads(await m.read())) #In case there's no data by_hour_df will be None

        if by_hour_df is not None and by_hour_df.shape[0] > 0:

            await update_trp_metadata_async(trp_id=trp_id, value=trp_id + GlobalDefinitions.RAW_VOLUME_FILENAME_ENDING.value + ".json", metadata_keys_map=["files", GlobalDefinitions.VOLUME.value, "raw"], mode="e")

            try:
                print("Shape before MICE: ", len(by_hour_df), len(by_hour_df.columns))
                print("Number of zeros before MICE: ", len(by_hour_df[by_hour_df["volume"] == 0]))

                by_hour_df = pd.concat([by_hour_df[["trp_id", "date", "year", "month", "day", "week"]],
                                        await asyncio.to_thread(BaseCleaner()._impute_missing_values,
                                                          by_hour_df.drop(columns=["trp_id", "date", "year", "month", "day", "week"], axis=1), r="gamma")], axis=1)

                print("Shape after MICE: ", len(by_hour_df), len(by_hour_df.columns))
                print("Number of zeros after MICE: ", len(by_hour_df[by_hour_df["volume"] == 0]))
                print("Number of negative values (after MICE): ", len(by_hour_df[by_hour_df["volume"] < 0]))

            except ValueError as e:
                print(f"\033[91mValue error raised. Error: {e} Continuing with the cleaning.\033[0m")
                return

            for col in ("year", "month", "week", "day", "hour", "volume"):
                by_hour_df[col] = by_hour_df[col].astype("int")


        if export:
            try:
                by_hour_df.to_csv(await read_metainfo_key_async(keys_map=["folder_paths", "data", GlobalDefinitions.VOLUME.value, "subfolders", "clean", "path"]) + trp_id + GlobalDefinitions.CLEAN_VOLUME_FILENAME_ENDING.value + ".csv", index=False, encoding="utf-8")
                print(f"TRP: {trp_id} data exported correctly\n")

                await update_trp_metadata_async(trp_id=trp_id, value=True, metadata_keys_map=["checks", GlobalDefinitions.HAS_VOLUME_CHECK.value], mode="equals") #Only updating the has_volumes only if it has clean volumes data
                await update_trp_metadata_async(trp_id=trp_id, value=trp_id + GlobalDefinitions.CLEAN_VOLUME_FILENAME_ENDING.value + ".csv", metadata_keys_map=["files", GlobalDefinitions.VOLUME.value, "clean"], mode="e")
                await update_trp_metadata_async(trp_id=trp_id, value=str(by_hour_df["date"].min()), metadata_keys_map=["data_info", GlobalDefinitions.VOLUME.value, "start_date"], mode="e")
                await update_trp_metadata_async(trp_id=trp_id, value=str(by_hour_df["date"].max()), metadata_keys_map=["data_info", GlobalDefinitions.VOLUME.value, "end_date"], mode="e")

                print(f"TRP: {trp_id} metadata updated correctly\n\n")
            except AttributeError as e:
                print(f"\033[91mCouldn't export {trp_id} TRP volumes data. Error: {e}\033[0m")

        print("--------------------------------------------------------\n\n")



class AverageSpeedCleaner(BaseCleaner):
    def __init__(self):
        super().__init__()


    async def _parse_speeds_async(self, speeds: pd.DataFrame) -> pd.DataFrame | dd.DataFrame | None:
        if speeds.empty:
            return None

        trp_id = str(speeds["trp_id"].unique()[0])

        await update_trp_metadata_async(trp_id=trp_id, value=trp_id + GlobalDefinitions.CLEAN_MEAN_SPEED_FILENAME_ENDING.value + ".csv", metadata_keys_map=["files", GlobalDefinitions.MEAN_SPEED.value, "raw"], mode="e")

        # Data cleaning
        speeds["coverage"] = speeds["coverage"].replace(",", ".", regex=True).astype("float") * 100
        speeds["mean_speed"] = speeds["mean_speed"].replace(",", ".", regex=True).astype("float")
        speeds["percentile_85"] = speeds["percentile_85"].replace(",", ".", regex=True).astype("float")
        speeds["hour_start"] = speeds["hour_start"].str[:2]

        speeds["trp_id"] = speeds["trp_id"].astype("str")
        speeds["date"] = pd.to_datetime(speeds["date"])

        t_min = speeds["date"].min()
        t_max = speeds["date"].max()

        await update_trp_metadata_async(trp_id=trp_id, value=str(t_min), metadata_keys_map=["data_info", GlobalDefinitions.MEAN_SPEED.value, "start_date"], mode="e")
        await update_trp_metadata_async(trp_id=trp_id, value=str(t_max), metadata_keys_map=["data_info", GlobalDefinitions.MEAN_SPEED.value, "end_date"], mode="e")

        print("Registrations time-range:")
        print("First day:", t_min)
        print("Last day:", t_max, "\n\n")

        speeds["year"] = speeds["date"].dt.year.astype("int")
        speeds["month"] = speeds["date"].dt.month.astype("int")
        speeds["week"] = speeds["date"].dt.isocalendar().week.astype("int")
        speeds["day"] = speeds["date"].dt.day.astype("int")
        speeds["hour_start"] = speeds["hour_start"].astype("int")

        print("Shape before MICE:", speeds.shape)
        print("Number of zeros before MICE:", len(speeds[speeds["mean_speed"] == 0]))

        try:
            speeds = await asyncio.to_thread(lambda: pd.concat([speeds[["trp_id", "date", "year", "month", "day", "week"]], BaseCleaner()._impute_missing_values(speeds.drop(columns=["trp_id", "date", "year", "month", "day", "week"]), r="gamma")], axis=1))

            print("Shape after MICE:", speeds.shape, "\n")
            print("Number of zeros after MICE:", len(speeds[speeds["mean_speed"] == 0]))
            print("Negative values (volume):", len(speeds[speeds["mean_speed"] < 0]))
        except ValueError as e:
            print(f"\033[91mValueError: {e}. Skipping...\033[0m")
            return None

        for col in ("year", "month", "week", "day", "hour_start"):
            speeds[col] = speeds[col].astype("int")

        #print("Dataframe head:\n", speeds.head(15), "\n")
        #print("Statistics:\n", speeds.drop(columns=["year", "month", "week", "day", "hour_start"]).describe(), "\n")

        grouped = speeds.groupby(["date", "hour_start"], as_index=False).agg({
            "mean_speed": lambda x: np.round(np.mean(x), 2),
            "percentile_85": lambda x: np.round(np.mean(x), 2),
            "coverage": lambda x: np.round(np.mean(x), 2)
        })

        grouped["year"] = grouped["date"].dt.year.astype("int")
        grouped["month"] = grouped["date"].dt.month.astype("int")
        grouped["week"] = grouped["date"].dt.isocalendar().week.astype("int")
        grouped["day"] = grouped["date"].dt.day.astype("int")
        grouped["trp_id"] = trp_id

        return pd.DataFrame({
            "trp_id": grouped["mean_speed"].to_list(),
            "date": grouped["percentile_85"].to_list(),
            "year": grouped["coverage"].to_list(),
            "month": grouped["hour_start"].to_list(),
            "week": grouped["year"].to_list(),
            "day": grouped["month"].to_list(),
            "hour_start": grouped["week"].to_list(),
            "mean_speed": grouped["day"].to_list(),
            "percentile_85": grouped["date"].to_list(),
            "coverage": grouped["trp_id"].to_list()
        }).reindex(sorted(speeds.columns), axis=1)


    async def clean_async(self, trp_id: str, export: bool = True) -> pd.DataFrame | dd.DataFrame | None:
        """
        This function asynchronously executes a cleaning pipeline and lets the user choose to export the cleaned data or not.
        Parameters:
            trp_id: the ID of the traffic registration point (TRP) which we want to clean average speed data for
            export: lets the user export the clean data. By default, this is set to True. If set to False the function will just return the clean data

        Returns:
            pd.DataFrame | None
        """
        try:
            if export:
                # Using to_thread since this is a CPU-bound operation which would otherwise block the event loop until it's finished executing
                data = await self._parse_speeds_async(
                            await asyncio.to_thread(pd.read_csv,
                            await read_metainfo_key_async(keys_map=["folder_paths", "data", GlobalDefinitions.MEAN_SPEED.value, "subfolders", "raw", "path"]) + trp_id + GlobalDefinitions.RAW_MEAN_SPEED_FILENAME_ENDING.value + ".csv", sep=";", **{"engine": "c"}))

                if data is not None: #Checking if data isn't None. If it is, that means that the speeds file was empty
                    await asyncio.to_thread(data.to_csv,await read_metainfo_key_async(keys_map=["folder_paths", "data", GlobalDefinitions.MEAN_SPEED.value, "subfolders", "clean", "path"]) + trp_id + GlobalDefinitions.RAW_MEAN_SPEED_FILENAME_ENDING.value + ".csv")

                    await update_trp_metadata_async(trp_id=trp_id, value=True, metadata_keys_map=["checks", GlobalDefinitions.HAS_MEAN_SPEED_CHECK.value], mode="e")
                    await update_trp_metadata_async(trp_id=trp_id, value=trp_id + GlobalDefinitions.RAW_MEAN_SPEED_FILENAME_ENDING.value + ".csv", metadata_keys_map=["files", GlobalDefinitions.MEAN_SPEED.value, "clean"], mode="e")

                return None

            else:
                return await self._parse_speeds_async(await asyncio.to_thread(pd.read_csv, project_metadata_manager.get(key="folder_paths.data." + GlobalDefinitions.MEAN_SPEED.value + ".subfolders.raw.path") + trp_id + GlobalDefinitions.RAW_MEAN_SPEED_FILENAME_ENDING + ".csv", sep=";", **{"engine":"c"}))

        except FileNotFoundError or IndexError as e:
            print(f"\033[91mNo average speed data for TRP: {trp_id}. Error: {e}\033[0m\n")
            return None
        except Exception as e:
            print(f"\033[91mFailed to export speeds for TRP: {trp_id}. Error: {e}\033[0m")
            return None
