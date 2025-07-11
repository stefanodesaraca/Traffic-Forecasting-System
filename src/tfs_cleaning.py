import os
from typing import Any, Generator
import json
import datetime
from datetime import datetime
from pathlib import Path
import pprint
import asyncio
import aiofiles
import numpy as np
import pandas as pd
import dask.dataframe as dd

from tfs_base_config import pjh, pmm, tmm

from tfs_utils import GlobalDefinitions




class BaseCleaner:
    def __init__(self):
        self._cwd: str | Path = GlobalDefinitions.CWD.value
        self._ops_folder: str = "ops"
        self._ops_name: str | None = pjh.get_current_project()



class TrafficVolumesCleaner(BaseCleaner):
    def __init__(self):
        super().__init__()


    @staticmethod
    def _get_lanes_number(data: dict[Any, Any]) -> int:
        return max((ln["lane"]["laneNumberAccordingToRoadLink"] for ln in data[0]["node"]["byLane"]))  # Determining the total number of lanes for the TRP taken into consideration. Using a generator comprehension to improve performances


    @staticmethod
    def _get_directions(data: dict[Any, Any]) -> Generator[int, None, None]:
        return (d["heading"] for d in data[0]["node"]["byDirection"])


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


class AverageSpeedCleaner(BaseCleaner):
    def __init__(self):
        super().__init__()


    async def _parse_speeds_async(self, speeds: pd.DataFrame) -> pd.DataFrame | dd.DataFrame | None:
        if speeds.empty:
            return None

        trp_id = str(speeds["trp_id"].unique()[0])

        # Data cleaning
        speeds["coverage"] = speeds["coverage"].replace(",", ".", regex=True).astype("float") * 100
        speeds["mean_speed"] = speeds["mean_speed"].replace(",", ".", regex=True).astype("float")
        speeds["percentile_85"] = speeds["percentile_85"].replace(",", ".", regex=True).astype("float")

        #TODO MERGE DATE AND HOUR TO CREATE ONE zoned_dt_iso COLUMN

        speeds["zoned_dt_iso"] = speeds["date"] + speeds["hour_start"] + GlobalDefinitions.NORWEGIAN_UTC_TIME_ZONE.value

        speeds["hour_start"] = speeds["hour_start"].str[:2] #TODO TO DELETE

        speeds["trp_id"] = speeds["trp_id"].astype("str")
        speeds["date"] = pd.to_datetime(speeds["date"])

        #TODO TO DELETE
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
                            pmm.get(key="folder_paths.data." + GlobalDefinitions.MEAN_SPEED.value + ".subfolders.raw.path") + trp_id + GlobalDefinitions.RAW_MEAN_SPEED_FILENAME_ENDING.value + ".csv", sep=";", **{"engine": "c"}))

                if data is not None: #Checking if data isn't None. If it is, that means that the speeds file was empty
                    await asyncio.to_thread(data.to_csv,pmm.get(key="folder_paths.data." + GlobalDefinitions.MEAN_SPEED.value + ".subfolders.clean.path") + trp_id + GlobalDefinitions.RAW_MEAN_SPEED_FILENAME_ENDING.value + ".csv")

                return None

            else:
                return await self._parse_speeds_async(await asyncio.to_thread(pd.read_csv, pmm.get(key="folder_paths.data." + GlobalDefinitions.MEAN_SPEED.value + ".subfolders.raw.path") + trp_id + GlobalDefinitions.RAW_MEAN_SPEED_FILENAME_ENDING + ".csv", sep=";", **{"engine":"c"}))

        except Exception as e:
            print(f"\033[91mFailed to export speeds for TRP: {trp_id}. Error: {e}\033[0m")
            return None
