from typing import Any, Generator
import datetime
from datetime import datetime
import asyncio
import pandas as pd
import dask.dataframe as dd



class BaseCleaner:
    def __init__(self):
        ...

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







