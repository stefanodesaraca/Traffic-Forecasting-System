from tfs_utils import *
from datetime import datetime
import os

metainfo_filename = "metainfo"
forecasting_dt_format = "%Y-%m-%dT%H"  # Datetime format, the hour (H) must be zero-padded and 24-h base, for example: 01, 02, ..., 12, 13, 14, 15, etc.
#In this case we'll only ask for the hour value since, for now, it's the maximum granularity for the predictions we're going to make
cwd = os.getcwd()
target_data = ["V", "AS"]
ops_folder = "ops"

#TODO FIND A WAY TO CHECK WHICH IS THE LAST DATETIME AVAILABLE FOR BOTH AVERAGE SPEED (CLEAN) AND TRAFFIC VOLUMES (CLEAN)

def write_forecasting_target_datetime(ops_name: str) -> None:

    assert os.path.isdir(get_clean_traffic_volumes_folder_path()), "Clean traffic volumes folder missing. Initialize an operation first and then set a forecasting target datetime"
    assert os.path.isdir(get_clean_average_speed_folder_path()), "Clean average speeds folder missing. Initialize an operation first and then set a forecasting target datetime"

    option = input("Press V to set forecasting target datetime for traffic volumes or AS for average speeds: ")
    dt = str(input("Insert Target Datetime (YYYY-MM-DDTHH): ")) #The month number must be zero-padded, for example: 01, 02, etc.

    if check_datetime(dt) is True and option in target_data:
        print("Target datetime set to: ", dt, "\n\n")
        with open(f"{ops_folder}/{ops_name}/{metainfo_filename}.json", "r") as m:
            metainfo = json.load(m)
            metainfo["forecasting"]["target_datetimes"][option] = dt
        with open(f"{ops_folder}/{ops_name}/{metainfo_filename}.json", "w") as m: json.dump(metainfo, m, indent=4)
        return None

    else:
        if check_datetime(dt) is False:
            print("\033[91mWrong datetime format, try again\033[0m")
            exit(code=1)
        elif option not in target_data:
            print("\033[91mWrong data forecasting target datetime, try again\033[0m")
            exit(code=1)


def read_forecasting_target_datetime(data_kind: str, ops_name: str) -> datetime:
    try:
        with open(f"{ops_folder}/{ops_name}/{metainfo_filename}.json", "r") as m:
            target_dt = json.load(m)["forecasting"]["target_datetimes"][data_kind]
            target_dt = datetime.strptime(target_dt, forecasting_dt_format)
            return target_dt
    except FileNotFoundError:
        print("\033[91mTarget Datetime File Not Found\033[0m")
        exit(code=1)


def rm_forecasting_target_datetime(ops_name: str) -> None:
    try:
        print("For which data kind do you want to remove the forecasting target datetime?")
        option = input("Press V to set forecasting target datetime for traffic volumes or AS for average speeds:" )
        with open(f"{ops_folder}/{ops_name}/{metainfo_filename}.json", "r") as m:
            metainfo = json.load(m)
            metainfo["forecasting"]["target_datetimes"][option] = None
        with open(f"{ops_folder}/{ops_name}/{metainfo_filename}.json", "w") as m: json.dump(metainfo, m, indent=4)
        print("Target datetime file deleted successfully\n\n")
        return None
    except KeyError:
        print("\033[91mTarget datetime not found\033[0m")
        exit(code=1)










































