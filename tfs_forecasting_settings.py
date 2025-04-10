from tfs_utilities import *
from datetime import datetime
import os

target_datetime_filename = "target_datetime"
forecasting_dt_format = "%Y-%m-%dT%H"  # Datetime format, the hour (H) must be zero-padded and 24-h base, for example: 01, 02, ..., 12, 13, 14, 15, etc.
#In this case we'll only ask for the hour value since, for now, it's the maximum granularity for the predictions we're going to make
cwd = os.getcwd()
target_data = ["V", "AS"]


def write_forecasting_target_datetime() -> None:

    assert os.path.isdir(get_clean_traffic_volumes_folder_path()), "Clean traffic volumes folder missing. Initialize an operation first and then set a forecasting target datetime"
    assert os.path.isdir(get_clean_average_speed_folder_path()), "Clean average speeds folder missing. Initialize an operation first and then set a forecasting target datetime"

    option = input("Press V to set forecasting target datetime for traffic volumes or AS for average speeds: ")
    dt = str(input("Insert Target Datetime (YYYY-MM-DDTHH): ")) #The month number must be zero-padded, for example: 01, 02, etc.

    if os.path.isfile(f"{target_datetime_filename}.json") is False:
        with open(f"{target_datetime_filename}.json", "w") as json_dt: json.dump({"target_dts": {k: None for k in target_data}}, json_dt)

    if check_datetime(dt) is True and option in target_data:
        print("Target datetime set to: ", dt, "\n\n")
        with open(f"{target_datetime_filename}.json", "r") as json_dt:
            t_dts = json.load(json_dt)
            t_dts["target_dts"][option] = dt
        with open(f"{target_datetime_filename}.json", "w") as json_dt: json.dump(t_dts, json_dt)
        return None

    else:
        if check_datetime(dt) is False:
            print("\033[91mWrong datetime format, try again\033[0m")
            exit()
        elif option not in target_data:
            print("\033[91mWrong data forecasting target datetime, try again\033[0m")
            exit()


def read_forecasting_target_datetime(data_kind: str) -> datetime:

    try:
        with open(f"{target_datetime_filename}.json", "r") as json_dt:
            target_dt = json.load(json_dt)["target_dts"][data_kind]
            target_dt = datetime.strptime(target_dt, forecasting_dt_format)
    except FileNotFoundError:
        print("\033[91mTarget Datetime File Not Found\033[0m")
        exit()

    return target_dt


def del_forecasting_target_datetime() -> None:

    try:
        os.remove(cwd + "/" + target_datetime_filename + ".txt")
        print("Target datetime file deleted successfully\n\n")
    except FileNotFoundError:
        print("\033[91mTarget Datetime File Not Found\033[0m")
        exit()

    return None










































