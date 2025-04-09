from tfs_utilities import check_datetime
from datetime import datetime
import os

target_datetime_filename = "target_datetime"
forecasting_dt_format = "%Y-%m-%dT%H"  # Datetime format, the hour (H) must be zero-padded and 24-h base, for example: 01, 02, ..., 12, 13, 14, 15, etc.
#In this case we'll only ask for the hour value since, for now, it's the maximum granularity for the predictions we're going to make
cwd = os.getcwd()


def write_forecasting_target_datetime() -> None:

    dt = str(input("Insert Target Datetime (YYYY-MM-DDTHH): ")) #The month number must be zero-padded, for example: 01, 02, etc.

    if check_datetime(dt) is True:
        print("Target datetime set to: ", dt, "\n\n")
        with open(f"{target_datetime_filename}.txt", "w") as t_dt_writer: t_dt_writer.write(dt)
        return None
    else:
        print("\033[91mWrong datetime format, try again\033[0m")
        exit()


def read_forecasting_target_datetime() -> datetime:

    try:
        with open(f"{target_datetime_filename}.txt", "r") as t_dt_reader:
            target_dt = t_dt_reader.read()
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










































