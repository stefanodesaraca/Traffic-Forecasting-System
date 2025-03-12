from datetime import datetime
import os

target_datetime_filename = "target_datetime"
dt_format = "%Y-%m-%dT%H:%M:%S"  # Datetime format, the hour (H) must be zero-padded and 24-h base, for example: 01, 02, ..., 12, 13, 14, 15, etc.


def write_forecasting_target_datetime():

    dt = input("Insert Target Datetime (YYYY-MM-DD hh:mm): ") #The month number must be zero-padded, for example: 01, 02, etc.

    formatted_datetime = datetime.strptime(dt, dt_format)
    print("Target Datetime: ", formatted_datetime)

    with open(f"{target_datetime_filename}.txt", "w") as t_dt_writer:
        t_dt_writer.write(str(formatted_datetime))

    return None


def read_forecasting_target_datetime():

    #target_dt = None

    try:
        with open(f"{target_datetime_filename}.txt", "r") as t_dt_reader:

            target_dt = t_dt_reader.read()
            target_dt = datetime.strptime(target_dt, dt_format)

    except FileNotFoundError:
        print("\033[91mTarget Datetime File Not Found\033[0m")
        exit()

    return target_dt


def del_forecasting_target_datetime():

    try:
        os.remove(target_datetime_filename)
    except FileNotFoundError:
        print("\033[91mTarget Datetime File Not Found\033[0m")
        exit()

    return None










































