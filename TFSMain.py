from OpsSettings import *
from DataDownloader import *
from TFSUtilities import *
import time

def manage_ops(functionality: str):

    if functionality == "1.1":
        ops_name = input("Insert new operation name: ")
        create_ops_folder(ops_name)

    if functionality == "1.2":
        ops_name = input("Insert the operation to set as active: ")
        write_active_ops_file(ops_name)

    if functionality == "1.3":
        print("Active operation: ", read_active_ops_file(), "\n\n")

    else:
        print("Functionality not found, try again with a correct one")
        print("Returning to the main menu...\n\n")
        main()
        
    return None


def download_data(functionality: str):

    if functionality == "2.1":

        print("Downloading traffic measurement points information for the active operation...")

        ops_name = read_active_ops_file()
        traffic_measurement_points_to_json(ops_name)

    if functionality == "2.2":

        time_start = input("Insert starting datetime (of the time frame which you're interested in) - YYYY-MM-DDTHH:MM:SS: ")
        time_end = input("Insert ending datetime (of the time frame which you're interested in) - YYYY-MM-DDTHH:MM:SS: ")

        if check_datetime(time_start) is True and check_datetime(time_end) is True:
            pass
        else:
            print("Wrong datetime format, try again with a correct one")
            print("Returning to the main menu...\n\n")
            main()

        time_start += ".000Z"
        time_end += ".000Z"

        print("Downloading traffic volumes data for every measurement point for the active operation...")

        ops_name = read_active_ops_file()
        traffic_volumes_data_to_json(ops_name, time_start=time_start, time_end=time_end)


    return None


























def main():
    while True:
        print("""==================== MENU ==================== 
1. Set pre-analysis information
    1.1 Create an operation
    1.2 Set an operation as active (current one)
    1.3 Check the active operation name
 2. Download data (Trafikkdata API)
    2.1 Traffic measurement points information
    2.2 Traffic volumes for every measurement point
 3. Forecast
    3.1 Set forecasting target datetime
    3.2 Forecast warmup
    3.3 Execute forecast
        3.3.1 One-Point Forecast
        3.3.2 A2B Forecast
 4. Road network graph generation
 5. Other options
    5.1 Set forecasting system folders (manually)
    5.2 EDA (Exploratory Data Analysis)
    5.3 Erase all data about an operation
    5.4 Find best model for the current operation
    5.5 Analyze pre-existing road network graph
 0. Exit""")

        option = input("Choice: ")
        print()

        if option == "1.1":
            manage_ops("1.1")

        if option == "1.2":
            manage_ops("1.2")

        if option == "1.3":
            manage_ops("1.3")

        if option == "2.1":
            download_data("2.1")

        if option == "2.2":
            download_data("2.2")

        elif option == "0":
            exit()

        else:
            print("Wrong option. Insert a valid one")
            print()


if __name__ == "__main__":
    main()

























