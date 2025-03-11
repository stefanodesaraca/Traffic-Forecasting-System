from tfs_ops_settings import *
from tfs_data_downloader import *
from tfs_utilities import *
from tfs_cleaning import *
import time

def manage_ops(functionality: str):

    if functionality == "1.1":
        ops_name = input("Insert new operation name: ")
        create_ops_folder(ops_name)

    elif functionality == "1.2":
        ops_name = input("Insert the operation to set as active: ")
        write_active_ops_file(ops_name)

    elif functionality == "1.3":
        print("Active operation: ", read_active_ops_file(), "\n\n")

    else:
        print("Functionality not found, try again with a correct one")
        print("Returning to the main menu...\n\n")
        main()
        
    return None


def download_data(functionality: str):

    if functionality == "2.1":

        try:
            print("Downloading traffic measurement points information for the active operation...")

            ops_name = read_active_ops_file()
            traffic_measurement_points_to_json(ops_name)

            print("Traffic measurement points information downloaded successfully\n\n")

        except:
            print("\033[91mCouldn't download traffic measurement points information for the active operation\033[0m")


    elif functionality == "2.2":

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


def clean_data(functionality: str):

    if functionality == "5.6.1":

        traffic_volumes_folder = get_raw_traffic_volumes_folder_path()
        traffic_volumes_file_list = get_traffic_volume_file_list()
        cleaner = TrafficVolumesCleaner()

        #TODO REMOVE THE [:2] AFTER TESTING
        for file in traffic_volumes_file_list[:2]:
            cleaner.execute_cleaning(traffic_volumes_folder + file)

    elif functionality == "5.6.2":

        average_speed_folder = get_raw_average_speed_folder_path()
        average_speed_file_list = get_raw_avg_speed_file_list()

        cleaner = AverageSpeedCleaner()

        #TODO TESTING WITH ONLY THE FIRST TWO FILES WITH [:2]
        for file in average_speed_file_list[:2]:
            cleaner.execute_cleaning(file_path=average_speed_folder + file, file_name=file)


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
    5.6 Clean data
        5.6.1 Clean traffic volumes data
        5.6.2 Clean average speed data
    
 0. Exit""")

        option = input("Choice: ")
        print()

        if option == "1.1":
            manage_ops(option)

        elif option == "1.2":
            manage_ops(option)

        elif option == "1.3":
            manage_ops(option)

        elif option == "2.1":
            download_data(option)

        elif option == "2.2":
            download_data(option)

        elif option == "5.6.1":
            clean_data(option)

        elif option == "5.6.2":
            clean_data(option)

        elif option == "0":
            exit()

        else:
            print("Wrong option. Insert a valid one")
            print()


if __name__ == "__main__":
    main()

























