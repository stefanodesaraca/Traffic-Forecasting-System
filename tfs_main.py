import tfs_ml
from tfs_ops_settings import *
from tfs_data_downloader import *
from tfs_forecasting_settings import *
from tfs_data_exploration import *
from tfs_utilities import *
from tfs_cleaning import *
from tfs_ml import *
from tfs_models import model_names_and_functions
import os
import time
from datetime import datetime


def manage_ops(functionality: str) -> None:

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


def download_data(functionality: str) -> None:

    if functionality == "2.1":

        try:
            print("Downloading traffic measurement points information for the active operation...")

            ops_name = read_active_ops_file()
            traffic_registration_points_to_json(ops_name)

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


def clean_data(functionality: str) -> None:

    if functionality == "5.6.1":

        traffic_volumes_folder = get_raw_traffic_volumes_folder_path()
        traffic_volumes_file_list = get_traffic_volume_file_list()
        cleaner = TrafficVolumesCleaner()

        for file in traffic_volumes_file_list:
            cleaner.execute_cleaning(traffic_volumes_folder + file)

    elif functionality == "5.6.2":

        average_speed_folder = get_raw_average_speed_folder_path()
        average_speed_file_list = get_raw_avg_speed_file_list()

        cleaner = AverageSpeedCleaner()

        for file in average_speed_file_list:
            cleaner.execute_cleaning(file_path=average_speed_folder + file, file_name=file)


    return None


def set_forecasting_options(functionality: str) -> None:

    if functionality == "3.1.1":
        write_forecasting_target_datetime()

    elif functionality == "3.1.2":
        print("Target datetime: ", read_forecasting_target_datetime(), "\n\n")

    elif functionality == "3.1.3":
        del_forecasting_target_datetime()

    return None


def execute_eda() -> None:

    clean_traffic_volumes_folder_path = get_clean_traffic_volumes_folder_path()
    clean_average_speed_folder_path = get_clean_average_speed_folder_path()

    clean_traffic_volume_files = [clean_traffic_volumes_folder_path + vf for vf in os.listdir(get_clean_traffic_volumes_folder_path())]
    print("Clean traffic volume files: ", clean_traffic_volume_files, "\n")

    for v in clean_traffic_volume_files:
        volumes = retrieve_volumes_data(v)
        analyze_volumes(volumes)
        test_volumes_data_for_multicollinearity(volumes)


    clean_average_speed_files = [clean_average_speed_folder_path + sf for sf in os.listdir(get_clean_average_speed_folder_path())]
    print("Clean average speed files: ", clean_average_speed_files, "\n")

    for s in clean_average_speed_files:
        speeds = retrieve_avg_speed_data(s)
        analyze_avg_speeds(speeds)
        test_avg_speeds_data_for_multicollinearity(speeds)


    volumes_and_speeds = [vs for vs in clean_traffic_volume_files if vs.split("/")[-1].split("_")[0] in [v.split("/")[-1].split("_")[0] for v in clean_average_speed_files]] #Determinig the TRPs which have both traffic volumes and speed data

    print("\n\nClean volumes and average speeds files: ", volumes_and_speeds)
    print("Number of clean volumes and average speeds files: ", len(volumes_and_speeds))


    print("\n\n")

    return None


def execute_forecast_warmup(functionality: str) -> None:

    models = [m for m in model_names_and_functions.keys()]


    #TODO SPLIT THE DATA BY ROAD CATEGORY AND SORT IT BY YEAR, MONTH AND DAY ascending=True AFTER HAVING CONCATENATED IT



    # ------------ Hyperparameter tuning for traffic volumes ML models ------------
    if functionality == "3.2.1":

        clean_traffic_volume_files = get_clean_volume_files()

        for v in clean_traffic_volume_files[:2]: #TODO AFTER TESTING -> REMOVE [:2] COMBINE ALL FILES DATA INTO ONE BIG DASK DATAFRAME AND REMOVE THIS FOR CYCLE
            volumes_forecaster = TrafficVolumesModelLearner(v)
            volumes_preprocessed = volumes_forecaster.volumes_ml_preprocessing_pipeline()

            #We'll skip variable selection since there aren't many variables to choose as predictors

            X_train, X_test, y_train, y_test = volumes_forecaster.split_data(volumes_preprocessed)

            # -------------- GridSearchCV phase --------------
            for model_name in models: volumes_forecaster.gridsearch_for_model(X_train, y_train, model_name=model_name)


    # ------------ Hyperparameter tuning for average speed ML models ------------
    elif functionality == "3.2.2":

        print()


    # ------------ Train ML models on traffic volumes data ------------
    elif functionality == "3.2.3":

        clean_traffic_volume_files = get_clean_volume_files()

        for v in clean_traffic_volume_files[:1]:  #TODO AFTER TESTING -> REMOVE [:2] COMBINE ALL FILES DATA INTO ONE BIG DASK DATAFRAME AND REMOVE THIS FOR CYCLE
            volumes_forecaster = TrafficVolumesModelLearner(v)
            volumes_preprocessed = volumes_forecaster.volumes_ml_preprocessing_pipeline()

            X_train, X_test, y_train, y_test = volumes_forecaster.split_data(volumes_preprocessed, return_pandas=True)

            # -------------- Training phase --------------
            for model_name in models: volumes_forecaster.train_model(X_train, y_train, model_name=model_name)



    elif functionality == "3.2.4":

        print()

    elif functionality == "3.2.5":

        clean_traffic_volume_files = get_clean_volume_files()

        for v in clean_traffic_volume_files[:1]:  # TODO AFTER TESTING -> REMOVE [:2] COMBINE ALL FILES DATA INTO ONE BIG DASK DATAFRAME AND REMOVE THIS FOR CYCLE
            volumes_forecaster = TrafficVolumesModelLearner(v)
            volumes_preprocessed = volumes_forecaster.volumes_ml_preprocessing_pipeline()

            X_train, X_test, y_train, y_test = volumes_forecaster.split_data(volumes_preprocessed, return_pandas=True)

            # -------------- Testing phase --------------
            for model_name in models: volumes_forecaster.test_model(X_test, y_test, model_name=model_name)




    return None



def execute_one_point_forecast(functionality: str):

    if functionality == "3.3.1":

        trp_id_list = get_trp_id_list()
        print("TRP IDs: ", trp_id_list)

        trp_id = input("Insert TRP ID for forecasting: ")

        if trp_id in trp_id_list:

            trp_road_category = get_trp_road_category(trp_id)

            one_point_volume_forecaster = OnePointVolumesForecaster(trp_id=trp_id, road_category=trp_road_category)
            #one_point_volume_forecaster.pre_process_data()

        else:
            print("Non-valid TRP ID, returning to main menu")
            main()

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
        3.1.1 Write forecasting target datetime
        3.1.2 Read forecasting target datetime
        3.1.3 Delete forecasting target datetime
    3.2 Forecast warmup
        3.2.1 Execute hyperparameter tuning (GridSearchCV) for traffic volumes ML models
        3.2.2 Execute hyperparameter tuning (GridSearchCV) for average speed ML models
        3.2.3 Train models on traffic volumes data
        3.2.4 Train models on average speed data
        3.2.5 Test models on traffic volumes data
        3.2.6 Test models on average speed data
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

        if option in ["1.1", "1.2", "1.3"]:
            manage_ops(option)

        elif option in ["2.1", "2.2"]:
            download_data(option)

        elif option in ["3.1.1", "3.1.2", "3.1.3"]:
            set_forecasting_options(option)

        elif option in ["3.2.1", "3.2.2", "3.2.3", "3.2.4", "3.2.5"]:
            execute_forecast_warmup(option)

        elif option in ["3.3.1"]:
            execute_one_point_forecast(option)

        elif option == "5.2":
            execute_eda()

        elif option in ["5.6.1", "5.6.2"]:
            clean_data(option)

        elif option == "0":
            exit()

        else:
            print("Wrong option. Insert a valid one")
            print()


if __name__ == "__main__":
    main()

























