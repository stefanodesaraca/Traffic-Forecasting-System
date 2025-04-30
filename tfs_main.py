from tfs_downloader import *
from tfs_eda import *
from tfs_utils import *
from tfs_cleaning import *
from tfs_ml import *
from tfs_models import model_names_and_functions
from tfs_road_network import *
import os
import time
from datetime import datetime
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
import pprint
import math
import asyncio
import dask.distributed
from dask.distributed import Client, LocalCluster

dt_iso_format = "%Y-%m-%dT%H:%M:%S.%fZ"



def manage_ops(functionality: str) -> None:

    if functionality == "1.1":
        ops_name = input("Insert new operation name: ")
        create_ops_folder(ops_name)

    elif functionality == "1.2":
        ops_name = input("Insert the operation to set as active: ")
        write_active_ops_file(ops_name)

    elif functionality == "1.3":
        print("Active operation: ", get_active_ops(), "\n\n")

    else:
        print("\033[91mFunctionality not found, try again with a correct one\033[0m")
        print("\033[91mReturning to the main menu...\033[0m\n\n")
        main()
        
    return None


async def download_data(functionality: str) -> None:

    if functionality == "2.1":
        try:
            print("\nDownloading traffic registration points information for the active operation...")
            ops_name = get_active_ops()
            await traffic_registration_points_to_json(ops_name)
            print("Traffic registration points information downloaded successfully\n\n")
        except Exception as e:
            print(f"\033[91mCouldn't download traffic registration points information for the active operation. Error: {e}\033[0m")
            exit(code=1)

    elif functionality == "2.2":
        time_start = input("Insert starting datetime (of the time frame which you're interested in) - YYYY-MM-DDTHH: ")
        time_end = input("Insert ending datetime (of the time frame which you're interested in) - YYYY-MM-DDTHH: ")

        if check_datetime(time_start) is True and check_datetime(time_end) is True:
            pass
        else:
            print("\033[91mWrong datetime format, try again with a correct one\033[0m")
            print("Returning to the main menu...\n\n")
            main()

        time_start += ":00:00.000Z"
        time_end += ":00:00.000Z"

        await update_metainfo_async(time_start, ["traffic_volumes", "start_date_iso"], mode="equals")
        await update_metainfo_async(time_end, ["traffic_volumes", "end_date_iso"], mode="equals")

        await update_metainfo_async(datetime.strptime(time_start, dt_iso_format).strftime("%Y"), ["traffic_volumes", "start_year"], mode="equals")
        await update_metainfo_async(datetime.strptime(time_start, dt_iso_format).strftime("%m"), ["traffic_volumes", "start_month"], mode="equals")
        await update_metainfo_async(datetime.strptime(time_start, dt_iso_format).strftime("%d"), ["traffic_volumes", "start_day"], mode="equals")
        await update_metainfo_async(datetime.strptime(time_start, dt_iso_format).strftime("%H"), ["traffic_volumes", "start_hour"], mode="equals")

        await update_metainfo_async(datetime.strptime(time_end, dt_iso_format).strftime("%Y"), ["traffic_volumes", "end_year"], mode="equals")
        await update_metainfo_async(datetime.strptime(time_end, dt_iso_format).strftime("%m"), ["traffic_volumes", "end_month"], mode="equals")
        await update_metainfo_async(datetime.strptime(time_end, dt_iso_format).strftime("%d"), ["traffic_volumes", "end_day"], mode="equals")
        await update_metainfo_async(datetime.strptime(time_end, dt_iso_format).strftime("%H"), ["traffic_volumes", "end_hour"], mode="equals")

        relative_delta = relativedelta(datetime.strptime(time_end, dt_iso_format).date(), datetime.strptime(time_start, dt_iso_format).date())
        days_delta = (datetime.strptime(time_end, dt_iso_format).date() - datetime.strptime(time_start, dt_iso_format).date()).days
        years_delta = relative_delta.years if relative_delta.years is not None else 0
        months_delta = relative_delta.months + (years_delta * 12)
        weeks_delta = days_delta // 7

        await update_metainfo_async(days_delta, ["traffic_volumes", "n_days"], mode="equals")
        await update_metainfo_async(months_delta, ["traffic_volumes", "n_months"], mode="equals")
        await update_metainfo_async(years_delta, ["traffic_volumes", "n_years"], mode="equals")
        await update_metainfo_async(weeks_delta, ["traffic_volumes", "n_weeks"], mode="equals")

        print("Downloading traffic volumes data for every registration point for the active operation...")
        await traffic_volumes_data_to_json(time_start=time_start, time_end=time_end)

    elif functionality == "2.3":
        trp_id_list = get_trp_id_list()
        if len(os.listdir(get_raw_traffic_volumes_folder_path())) == 0:
            print("\033[91mDownload volumes data before writing metadata\033[0m")
            return None

        print("\nWriting metadata files...")
        for trp_id in tqdm(trp_id_list): write_trp_metadata(trp_id)
        print("Metadata files successfully written\n\n")

    return None

#TODO ASYNCHRONIZE CLEANING AS WELL
def clean_data(functionality: str) -> None:

    if functionality == "5.6.1":

        traffic_volumes_folder = get_raw_traffic_volumes_folder_path()
        traffic_volumes_file_list = get_raw_traffic_volume_file_list()

        cleaner = TrafficVolumesCleaner()

        for file in traffic_volumes_file_list:
            if file.endswith(".DS_Store") is not True: cleaner.execute_cleaning(traffic_volumes_folder + file)

    elif functionality == "5.6.2":

        average_speed_folder = get_raw_average_speed_folder_path()
        average_speed_file_list = get_raw_avg_speed_file_list()

        cleaner = AverageSpeedCleaner()

        for file in average_speed_file_list:
            if file.endswith(".DS_Store") is not True: cleaner.execute_cleaning(
                file_path=average_speed_folder + file, file_name=file)

    return None


def set_forecasting_options(functionality: str) -> None:

    if functionality == "3.1.1":
        write_forecasting_target_datetime()

    elif functionality == "3.1.2":
        option = input("Press V to read forecasting target datetime for traffic volumes or AS for average speeds: ")
        print("Target datetime: ", read_forecasting_target_datetime(data_kind=option), "\n\n")

    elif functionality == "3.1.3":
        rm_forecasting_target_datetime()

    return None


def execute_eda() -> None:

    clean_traffic_volumes_folder_path = get_clean_traffic_volumes_folder_path()
    clean_average_speed_folder_path = get_clean_average_speed_folder_path()

    clean_traffic_volume_files = os.listdir(get_clean_traffic_volumes_folder_path())
    print("Clean traffic volume files: ", clean_traffic_volume_files, "\n")
    clean_average_speed_files = os.listdir(get_clean_average_speed_folder_path())
    print("Clean average speed files: ", clean_average_speed_files, "\n")

    for v in clean_traffic_volume_files:
        volumes = retrieve_volumes_data(clean_traffic_volumes_folder_path + v)
        analyze_volumes(volumes)
        volumes_data_multicollinearity_test(volumes)

    for s in clean_average_speed_files:
        speeds = retrieve_avg_speed_data(clean_average_speed_folder_path + s)
        analyze_avg_speeds(speeds)
        avg_speeds_data_multicollinearity_test(speeds)

    volumes_speeds = [vs for vs in clean_traffic_volume_files if get_trp_id_from_filename(vs) in [get_trp_id_from_filename(s) for s in clean_average_speed_files]] #Determinig the TRPs which have both traffic volumes and speed data
    #Checking which TRPs have both traffic volumes and speed data available
    print("\n\nClean volumes and average speeds files: ", volumes_speeds)
    print("Number of clean volumes and average speeds files: ", len(volumes_speeds))
    print("\n\n")

    return None

#TODO IN THE FUTURE WE COULD PREDICT percentile_85 AS WELL. EXPLICITELY PRINT THAT FILES METADATA IS NEEDED BEFORE EXECUTING THE WARMUP
def execute_forecast_warmup(functionality: str) -> None:

    models = [m for m in model_names_and_functions.keys()]
    targets = ["volume", "mean_speed"]

    trps = get_trp_id_list()

    #TRPs - Volumes files and road categories
    trps_ids_volumes_by_road_category = {category: [retrieve_trp_clean_volumes_filepath_by_id(trp_id) for trp_id in trps if get_trp_road_category(trp_id) == category and os.path.isdir(retrieve_trp_clean_volumes_filepath_by_id(trp_id)) is False]
                                         for category in get_all_available_road_categories()}
    #The isdir() method is needed since there could be some cases where the volumes files are absent, but TRPs are included in the trps list, so if there isn't on we'll just obtain the path for the clean volumes files folder. Thus, if the string is a path to a folder then don't include it in the trps_ids_by_road_category
    #pprint.pprint(trps_ids_by_road_category)


    #TRPs - Average files and road categories
    trps_ids_avg_speeds_by_road_category = {category: [retrieve_trp_clean_average_speed_filepath_by_id(trp_id) for trp_id in trps if get_trp_road_category(trp_id) == category and os.path.isdir(retrieve_trp_clean_average_speed_filepath_by_id(trp_id)) is False] for category in
                                            get_all_available_road_categories()}

    #Initializing a client to support parallel backend computing and to be able to visualize the Dask client dashboard
    #It's important to instantiate it here since, if it was done in the gridsearch function, it would mean the client would be started and closed everytime the function runs (which is not good)
    cluster = LocalCluster(processes=False) #Check localhost:8787 to watch real-time.
    #By default the number of workers is obtained by dask using the standard os.cpu_count()
    client = Client(cluster)
    #More information about Dask local clusters here: https://docs.dask.org/en/stable/deploying-python.html


    # ------------ Hyperparameter tuning for traffic volumes ML models ------------
    if functionality == "3.2.1":

        merged_volumes_by_category = {}

        # Merge all volumes files by category
        for road_category, volumes_files in trps_ids_volumes_by_road_category.items():
            merged_volumes_by_category[road_category] = merge_volumes_data(volumes_files, road_category=road_category)

        for road_category, v in merged_volumes_by_category.items():

            print(f"\n********************* Executing hyperparameter tuning on traffic volumes data for road category: {road_category} *********************\n")

            volumes_learner = TrafficVolumesLearner(v, client)
            volumes_preprocessed = volumes_learner.preprocess()

            X_train, X_test, y_train, y_test = volumes_learner.split_data(volumes_preprocessed, target=targets[0])

            # -------------- GridSearchCV phase --------------
            for model_name in models:
                volumes_learner.gridsearch(X_train, y_train, target=targets[0], model_name=model_name,
                                           road_category=road_category)

                #Check if workers are still alive
                print("Alive Dask cluster workers: ", dask.distributed.worker.Worker._instances)

                time.sleep(1) #To cool down the system


    # ------------ Hyperparameter tuning for average speed ML models ------------
    elif functionality == "3.2.2":

        clean_average_speed_files = get_clean_average_speed_files_list()

        #TODO TO ADD print(f"********************* Executing hyperparameter tuning on traffic volumes data for road category: {road_category} *********************")

        for s in clean_average_speed_files[:2]:  #TODO AFTER TESTING -> REMOVE [:2] COMBINE ALL FILES DATA INTO ONE BIG DASK DATAFRAME AND REMOVE THIS FOR CYCLE
            avg_speed_learner = AverageSpeedLearner(s, client)
            avg_speeds_preprocessed = avg_speed_learner.preprocess()

            X_train, X_test, y_train, y_test = avg_speed_learner.split_data(avg_speeds_preprocessed, target=targets[1])

            for model_name in models:
                avg_speed_learner.gridsearch(X_train, y_train, target=targets[1], model_name=model_name,
                                             road_category="E")  #TODO "E" IS JUST FOR TESTING PURPOSES

                # Check if workers are still alive
                print("Alive Dask cluster workers: ", dask.distributed.worker.Worker._instances)

                time.sleep(1) #To cool down the system


    # ------------ Train ML models on traffic volumes data ------------
    elif functionality == "3.2.3":

        merged_volumes_by_category = {}

        # Merge all volumes files by category
        for road_category, volumes_files in trps_ids_volumes_by_road_category.items():
            merged_volumes_by_category[road_category] = merge_volumes_data(volumes_files, road_category=road_category)

        for road_category, v in merged_volumes_by_category.items():

            print(f"\n********************* Training models on traffic volumes data for road category: {road_category} *********************\n")

            volumes_learner = TrafficVolumesLearner(v, client)
            volumes_preprocessed = volumes_learner.preprocess()

            X_train, X_test, y_train, y_test = volumes_learner.split_data(volumes_preprocessed, target=targets[0])

            # -------------- Training phase --------------
            for model_name in models: volumes_learner.train_model(X_train, y_train, model_name=model_name, target=targets[0], road_category=road_category)



    elif functionality == "3.2.4":

        print()


    # ------------ Test ML models on traffic volumes data ------------
    elif functionality == "3.2.5":

        merged_volumes_by_category = {}

        # Merge all volumes files by category
        for road_category, volumes_files in trps_ids_volumes_by_road_category.items():
            merged_volumes_by_category[road_category] = merge_volumes_data(volumes_files, road_category=road_category)

        for road_category, v in merged_volumes_by_category.items():

            print(f"\n********************* Testing models on traffic volumes data for road category: {road_category} *********************\n")

            volumes_learner = TrafficVolumesLearner(v, client)
            volumes_preprocessed = volumes_learner.preprocess()

            X_train, X_test, y_train, y_test = volumes_learner.split_data(volumes_preprocessed, target=targets[0])

            # -------------- Testing phase --------------
            for model_name in models: volumes_learner.test_model(X_test, y_test, model_name=model_name, target=targets[0], road_category=road_category)


        print("\n\n")



    elif functionality == "3.2.6":

        print()




    client.close()
    cluster.close()

    return None








def execute_forecasts(functionality: str) -> None:

    #We'll check if the target datetime exists before any forecasting operation could begin.
    #Also, we'll check if the date is within the data we already have (since there's nothing to forecast if we already have the true values (the measurements executed by the TRP sensors) for a specific day)
    #If we already have the data we'll just re-direct the user the main menu.
    #This check will be handled internally by the write_forecasting_target_datetime() function
    check_metainfo_file()

    print("Which kind of data would you like to forecast?")
    print("V: Volumes | AS: Average Speeds")
    option = input("Choice: ")
    target_datetime = read_forecasting_target_datetime(option)

    #One-Point Forecast
    if functionality == "3.3.1":
        trp_id_list = get_trp_id_list()
        print("TRP IDs: ", trp_id_list)
        trp_id = input("Insert TRP ID for forecasting: ")

        if trp_id in trp_id_list:
            trp_road_category = get_trp_road_category(trp_id)
            print("\nTRP road category:", trp_road_category)

            if option == "V":
                one_point_volume_forecaster = OnePointVolumesForecaster(trp_id=trp_id, road_category=trp_road_category)
                one_point_volume_forecaster.preprocess_data(forecasting_target_datetime=target_datetime)
            elif option == "AS":
                pass #TODO DEVELEOP HERE

        else:
            print("\033[91mNon-valid TRP ID, returning to main menu\033[0m")
            main()

    return None










def manage_road_network(functionality: str) -> None:

    if functionality == "4.1":
        pass #TODO TO DEVELOP

    elif functionality == "4.2": #TODO TESTING FOR NOW
        retrieve_edges()
        retrieve_arches()





    return None



































def main():
    while True:
        print("""==================== MENU ==================== 
1. Set pre-analysis information
    1.1 Create an operation
    1.2 Set an operation as active (current one)
    1.3 Check the active operation name
 2. Download data (Trafikkdata API)
    2.1 Traffic registration points information
    2.2 Traffic volumes for every registration point
    2.3 Write metadata file for every TRP
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
 4. Road network graph
    4.1 Graph generation
    4.2 Graph read (from already existing graph)
    4.3 Graph analysis
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

        elif option in ["2.1", "2.2", "2.3"]:
            asyncio.run(download_data(option))

        elif option in ["3.1.1", "3.1.2", "3.1.3"]:
            set_forecasting_options(option)

        elif option in ["3.2.1", "3.2.2", "3.2.3", "3.2.4", "3.2.5"]:
            execute_forecast_warmup(option)

        elif option in ["3.3.1"]:
            execute_forecasts(option)

        elif option in ["4.1", "4.2", "4.3"]:
            manage_road_network(option)

        elif option == "5.2":
            execute_eda()

        elif option in ["5.6.1", "5.6.2"]:
            clean_data(option)

        elif option == "0":
            exit(code=0)

        else:
            print("Wrong option. Insert a valid one")
            print()


if __name__ == "__main__":
    main()

























