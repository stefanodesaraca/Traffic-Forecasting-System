from datetime import datetime
import os
import time
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
import pprint
import asyncio
import dask.distributed
from dask.distributed import Client, LocalCluster

from tfs_downloader import *
from tfs_eda import *
from tfs_utils import *
from tfs_cleaning import *
from tfs_ml import *
from tfs_road_network import *

dt_iso = "%Y-%m-%dT%H:%M:%S.%fZ"


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
            await traffic_registration_points_to_json()
            print("Traffic registration points information downloaded successfully\n\n")
        except Exception as e:
            print(f"\033[91mCouldn't download traffic registration points information for the active operation. Error: {e}\033[0m")
            sys.exit(1)

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

        await update_metainfo_async(datetime.datetime.strptime(time_start, dt_iso).strftime("%Y"), ["traffic_volumes", "start_year"], mode="equals", )
        await update_metainfo_async(datetime.datetime.strptime(time_start, dt_iso).strftime("%m"), ["traffic_volumes", "start_month"], mode="equals", )
        await update_metainfo_async(datetime.datetime.strptime(time_start, dt_iso).strftime("%d"), ["traffic_volumes", "start_day"], mode="equals", )
        await update_metainfo_async(datetime.datetime.strptime(time_start, dt_iso).strftime("%H"), ["traffic_volumes", "start_hour"], mode="equals", )

        await update_metainfo_async(datetime.datetime.strptime(time_end, dt_iso).strftime("%Y"), ["traffic_volumes", "end_year"], mode="equals", )
        await update_metainfo_async(datetime.datetime.strptime(time_end, dt_iso).strftime("%m"), ["traffic_volumes", "end_month"], mode="equals", )
        await update_metainfo_async(datetime.datetime.strptime(time_end, dt_iso).strftime("%d"), ["traffic_volumes", "end_day"], mode="equals", )
        await update_metainfo_async(datetime.datetime.strptime(time_end, dt_iso).strftime("%H"), ["traffic_volumes", "end_hour"], mode="equals", )

        relative_delta = relativedelta(datetime.datetime.strptime(time_end, dt_iso).date(), datetime.datetime.strptime(time_start, dt_iso).date(), )
        days_delta = (datetime.datetime.strptime(time_end, dt_iso).date() - datetime.datetime.strptime(time_start, dt_iso).date()).days
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

        assert os.path.isfile(read_metainfo_key(keys_map=["common", "traffic_registration_points_file"])), "Download traffic registration points"
        with open(read_metainfo_key(keys_map=["common", "traffic_registration_points_file"]), "r") as trps_data:
            for trp_id in tqdm(list(json.load(trps_data).keys())):
                write_trp_metadata(trp_id)

    return None


# TODO ASYNCHRONIZE CLEANING AS WELL
def clean_data(functionality: str) -> None:
    if functionality == "5.6.1":
        traffic_volumes_folder = read_metainfo_key(keys_map=["folder_paths", "data", "traffic_volumes", "subfolders", "raw", "path"])

        for file in os.listdir(traffic_volumes_folder):
            if file.endswith(".DS_Store") is not True:
                TrafficVolumesCleaner().execute_cleaning(traffic_volumes_folder + file)

    elif functionality == "5.6.2":
        average_speed_folder = read_metainfo_key(keys_map=["folder_paths", "data", "average_speed", "subfolders", "raw", "path"])

        for file in os.listdir(average_speed_folder):
            if file.endswith(".DS_Store") is not True:
                AverageSpeedCleaner().execute_cleaning(file_path=average_speed_folder + file, file_name=file)

    return None


def set_forecasting_options(functionality: str) -> None:
    if functionality == "3.1.1":
        write_forecasting_target_datetime()

    elif functionality == "3.1.2":
        option = input("Press V to read forecasting target datetime for traffic volumes or AS for average speeds: ")
        print("Target datetime: ", read_forecasting_target_datetime(data_kind=option), "\n\n",)

    elif functionality == "3.1.3":
        rm_forecasting_target_datetime()

    return None


def execute_eda() -> None:
    trp_data = import_TRPs_data()
    clean_volumes_folder = read_metainfo_key(keys_map=["folder_paths", "data", "traffic_volumes", "subfolders", "clean", "path"])
    clean_speeds_folder = read_metainfo_key(keys_map=["folder_paths", "data", "average_speed", "subfolders", "clean", "path", ""])

    for v in (trp_id for trp_id in trp_data.keys() if trp_data[trp_id]["checks"]["has_volumes"]):
        volumes = retrieve_volumes_data(clean_volumes_folder + v)
        analyze_volumes(volumes)
        volumes_data_multicollinearity_test(volumes)

    for s in (trp_id for trp_id in trp_data.keys() if trp_data[trp_id]["checks"]["has_speeds"]):
        speeds = retrieve_avg_speed_data(clean_speeds_folder + s)
        analyze_avg_speeds(speeds)
        avg_speeds_data_multicollinearity_test(speeds)

    volumes_speeds = (vs for vs in (trp_id for trp_id in trp_data.keys() if trp_data[trp_id]["checks"]["has_volumes"] and trp_data[trp_id]["checks"]["has_speeds"]))
    # Determining the TRPs which have both traffic volumes and speed data

    print("\n\n")

    return None


# TODO IN THE FUTURE WE COULD PREDICT percentile_85 AS WELL. EXPLICITELY PRINT THAT FILES METADATA IS NEEDED BEFORE EXECUTING THE WARMUP
def execute_forecast_warmup(functionality: str) -> None:
    models = [m for m in model_names_and_functions.keys()]
    clean_volumes_folder = read_metainfo_key(keys_map=["folder_paths", "data", "traffic_volumes", "subfolders", "clean", "path"])
    clean_speeds_folder = read_metainfo_key(keys_map=["folder_paths", "data", "average_speed", "subfolders", "clean", "path"])
    road_categories = set(trp["location"]["roadReference"]["roadCategory"]["id"] for trp in import_TRPs_data())


    # TRPs - Volumes files and road categories
    trps_ids_volumes_by_road_category = {
        category: [clean_volumes_folder + trp_id + "_volumes.csv" for trp_id in
                   filter(lambda trp_id: get_trp_metadata(trp_id)[["road_category"] == category] and get_trp_metadata(trp_id)["checks"]["has_speeds"], get_trp_ids())]
        for category in road_categories
    }
    print(trps_ids_volumes_by_road_category)
    # pprint.pprint(trps_ids_by_road_category)

    # Removing key value pairs from the dictionary where there are less than two dataframes to concatenate, otherwise this would throw an error in the merge() function
    # In this case we're removing the keys directly from the dictionary while iterating on it, so we need to create a shallow copy of the references to its keys and values through the list() methods to then safely delete from the dict itself while iterating on it
    for k, v in list(trps_ids_volumes_by_road_category.items()):
        if len(v) < 2:
            del trps_ids_volumes_by_road_category[k]


    # TRPs - Average speed files and road categories
    trps_ids_avg_speeds_by_road_category = {
        category: [clean_speeds_folder + trp_id + "_speeds.csv" for trp_id in
                   filter(lambda trp_id: get_trp_metadata(trp_id)[["road_category"] == category] and get_trp_metadata(trp_id)["checks"]["has_speeds"], get_trp_ids())]
        for category in road_categories
    }

    # Removing key value pairs from the dictionary where there are less than two dataframes to concatenate, otherwise this would throw an error in the merge() function
    # In this case we're removing the keys directly from the dictionary while iterating on it, so we need to create a shallow copy of the references to its keys and values through the list() methods to then safely delete from the dict itself while iterating on it
    for k, v in list(trps_ids_avg_speeds_by_road_category.items()):
        if len(v) < 2:
            del trps_ids_avg_speeds_by_road_category[k]


    # Initializing a client to support parallel backend computing and to be able to visualize the Dask client dashboard
    # It's important to instantiate it here since, if it was done in the gridsearch function, it would mean the client would be started and closed everytime the function runs (which is not good)
    cluster = LocalCluster(processes=False)  # Check localhost:8787 to watch real-time.
    # By default, the number of workers is obtained by dask using the standard os.cpu_count()
    client = Client(cluster)
    # More information about Dask local clusters here: https://docs.dask.org/en/stable/deploying-python.html

    # ------------ Hyperparameter tuning for traffic volumes ML models ------------
    if functionality == "3.2.1":
        merged_volumes_by_category = {}

        # Merge all volumes files by category
        for road_category, files in trps_ids_volumes_by_road_category.items():
            merged_volumes_by_category[road_category] = merge(files) #Each file contains volumes from a single TRP
            print(f"Shape of the merged data for road category {road_category}: ", (merged_volumes_by_category[road_category].shape[0].compute(), merged_volumes_by_category[road_category].shape[1]))

        for road_category, v in merged_volumes_by_category.items():
            print(f"\n********************* Executing hyperparameter tuning on traffic volumes data for road category: {road_category} *********************\n")

            volumes_learner = TrafficVolumesLearner(v, road_category=road_category, target=target_data["V"], client=client)
            volumes_preprocessed = volumes_learner.preprocess()

            X_train, X_test, y_train, y_test = split_data(volumes_preprocessed, target=target_data_temp["V"])

            # -------------- GridSearchCV phase --------------
            for model_name in models:
                volumes_learner.gridsearch(
                    X_train,
                    y_train,
                    model_name=model_name
                )

                # Check if workers are still alive
                print("Alive Dask cluster workers: ", dask.distributed.worker.Worker._instances)

                time.sleep(1)  # To cool down the system

    # ------------ Hyperparameter tuning for average speed ML models ------------
    elif functionality == "3.2.2":
        merged_speeds_by_category = {}

        for road_category, speeds_files in trps_ids_avg_speeds_by_road_category.items():
            merged_speeds_by_category[road_category] = merge(speeds_files)

        for road_category, s in merged_speeds_by_category.items():
            print(f"********************* Executing hyperparameter tuning on average speed data for road category: {road_category} *********************")

            speeds_learner = AverageSpeedLearner(s, road_category=road_category, target=target_data["AS"], client=client)
            speeds_preprocessed = speeds_learner.preprocess()

            X_train, X_test, y_train, y_test = split_data(speeds_preprocessed, target=target_data_temp["AS"])

            for model_name in models:
                speeds_learner.gridsearch(
                    X_train,
                    y_train,
                    model_name=model_name,
                )

                # Check if workers are still alive
                print("Alive Dask cluster workers: ", dask.distributed.worker.Worker._instances)

                time.sleep(1)  # To cool down the system

    # ------------ Train ML models on traffic volumes data ------------
    elif functionality == "3.2.3":
        merged_volumes_by_category = {}

        # Merge all volumes files by category
        for road_category, files in trps_ids_volumes_by_road_category.items():
            merged_volumes_by_category[road_category] = merge(files)

        for road_category, v in merged_volumes_by_category.items():
            print(f"\n********************* Training models on traffic volumes data for road category: {road_category} *********************\n")

            volumes_learner = TrafficVolumesLearner(v, road_category=road_category, target=target_data["V"], client=client)
            volumes_preprocessed = volumes_learner.preprocess()

            X_train, X_test, y_train, y_test = split_data(volumes_preprocessed, target=target_data_temp["V"])

            # -------------- Training phase --------------
            for model_name in models:
                volumes_learner.train_model(
                    X_train,
                    y_train,
                    model_name=model_name,
                )

    elif functionality == "3.2.4":
        print()

    # ------------ Test ML models on traffic volumes data ------------
    elif functionality == "3.2.5":
        merged_volumes_by_category = {}

        # Merge all volumes files by category
        for road_category, files in trps_ids_volumes_by_road_category.items():
            merged_volumes_by_category[road_category] = merge(files)

        for road_category, v in merged_volumes_by_category.items():
            print(
                f"\n********************* Testing models on traffic volumes data for road category: {road_category} *********************\n"
            )

            volumes_learner = TrafficVolumesLearner(v, road_category=road_category, target=target_data["V"], client=client)
            volumes_preprocessed = volumes_learner.preprocess()

            X_train, X_test, y_train, y_test = split_data(volumes_preprocessed, target=target_data_temp["V"])

            # -------------- Testing phase --------------
            for model_name in models:
                volumes_learner.test_model(
                    X_test,
                    y_test,
                    model_name=model_name,
                )

        print("\n\n")

    elif functionality == "3.2.6":
        print()

    client.close()
    cluster.close()

    return None


def execute_forecasts(functionality: str) -> None:

    check_metainfo_file()

    print("Which kind of data would you like to forecast?")
    print("V: Volumes | AS: Average Speeds")
    option = input("Choice: ")
    dt = read_forecasting_target_datetime(option)

    cluster = LocalCluster(processes=False)
    client = Client(cluster)

    # One-Point Forecast
    if functionality == "3.3.1":
        trp_ids = list(get_trp_ids())
        print("TRP IDs: ", trp_ids)
        trp_id = input("Insert TRP ID for forecasting: ")

        if trp_id in trp_ids:
            trp_road_category = get_trp_metadata(trp_id)["road_category"]
            print("\nTRP road category:", trp_road_category)

            if option == "V":
                one_point_volume_forecaster = OnePointVolumesForecaster(trp_id=trp_id, road_category=trp_road_category)
                volumes_preprocessed = one_point_volume_forecaster.preprocess(target_datetime=dt)

                for model_name in model_names_and_functions.keys():
                    results = one_point_volume_forecaster.forecast_volumes(volumes_preprocessed, model_name=model_name)
                    print(results)


            elif option == "AS":
                pass  #TODO DEVELOP HERE

        else:
            print("\033[91mNon-valid TRP ID, returning to main menu\033[0m")
            main()

    client.close()
    cluster.close()

    return None


def manage_road_network(functionality: str) -> None:
    if functionality == "4.1":
        pass  # TODO TO DEVELOP

    elif functionality == "4.2":  # TODO TESTING FOR NOW
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
            sys.exit(0)

        else:
            print("Wrong option. Insert a valid one")
            print()


if __name__ == "__main__":
    main()
