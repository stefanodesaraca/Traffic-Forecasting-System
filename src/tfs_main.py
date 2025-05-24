from datetime import datetime
import os
import time
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
import pprint
import asyncio
from typing import cast
from asyncio import Semaphore

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


async def download_volumes(functionality: str) -> None:
    if functionality == "2.1":
        try:
            print("\nDownloading traffic registration points information for the active operation...")
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

        relative_delta = relativedelta(datetime.datetime.strptime(time_end, dt_iso).date(), datetime.datetime.strptime(time_start, dt_iso).date(), )
        days_delta = (datetime.datetime.strptime(time_end, dt_iso).date() - datetime.datetime.strptime(time_start, dt_iso).date()).days
        years_delta = relative_delta.years if relative_delta.years is not None else 0
        months_delta = relative_delta.months + (years_delta * 12)
        weeks_delta = days_delta // 7

        await update_metainfo_async(days_delta, ["traffic_volumes", "n_days"], mode="equals")
        await update_metainfo_async(months_delta, ["traffic_volumes", "n_months"], mode="equals")
        await update_metainfo_async(years_delta, ["traffic_volumes", "n_years"], mode="equals") #TODO THIS CREATES A SECOND n_years. WHY DOESN'T IT OVERWRITE THE OLD n_years: null?
        await update_metainfo_async(weeks_delta, ["traffic_volumes", "n_weeks"], mode="equals")

        print("Downloading traffic volumes data for every registration point for the active operation...")
        await traffic_volumes_data_to_json(time_start=time_start, time_end=time_end)

    elif functionality == "2.3":

        if len(os.listdir(read_metainfo_key(keys_map=["folder_paths", "data", "trp_metadata", "path"]))) == 0:
            for trp_id in tqdm(import_TRPs_data().keys()):
                write_trp_metadata(trp_id, **{"trp_data": import_TRPs_data()[trp_id]})
        else:
            print("Metadata had already been computed.")

    return None


async def clean_data(functionality: str) -> None:
    semaphore = Semaphore(50)

    async def limited_clean(trp_id: str, cleaner: TrafficVolumesCleaner | AverageSpeedCleaner, export: bool = True):
        async with semaphore:
            await cleaner.clean_async(trp_id, export=export)

    if functionality == "5.6.1":
            await asyncio.gather(*(limited_clean(trp_id=trp_id, cleaner=TrafficVolumesCleaner(), export=True) for trp_id in get_trp_ids())) # The star (*) in necessary since gather() requires the coroutines to fed as positional arguments of the function. So we can unpack the list with *
    elif functionality == "5.6.2":
            await asyncio.gather(*(limited_clean(trp_id=trp_id, cleaner=AverageSpeedCleaner(), export=True) for trp_id in get_trp_ids()))

    return None


def set_forecasting_options(functionality: str) -> None:
    if functionality == "3.1.1":
        write_forecasting_target_datetime()

    elif functionality == "3.1.2":
        option = input("Press V to read forecasting target datetime for traffic volumes or AS for average speeds: ")
        print("Target datetime: ", read_forecasting_target_datetime(data_kind=option), "\n\n",)

    elif functionality == "3.1.3":
        reset_forecasting_target_datetime()

    return None


def execute_eda() -> None:
    trp_data = import_TRPs_data()
    clean_volumes_folder = read_metainfo_key(keys_map=["folder_paths", "data", "traffic_volumes", "subfolders", "clean", "path"])
    clean_speeds_folder = read_metainfo_key(keys_map=["folder_paths", "data", "average_speed", "subfolders", "clean", "path"])

    for v in (trp_id for trp_id in trp_data.keys() if get_trp_metadata(trp_id=trp_id)["checks", "has_volumes"]):
        volumes = pd.read_csv(clean_volumes_folder + v + "_volumes_C.csv")
        analyze_volumes(volumes)
        volumes_data_multicollinearity_test(volumes)

    for s in (trp_id for trp_id in trp_data.keys() if get_trp_metadata(trp_id=trp_id)["checks", "has_speeds"]):
        speeds = pd.read_csv(clean_speeds_folder + s + "_speeds_C.csv")
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
    road_categories = set(trp["location"]["roadReference"]["roadCategory"]["id"] for trp in import_TRPs_data().values())


    # TRPs - Volumes files and road categories
    trps_ids_volumes_by_road_category = {
        category: [clean_volumes_folder + trp_id + "_volumes_C.csv" for trp_id in
                   filter(lambda trp_id: get_trp_metadata(trp_id)["trp_data"]["location"]["roadReference"]["roadCategory"]["id"] == category and get_trp_metadata(trp_id)["checks"]["has_volumes"], get_trp_ids())]
        for category in road_categories
    }
    trps_ids_volumes_by_road_category = {k: v for k, v in trps_ids_volumes_by_road_category.items() if len(v) >= 2}
    # Removing key value pairs from the dictionary where there are less than two dataframes to concatenate, otherwise this would throw an error in the merge() function

    print(trps_ids_volumes_by_road_category)
    # pprint.pprint(trps_ids_by_road_category)


    # TRPs - Average speed files and road categories
    trps_ids_avg_speeds_by_road_category = {
        category: [clean_speeds_folder + trp_id + "_speeds_C.csv" for trp_id in
                   filter(lambda trp_id: get_trp_metadata(trp_id)["trp_data"]["location"]["roadReference"]["roadCategory"]["id"] == category and get_trp_metadata(trp_id)["checks"]["has_speeds"], get_trp_ids())]
        for category in road_categories
    }
    trps_ids_avg_speeds_by_road_category = {k: v for k, v in trps_ids_avg_speeds_by_road_category.items() if len(v) >= 2}
    # Removing key value pairs from the dictionary where there are less than two dataframes to concatenate, otherwise this would throw an error in the merge() function


    def process_data(
            trps_ids_by_road_category: dict[str, list[str]],
            models: list[str],
            learner_class: type[TFSLearner], #Keeping the flexibility of this parameter for now to be able to add other types of learner classes in the future
            target: str,
            process_description: str,
            preprocessor_method: str,
            learner_method: str
    ) -> None:

        merged_data_by_category = {}

        for road_category, files in trps_ids_by_road_category.items():
            merged_data_by_category[road_category] = merge(files)
            print(f"Shape of the merged data for road category {road_category}: ", (merged_data_by_category[road_category].shape[0].compute(), merged_data_by_category[road_category].shape[1]))

        for road_category, data in merged_data_by_category.items():
            print(f"\n********************* Executing {process_description} for road category: {road_category} *********************\n")


            preprocessor = TFSPreprocessor(data=data, road_category=road_category, target=cast(Literal["traffic_volumes", "average_speed"], target), client=client)
            preprocessing_method = getattr(preprocessor, preprocessor_method) #Getting the appropriate preprocessing method based on the target variable to preprocess
            preprocessed_data = preprocessing_method() #Calling the preprocessing method

            X_train, X_test, y_train, y_test = split_data(preprocessed_data, target=target)
            #print(X_train.head(5), X_test.head(5), y_train.head(5), y_test.head(5))

            learner = learner_class(road_category=road_category, target=cast(Literal["traffic_volumes", "average_speed"], target), client=client) # This client is ok here since the process_data function (in which it's located) only gets called after the client is opened as a context manager afterward (see down below in the code) *
            #Using cast() to tell the type checker that the "target" variable is actually a Literal

            for model_name in models:
                method = getattr(learner, learner_method)
                method(X_train if learner_method != "test_model" else X_test,
                       y_train if learner_method != "test_model" else y_test,
                       model_name=model_name)

                print("Alive Dask cluster workers: ", dask.distributed.worker.Worker._instances)
                time.sleep(1)  # To cool down the system

        return None


    with dask_cluster_client(processes=False) as client: #*

        if functionality == "3.2.1":
            process_data(trps_ids_volumes_by_road_category, models, TFSLearner, target_data["V"], "hyperparameter tuning on traffic volumes data", "preprocess_volumes", "gridsearch")

        elif functionality == "3.2.2":
            process_data(trps_ids_avg_speeds_by_road_category, models, TFSLearner, target_data["AS"], "hyperparameter tuning on average speed data", "preprocess_speeds", "gridsearch")

        elif functionality == "3.2.3":
            process_data(trps_ids_volumes_by_road_category, models, TFSLearner, target_data["V"], "training models on traffic volumes data", "preprocess_volumes", "train_model")

        elif functionality == "3.2.4":
            process_data(trps_ids_volumes_by_road_category, models, TFSLearner, target_data["AS"], "training models on average speed data", "preprocess_speeds", "train_model")

        elif functionality == "3.2.5":
            process_data(trps_ids_volumes_by_road_category, models, TFSLearner, target_data["V"], "testing models on traffic volumes data", "preprocess_volumes", "test_model")

        elif functionality == "3.2.6":
            process_data(trps_ids_volumes_by_road_category, models, TFSLearner, target_data["AS"], "testing models on average speed data", "preprocess_speeds", "test_model")

    return None


#TODO TO IMPROVE, OPTIMIZE AND SIMPLIFY
def get_forecaster(option: str, trp_id: str, road_category: str, target_data: str) -> tuple[OnePointForecaster, str, dd.DataFrame] | tuple[None, None, None]:
    forecaster_info = {
        "V": {
            "class": OnePointForecaster,
            "method": "forecast",
            "check": "has_volumes"
        },
        "AS": {
            "class": OnePointForecaster,
            "method": "forecast",
            "check": "has_speeds"
        }
    }

    info = forecaster_info[option]
    if get_trp_metadata(trp_id)["checks"][info["check"]]:
        forecaster = info["class"](trp_id=trp_id, road_category=road_category, target=target_data)
        return forecaster, info["method"], forecaster.preprocess()
    else:
        print(f"TRP {trp_id} doesn't have {option.lower()} data, returning to main menu")
        return None, None, None


def execute_forecasts(functionality: str) -> None:
    check_metainfo_file()

    print("Which kind of data would you like to forecast?")
    print("V: Volumes | AS: Average Speeds")
    option = input("Choice: ").upper()

    if option not in ["V", "AS"]:
        print("Invalid option, returning to main menu")
        return

    with dask_cluster_client(processes=False) as client:
        if functionality == "3.3.1":
            trp_ids = get_trp_ids()
            print("TRP IDs: ", trp_ids)
            trp_id = input("Insert TRP ID for forecasting: ")

            if trp_id in trp_ids:
                trp_road_category = get_trp_metadata(trp_id)["trp_data"]["location"]["roadReference"]["roadCategory"]["id"]
                print("\nTRP road category:", trp_road_category)

                forecaster, method, preprocessed_data = get_forecaster(option, trp_id, trp_road_category, target_data[option])

                if forecaster:
                    for model_name in model_names_and_functions.keys():
                        results = getattr(forecaster, method)(preprocessed_data, model_name=model_name)
                        print(results)
            else:
                print("\033[91mNon-valid TRP ID, returning to main menu\033[0m")
                return

    return None


def manage_road_network(functionality: str) -> None:
    if functionality == "4.1":
        pass  # TODO TO DEVELOP

    elif functionality == "4.2":  # TODO TESTING FOR NOW
        retrieve_edges()
        retrieve_arches()

    return None


def main():
    menu_options = {
        "1.1": manage_ops,
        "1.2": manage_ops,
        "1.3": manage_ops,
        "2.1": download_volumes,
        "2.2": download_volumes,
        "2.3": download_volumes,
        "3.1.1": set_forecasting_options,
        "3.1.2": set_forecasting_options,
        "3.1.3": set_forecasting_options,
        "3.2.1": execute_forecast_warmup,
        "3.2.2": execute_forecast_warmup,
        "3.2.3": execute_forecast_warmup,
        "3.2.4": execute_forecast_warmup,
        "3.2.5": execute_forecast_warmup,
        "3.3.1": execute_forecasts,
        "4.1": manage_road_network,
        "4.2": manage_road_network,
        "4.3": manage_road_network,
        "5.2": execute_eda,
        "5.6.1": clean_data,
        "5.6.2": clean_data,
    }

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
    3.3 Execute forecast
        3.3.1 One-Point Forecast
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

        if option == "0":
            sys.exit(0)
        elif option in menu_options.keys():
            functionality = menu_options[option]
            if asyncio.iscoroutinefunction(functionality):
                asyncio.run(functionality(option))
            else:
                functionality(option)
        else:
            print("Wrong option. Insert a valid one")
            print()


if __name__ == "__main__":
    main()
