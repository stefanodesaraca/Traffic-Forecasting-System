import json
import os
import time
from datetime import datetime
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
from tfs_ml_configs import *



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
        years_delta = relative_delta.years or 0
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
        print("Target datetime: ", read_forecasting_target_datetime(target=option), "\n\n", )

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
    models = model_definitions["class_instance"].values()


    def preprocess_data(files: list[str], road_category: str, target: str) -> tuple[dd.DataFrame, dd.DataFrame, dd.DataFrame, dd.DataFrame]:

        print(f"\n********************* Executing data preprocessing for road category: {road_category} *********************\n")

        preprocessor = TFSPreprocessor(data=merge(files), road_category=road_category, client=client)
        print(f"Shape of the merged data for road category {road_category}: ", preprocessor.shape)

        if target == target_data["V"]:
            return split_data(preprocessor.preprocess_volumes(), target=target, mode=0)
        elif target == target_data["AS"]:
            return split_data(preprocessor.preprocess_speeds(), target=target, mode=0)
        else:
            raise TargetVariableNotFoundError("Wrong target variable imputed")


    def execute_gridsearch(X_train: dd.DataFrame, y_train: dd.DataFrame, learner: callable) -> None:

        gridsearch_result = learner.gridsearch(X_train, y_train)
        learner.export_gridsearch_results(gridsearch_result)
        print(f"============== {learner.get_model().name} grid search results ==============\n")
        print(gridsearch_result, "\n")

        return None


    def execute_training(X_train: dd.DataFrame, y_train: dd.DataFrame, learner: callable):
        learner.get_model().fit(X_train, y_train).export()
        print("Fitting phase ended")
        return


    def execute_testing(X_test: dd.DataFrame, y_test: dd.DataFrame, learner: callable):
        model = learner.get_model()
        y_pred = model.predict(X_test)
        print(model.evaluate_regression(y_test=y_test, y_pred=y_pred, scorer=learner.get_scorer()))
        return


    def process_functionality(target: str, function: callable) -> None:
        for road_category, files in get_trp_ids_by_road_category(target=target).items():
            X_train, X_test, y_train, y_test = preprocess_data(files=files, target=target_data[target], road_category=road_category)
            for model in models:
                learner = TFSLearner(model=model, road_category=road_category, target=cast(Literal["traffic_volumes", "average_speed"], target), client=client)
                function(X_train if function.__name__ in ["execute_gridsearch", "execute_training"] else X_test,
                         y_train if function.__name__ in ["execute_gridsearch", "execute_training"] else y_test,
                         learner)

    with dask_cluster_client(processes=False) as client:
        functionality_mapping = {
            "3.2.1": ("V", execute_gridsearch),
            "3.2.2": ("AS", execute_gridsearch),
            "3.2.3": ("V", execute_training),
            "3.2.4": ("AS", execute_training),
            "3.2.5": ("V", execute_testing),
            "3.2.6": ("AS", execute_testing)
        }

        if functionality in functionality_mapping:
            target, operation = functionality_mapping[functionality]
            process_functionality(target, operation)
        else:
            raise ValueError(f"Unknown functionality: {functionality}")

        print("Alive Dask cluster workers: ", dask.distributed.worker.Worker._instances)
        time.sleep(1)  # To cool down the system

    return None


def execute_forecasting(functionality: str) -> None:
    check_metainfo_file()

    print("Which kind of data would you like to forecast?")
    print("V: Volumes | AS: Average Speeds")
    option = input("Choice: ").upper()

    if option not in ["V", "AS"]:
        print("Invalid option, returning to main menu")
        return

    if functionality == "3.3.1":

        with dask_cluster_client(processes=False) as client:

            trp_ids = get_trp_ids()
            print("TRP IDs: ", trp_ids)
            trp_id = input("Insert TRP ID for forecasting: ")

            if trp_id not in trp_ids:
                raise TRPNotFoundError("TRP ID not in available TRP IDs list")

            trp_metadata = get_trp_metadata(trp_id)

            if not trp_metadata["checks"]["has_volumes" if option == "V" else "has_speeds"]:
                raise TargetDataNotAvailableError(f"Target data not available for TRP: {trp_id}")

            trp_road_category = trp_metadata["trp_data"]["location"]["roadReference"]["roadCategory"]["id"]
            print("\nTRP road category: ", trp_road_category)

            forecaster = OnePointForecaster(trp_id=trp_id, road_category=trp_road_category, target=target_data[option], client=client)
            future_records = forecaster.get_future_records(target_datetime=read_forecasting_target_datetime(target=target_data)) #Already preprocessed

            #TODO TEST training_mode = BOTH 0 AND 1
            model_training_dataset = forecaster.get_training_records(training_mode=0, road_category=trp_road_category, limit=future_records.shape[0].compute() * 24)
            X, y = split_data(model_training_dataset, target=target_data, mode=1)

            for name, model in model_definitions["class_instances"].items():

                with open(get_models_parameters_folder_path(target_data[option], trp_road_category) + get_active_ops() + "_" + trp_road_category + "_" + name + "_parameters.json", "r") as params_reader:
                    best_params = json.load(params_reader)[name] # Attributes which aren't included in the gridsearch grid are already included in best_params since they were first gathered together and then exported

                learner = TFSLearner(model=model(**best_params), road_category=trp_road_category, target=target_data[option], client=client)
                model = learner.get_model().fit(X, y)
                predictions = model.predict(future_records)
                print(predictions)

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
        "3.2.6": execute_forecast_warmup,
        "3.3.1": execute_forecasting,
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
        3.2.6 Test models on average speed data
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
