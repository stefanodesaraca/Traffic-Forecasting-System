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

#TODO GET MORE SPECIFIC WITH IMPORTS AND REPLACE * WITH ACTUAL METHODS, FUNCTIONS, ETC. USED HERE IN THE MAIN CODE
from tfs_downloader import *
from tfs_eda import *
from tfs_cleaning import *
from tfs_ml import *
from tfs_road_network import *
from tfs_ml_configs import *
from tfs_utils import *
from tfs_base_config import gp_toolbox, pjh, pjhmm, pmm, tmm, trp_toolbox, forecasting_toolbox


def manage_ops(functionality: str) -> None:
    if functionality == "1.1":
        pjhmm.create(gp_toolbox.clean_text(input("Insert new project name: ")))

    elif functionality == "1.2":
        pjhmm.set_current_project(gp_toolbox.clean_text(input("Insert the operation to set as active: ")))

    elif functionality == "1.3":
        print("Current project: ", pjhmm.get_current_project(), "\n\n")

    elif functionality == "1.4":
        pjhmm.reset_current_project()

    elif functionality == "1.5":
        pjhmm.delete(gp_toolbox.clean_text(input("Insert the name of the project to delete: ")))

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
            raise Exception(f"Couldn't download traffic registration points information for the active operation. Error: {e}")

    elif functionality == "2.2":
        time_start = input("Insert starting datetime (of the time frame which you're interested in) - YYYY-MM-DDTHH: ")
        time_end = input("Insert ending datetime (of the time frame which you're interested in) - YYYY-MM-DDTHH: ")

        if not gp_toolbox.check_datetime_format(time_start) and not gp_toolbox.check_datetime_format(time_end):
            print("\033[91mWrong datetime format, try again with a correct one\033[0m")
            print("Returning to the main menu...\n\n")
            main()

        time_start += ":00:00.000Z"
        time_end += ":00:00.000Z"

        await pmm.set_async(value=time_start, key=GlobalDefinitions.VOLUME.value + "start_date_iso", mode="e")
        await pmm.set_async(value=time_end, key=GlobalDefinitions.MEAN_SPEED.value + "end_date_iso", mode="e")

        relative_delta = relativedelta(datetime.datetime.strptime(time_end, GlobalDefinitions.DT_ISO.value).date(), datetime.datetime.strptime(time_start, GlobalDefinitions.DT_ISO.value).date())
        days_delta = (datetime.datetime.strptime(time_end, GlobalDefinitions.DT_ISO.value).date() - datetime.datetime.strptime(time_start, GlobalDefinitions.DT_ISO.value).date()).days
        years_delta = relative_delta.years or 0
        months_delta = relative_delta.months + (years_delta * 12)
        weeks_delta = days_delta // 7

        await pmm.set_async(value=days_delta, key=GlobalDefinitions.VOLUME.value + ".n_days", mode="e")
        await pmm.set_async(value=months_delta, key=GlobalDefinitions.VOLUME.value + ".n_months", mode="e")
        await pmm.set_async(value=years_delta, key=GlobalDefinitions.VOLUME.value + ".n_years", mode="e") #TODO THIS CREATES A SECOND n_years. WHY DOESN'T IT OVERWRITE .OLD n_years: null?
        await pmm.set_async(value=weeks_delta, key=GlobalDefinitions.VOLUME.value + ".n_weeks", mode="e")

        print("Downloading traffic volumes data for every registration point for the active operation...")
        await traffic_volumes_data_to_json(time_start=time_start, time_end=time_end)

    elif functionality == "2.3":
        if len(os.listdir(pjhmm.get(key="folder_paths.data.trp_metadata.path"))) == 0:
            for trp_id in tqdm(trp_toolbox.get_global_trp_data().keys()):
                tmm.write_trp_metadata(trp_id, **{"trp_data": trp_toolbox.get_global_trp_data()[trp_id]})
        else:
            print("Metadata had already been computed.")

    return None


async def clean_data(functionality: str) -> None:
    semaphore = Semaphore(50)

    async def limited_clean(trp_id: str, cleaner: TrafficVolumesCleaner | AverageSpeedCleaner, export: bool = True):
        async with semaphore:
            await cleaner.clean_async(trp_id, export=export)

    if functionality == "5.6.1":
        await asyncio.gather(*(limited_clean(trp_id=trp_id, cleaner=TrafficVolumesCleaner(), export=True) for trp_id in trp_toolbox.get_trp_ids())) # The star (*) in necessary since gather() requires the coroutines to fed as positional arguments of the function. So we can unpack the list with *
    elif functionality == "5.6.2":
        await asyncio.gather(*(limited_clean(trp_id=trp_id, cleaner=AverageSpeedCleaner(), export=True) for trp_id in trp_toolbox.get_trp_ids()))

    return None


def set_forecasting_options(functionality: str) -> None:
    if functionality == "3.1.1":
        print("-- Forecasting horizon setter --")
        forecasting_toolbox.set_forecasting_horizon()

    elif functionality == "3.1.2":
        print("-- Forecasting horizon reader --")
        target = input("V = Volumes | MS = Mean Speed")
        print("Target datetime: ", forecasting_toolbox.get_forecasting_target_datetime(target=target), "\n\n")

    elif functionality == "3.1.3":
        print("-- Forecasting horizon reset --")
        target = input("V = Volumes | MS = Mean Speed")
        forecasting_toolbox.reset_forecasting_target_datetime(target=target)

    return None


def execute_eda() -> None:
    trp_data = trp_toolbox.get_global_trp_data()
    clean_volumes_folder = pmm.get(key="folder_paths.data." + GlobalDefinitions.VOLUME.value + ".subfolders.clean.path")
    clean_speeds_folder = pmm.get(key="folder_paths.data." + GlobalDefinitions.MEAN_SPEED.value + ".subfolders.clean.path")

    for v in (trp_id for trp_id in trp_data.keys() if tmm.get_trp_metadata(trp_id=trp_id)["checks"].get(GlobalDefinitions.HAS_VOLUME_CHECK.value)):
        volumes = pd.read_csv(clean_volumes_folder + v + GlobalDefinitions.CLEAN_VOLUME_FILENAME_ENDING.value + ".csv")
        analyze_volume(volumes)
        volume_multicollinearity_test(volumes)

    for s in (trp_id for trp_id in trp_data.keys() if tmm.get_trp_metadata(trp_id=trp_id)["checks"].get(GlobalDefinitions.HAS_MEAN_SPEED_CHECK.value)):
        speeds = pd.read_csv(clean_speeds_folder + s + GlobalDefinitions.CLEAN_MEAN_SPEED_FILENAME_ENDING.value + ".csv")
        analyze_mean_speed(speeds)
        mean_speed_multicollinearity_test(speeds)

    volumes_speeds = (vs for vs in (trp_id for trp_id in trp_data.keys() if trp_data[trp_id]["checks"]["has_" + GlobalDefinitions.TARGET_DATA.value["V"]] and trp_data[trp_id]["checks"]["has_" + GlobalDefinitions.TARGET_DATA.value["MS"]]))
    # Determining the TRPs which have both traffic volumes and speed data

    print("\n\n")

    return None


# TODO IN THE FUTURE WE COULD PREDICT percentile_85 AS WELL. EXPLICITELY PRINT THAT FILES METADATA IS NEEDED BEFORE EXECUTING THE WARMUP
def execute_forecast_warmup(functionality: str) -> None:
    models = model_definitions["class_instance"].values()


    def preprocess_data(files: list[str], road_category: str, target: str) -> tuple[dd.DataFrame, dd.DataFrame, dd.DataFrame, dd.DataFrame]:

        print(f"\n********************* Executing data preprocessing for road category: {road_category} *********************\n")

        preprocessor = TFSPreprocessor(data=gp_toolbox.merge(files), road_category=road_category, client=client)
        print(f"Shape of the merged data for road category {road_category}: ", preprocessor.shape)

        if target == GlobalDefinitions.TARGET_DATA.value["V"]:
            return gp_toolbox.split_data(preprocessor.preprocess_volumes(), target=target, mode=0)
        elif target == GlobalDefinitions.TARGET_DATA.value["MS"]:
            return gp_toolbox.split_data(preprocessor.preprocess_speeds(), target=target, mode=0)
        else:
            raise TargetVariableNotFoundError("Wrong target variable imputed")


    def execute_gridsearch(X_train: dd.DataFrame, y_train: dd.DataFrame, learner: callable) -> None:

        gridsearch_result = learner.gridsearch(X_train, y_train)
        learner.export_gridsearch_results(gridsearch_result)
        print(f"============== {learner.get_model().name} grid search results ==============\n")
        print(gridsearch_result, "\n")

        return None


    def execute_training(X_train: dd.DataFrame, y_train: dd.DataFrame, learner: callable) -> None:
        learner.get_model().fit(X_train, y_train).export()
        print("Fitting phase ended")
        return None


    def execute_testing(X_test: dd.DataFrame, y_test: dd.DataFrame, learner: callable) -> None:
        model = learner.get_model()
        y_pred = model.predict(X_test)
        print(model.evaluate_regression(y_test=y_test, y_pred=y_pred, scorer=learner.get_scorer()))
        return None


    def process_functionality(target: str, function: callable) -> None:
        function_name = function.__name__

        for road_category, files in trp_toolbox.get_trp_ids_by_road_category(target=target).items():
            X_train, X_test, y_train, y_test = preprocess_data(files=files, target=target, road_category=road_category)

            for model in models:
                if function_name != "execute_gridsearch":
                    with open(pmm.get(key="folder_paths.ml.models_parameters.subfolders." + target + ".subfolders." + road_category + ".path") + pjhmm.get_current_project() + "_" + road_category + "_" + model.__name__ + "_parameters.json", "r", encoding="utf-8") as params_reader:
                        params = json.load(params_reader)[model.__name__]
                else:
                    params = model_definitions["auxiliary_parameters"].get(model.__name__, {})

                learner = TFSLearner(model=model(**params), road_category=road_category, target=cast(Literal["V", "MS"], target), client=client)
                function(X_train if function_name in ["execute_gridsearch", "execute_training"] else X_test,
                         y_train if function_name in ["execute_gridsearch", "execute_training"] else y_test,
                         learner)

        return None

    with dask_cluster_client(processes=False) as client:
        functionality_mapping = {
            "3.2.1": (GlobalDefinitions.TARGET_DATA.value["V"], execute_gridsearch),
            "3.2.2": (GlobalDefinitions.TARGET_DATA.value["MS"], execute_gridsearch),
            "3.2.3": (GlobalDefinitions.TARGET_DATA.value["V"], execute_training),
            "3.2.4": (GlobalDefinitions.TARGET_DATA.value["MS"], execute_training),
            "3.2.5": (GlobalDefinitions.TARGET_DATA.value["V"], execute_testing),
            "3.2.6": (GlobalDefinitions.TARGET_DATA.value["MS"], execute_testing)
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

    print("Which kind of data would you like to forecast?")
    print("V: Volumes | MS: Mean Speed")
    option = input("Choice: ").upper()

    if not gp_toolbox.check_target(option):
        raise TargetDataNotAvailableError("Invalid target variable")

    if functionality == "3.3.1":

        with dask_cluster_client(processes=False) as client:
            trp_ids = trp_toolbox.get_trp_ids()
            print("TRP IDs: ", trp_ids)
            trp_id = input("Insert TRP ID for forecasting: ")

            if trp_id not in trp_ids:
                raise TRPNotFoundError("TRP ID not in available TRP IDs list")

            trp_metadata = tmm.get_trp_metadata(trp_id)

            if not trp_metadata["checks"][GlobalDefinitions.HAS_VOLUME_CHECK.value if option == GlobalDefinitions.TARGET_DATA.value["V"] else GlobalDefinitions.HAS_MEAN_SPEED_CHECK.value]:
                raise TargetDataNotAvailableError(f"Target data not available for TRP: {trp_id}")

            trp_road_category = trp_metadata["trp_data"]["location"]["roadReference"]["roadCategory"]["id"]
            print("\nTRP road category: ", trp_road_category)

            forecaster = OnePointForecaster(trp_id=trp_id, road_category=trp_road_category, target=GlobalDefinitions.TARGET_DATA.value[option], client=client)
            future_records = forecaster.get_future_records(target_datetime=forecasting_toolbox.get_forecasting_target_datetime(target=GlobalDefinitions.TARGET_DATA.value[option])) #Already preprocessed

            #TODO TEST training_mode = BOTH 0 AND 1
            model_training_dataset = forecaster.get_training_records(training_mode=0, limit=future_records.shape[0].compute() * 24)
            X, y = gp_toolbox.split_data(model_training_dataset, target=GlobalDefinitions.TARGET_DATA.value[option], mode=1)

            for name, model in model_definitions["class_instances"].items():

                with open(pmm.get(key="folder_paths.ml.models_parameters.subfolders." + GlobalDefinitions.TARGET_DATA.value[option] + ".subfolders." + trp_road_category + ".path") + pjhmm.get_current_project() + "_" + trp_road_category + "_" + name + "_parameters.json", "r") as params_reader:
                    best_params = json.load(params_reader)[name] # Attributes which aren't included in the gridsearch grid are already included in best_params since they were first gathered together and then exported

                learner = TFSLearner(model=model(**best_params), road_category=trp_road_category, target=GlobalDefinitions.TARGET_DATA.value[option], client=client)
                model = learner.get_model().fit(X, y)
                predictions = model.predict(future_records)
                print(predictions)

    return None


def manage_road_network(functionality: str) -> None:
    if functionality == "4.1":
        ...

    elif functionality == "4.2":
        ...

    return None


def main():
    menu_options = {
        "1.1": manage_ops,
        "1.2": manage_ops,
        "1.3": manage_ops,
        "1.4": manage_ops,
        "1.5": manage_ops,
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
    1.4 Reset active operation
    1.5 Delete an operation
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
