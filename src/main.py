import sys
import time
import asyncio
from typing import cast
import pandas as pd
import dask
import dask.dataframe as dd

from exceptions import TRPNotFoundError, TargetVariableNotFoundError, TargetDataNotAvailableError

from db_config import DBConfig
from brokers import AIODBManagerBroker, AIODBBroker, DBBroker
from downloader import volumes_to_db
from tfs_eda import analyze_volume, volume_multicollinearity_test, analyze_mean_speed, mean_speed_multicollinearity_test
from ml import TFSLearner, TFSPreprocessor, OnePointForecaster
from road_network import *
from utils import GlobalDefinitions, dask_cluster_client, GeneralPurposeToolbox, ForecastingToolbox



async def get_aiodbmanager_broker():
    return AIODBManagerBroker(superuser=DBConfig.SUPERUSER.value,
                                superuser_password=DBConfig.SUPERUSER_PASSWORD.value,
                                tfs_user=DBConfig.TFS_USER.value,
                                tfs_password=DBConfig.TFS_PASSWORD.value,
                                hub_db=DBConfig.HUB_DB.value,
                                maintenance_db=DBConfig.MAINTENANCE_DB.value
    )


async def get_aiodb_broker():
    return AIODBBroker(db_user=DBConfig.TFS_USER.value,
                       db_password=DBConfig.TFS_PASSWORD.value,
                       db_name=await (await get_aiodbmanager_broker()).get_current_project(),
                       db_host=DBConfig.DB_HOST.value
    )


def get_db_broker():
    aiodbmanager_broker = asyncio.run(get_aiodbmanager_broker())
    return DBBroker(db_user=DBConfig.TFS_USER.value,
                    db_password=DBConfig.TFS_PASSWORD.value,
                    db_name=asyncio.run(aiodbmanager_broker.get_current_project()),
                    db_host=DBConfig.DB_HOST.value
    )


async def initialize() -> None:
    await (await get_aiodbmanager_broker()).init()
    return None


def get_gp_toolbox():
    return GeneralPurposeToolbox()



async def manage_global(functionality: str) -> None:
    db_manager_broker_async = await get_aiodbmanager_broker()
    if functionality == "1.1":
        await db_manager_broker_async.create_project(name=await asyncio.to_thread(input,"Insert new project name: "), lang="en", auto_project_setup=True)

    elif functionality == "1.2":
        await db_manager_broker_async.set_current_project(await asyncio.to_thread(input, "Insert the project to set as current: "))

    elif functionality == "1.3":
        print("Current project: ", await db_manager_broker_async.get_current_project(), "\n\n")

    elif functionality == "1.4":
        await db_manager_broker_async.reset_current_project(await asyncio.to_thread(input, "Insert the project to reset: "))

    elif functionality == "1.5":
        await db_manager_broker_async.delete_project(await asyncio.to_thread(input,"Insert the name of the project to delete: "))

    else:
        print("\033[91mFunctionality not found, try again with a correct one\033[0m")
        print("\033[91mReturning to the main menu...\033[0m\n\n")
        main()

    return None


async def download_volumes(functionality: str) -> None:
    if functionality == "2.1":
        print("\nDownloading traffic registration points information for the active operation...")
        #TODO CREATE A TRPS_TO_DB FUNCTION IN downloader.py
        print("Traffic registration points information downloaded successfully\n\n")

    elif functionality == "2.2":
        time_start = input("Insert starting datetime (of the time frame which you're interested in) - YYYY-MM-DDTHH: ") + ":00:00.000" + GlobalDefinitions.NORWEGIAN_UTC_TIME_ZONE.value
        time_end = input("Insert ending datetime (of the time frame which you're interested in) - YYYY-MM-DDTHH: ") + ":00:00.000" + GlobalDefinitions.NORWEGIAN_UTC_TIME_ZONE.value
        print("Downloading traffic volumes data for every registration point for the current project...")

        #TODO USE download_volumes()

    return None


#TODO EXECUTE PIPELINES



async def manage_forecasting_horizon(functionality: str) -> None:
    ft = ForecastingToolbox(db_broker_async=await get_aiodb_broker())

    if functionality == "3.1.1":
        print("-- Forecasting horizon setter --")
        await ft.set_forecasting_horizon()

    elif functionality == "3.1.2":
        print("-- Forecasting horizon reader --")
        target = asyncio.to_thread(input, "V: Volumes | MS: Mean Speed")
        print("Target datetime: ", await ft.get_forecasting_horizon(target=target), "\n\n")

    elif functionality == "3.1.3":
        print("-- Forecasting horizon reset --")
        target = asyncio.to_thread(input, "V: Volumes | MS: Mean Speed")
        await ft.reset_forecasting_horizon(target=target)

    return None


def execute_eda() -> None:
    trp_data = trp_toolbox.trps_data()

    for v in (trp_id for trp_id in trp_data.keys() if ....get_trp_metadata(trp_id=trp_id)["checks"].get(GlobalDefinitions.HAS_VOLUME_CHECK.value)):
        volumes = ...
        analyze_volume(volumes)
        volume_multicollinearity_test(volumes)

    for s in (trp_id for trp_id in trp_data.keys() if ....get_trp_metadata(trp_id=trp_id)["checks"].get(GlobalDefinitions.HAS_MEAN_SPEED_CHECK.value)):
        speeds = ...
        analyze_mean_speed(speeds)
        mean_speed_multicollinearity_test(speeds)

    volumes_speeds = (vs for vs in (trp_id for trp_id in trp_data.keys() if trp_data[trp_id]["checks"]["has_" + GlobalDefinitions.TARGET_DATA.value["V"]] and trp_data[trp_id]["checks"]["has_" + GlobalDefinitions.TARGET_DATA.value["MS"]]))
    # Determining the TRPs which have both traffic volumes and speed data

    print("\n\n")

    return None


# TODO IN THE FUTURE WE COULD PREDICT percentile_85 AS WELL. EXPLICITELY PRINT THAT FILES METADATA IS NEEDED BEFORE EXECUTING THE WARMUP
def execute_forecast_warmup(functionality: str) -> None:
    models = ... #TODO GET BINARY OBJECTS FROM DB
    gp_toolbox = get_gp_toolbox()
    db_broker = get_db_broker()

    def preprocess(road_category: str, target: str) -> tuple[dd.DataFrame, dd.DataFrame, dd.DataFrame, dd.DataFrame]:

        print(f"\n********************* Executing data preprocessing for road category: {road_category} *********************\n")

        preprocessor = TFSPreprocessor(data=..., road_category=road_category, client=client, gp_toolbox=gp_toolbox) #TODO MERGE ISN'T EVEN NEEDED, JUST SELECT ALL DATA WHERE ROAD CATEGORY = ... AND TRP_ID IS WITHIN A LIST OF TRP_IDS
        print(f"Shape of the merged data for road category {road_category}: ", preprocessor.shape)

        if target == GlobalDefinitions.TARGET_DATA.value["V"]:
            return gp_toolbox.split_data(preprocessor.preprocess_volumes(), target=target, mode=0)
        elif target == GlobalDefinitions.TARGET_DATA.value["MS"]:
            return gp_toolbox.split_data(preprocessor.preprocess_speeds(), target=target, mode=0)
        else:
            raise TargetVariableNotFoundError("Wrong target variable imputed")


    def ml_gridsearch(X_train: dd.DataFrame, y_train: dd.DataFrame, learner: callable) -> None:

        gridsearch_result = learner.gridsearch(X_train, y_train)
        learner.export_gridsearch_results(gridsearch_result)
        print(f"============== {learner.get_model().name} grid search results ==============\n")
        print(gridsearch_result, "\n")

        return None


    def ml_training(X_train: dd.DataFrame, y_train: dd.DataFrame, learner: callable) -> None:
        learner.get_model().fit(X_train, y_train).export()
        print("Fitting phase ended")
        return None


    def ml_testing(X_test: dd.DataFrame, y_test: dd.DataFrame, learner: callable) -> None:
        model = learner.get_model()
        y_pred = model.predict(X_test)
        print(model.evaluate_regression(y_test=y_test, y_pred=y_pred, scorer=learner.get_scorer()))
        return None


    def process_functionality(target: str, function: callable) -> None:
        function_name = function.__name__

        for road_category, trp_ids in db_broker.get_trp_ids_by_road_category().items():
            #TODO GET ALL DATA WHERE trp_id IS IN trp_id
            X_train, X_test, y_train, y_test = preprocess(dfs=..., target=target, road_category=road_category)

            for model in models:
                if function_name != "execute_gridsearch":
                    #TODO READ MODEL PARAMS
                    params = json.load(params_reader)[model.__name__]
                else:
                    params = model_definitions["auxiliary_parameters"].get(model.__name__, {})

                learner = TFSLearner(model=model(**params), road_category=road_category, target=cast(Literal["V", "MS"], target), client=client, db_broker=db_broker, gp_toolbox=gp_toolbox)
                function(X_train if function_name in ["execute_gridsearch", "execute_training"] else X_test,
                         y_train if function_name in ["execute_gridsearch", "execute_training"] else y_test,
                         learner)

        return None

    with dask_cluster_client(processes=False) as client:
        functionality_mapping = {
            "3.2.1": (GlobalDefinitions.TARGET_DATA.value["V"], ml_gridsearch),
            "3.2.2": (GlobalDefinitions.TARGET_DATA.value["MS"], ml_gridsearch),
            "3.2.3": (GlobalDefinitions.TARGET_DATA.value["V"], ml_training),
            "3.2.4": (GlobalDefinitions.TARGET_DATA.value["MS"], ml_training),
            "3.2.5": (GlobalDefinitions.TARGET_DATA.value["V"], ml_testing),
            "3.2.6": (GlobalDefinitions.TARGET_DATA.value["MS"], ml_testing)
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

    print("Enter target data to forecast: ")
    print("V: Volumes | MS: Mean Speed")
    option = input("Choice: ").upper()

    if not gp_toolbox.check_target(option):
        raise TargetDataNotAvailableError("Invalid target variable")

    if functionality == "3.3.1":

        with dask_cluster_client(processes=False) as client:
            trp_ids = trp_toolbox.get_trp_ids_async()
            print("TRP IDs: ", trp_ids)
            trp_id = input("Insert TRP ID for forecasting: ")

            if trp_id not in trp_ids:
                raise TRPNotFoundError("TRP ID not in available TRP IDs list")

            trp_metadata = ....get_trp_metadata(trp_id)

            if not trp_metadata["checks"][GlobalDefinitions.HAS_VOLUME_CHECK.value if option == GlobalDefinitions.TARGET_DATA.value["V"] else GlobalDefinitions.HAS_MEAN_SPEED_CHECK.value]:
                raise TargetDataNotAvailableError(f"Target data not available for TRP: {trp_id}")

            trp_road_category = trp_metadata["trp_data"]["location"]["roadReference"]["roadCategory"]["id"]
            print("\nTRP road category: ", trp_road_category)

            forecaster = OnePointForecaster(trp_id=trp_id, road_category=trp_road_category, target=GlobalDefinitions.TARGET_DATA.value[option], client=client)
            future_records = forecaster.get_future_records(
                forecasting_horizon=forecasting_toolbox.get_forecasting_horizon(
                    target=GlobalDefinitions.TARGET_DATA.value[option]))  #Already preprocessed

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
        "1.1": manage_global,
        "1.2": manage_global,
        "1.3": manage_global,
        "1.4": manage_global,
        "1.5": manage_global,
        "2.1": download_volumes,
        "2.2": download_volumes,
        "3.1.1": manage_forecasting_horizon,
        "3.1.2": manage_forecasting_horizon,
        "3.1.3": manage_forecasting_horizon,
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
        "5.2": execute_eda
    }

    while True:
        print("""==================== MENU ====================
1. Set pre-analysis information
    1.1 Create a new project
    1.2 Set a project as the current one
    1.3 Check the current project's name
    1.4 Reset current project
    1.5 Delete a project
2. Download data (Trafikkdata API)
    2.1 Traffic registration points information
    2.2 Traffic volumes for every registration point
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
0. Exit""")

        asyncio.run(initialize())

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
