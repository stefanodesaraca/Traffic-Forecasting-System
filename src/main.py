import os
import sys
import time
import pickle
import asyncio
import pandas as pd
import dask
import dask.dataframe as dd
import geojson

from exceptions import TRPNotFoundError, TargetDataNotAvailableError, ModelBestParametersNotFound
from db_config import DBConfig

from downloader import start_client_async, volumes_to_db, fetch_trps
from brokers import AIODBManagerBroker, AIODBBroker, DBBroker
from pipelines import MeanSpeedExtractionPipeline
from loaders import BatchStreamLoader
from ml import TFSLearner, TFSPreprocessor, OnePointForecaster
from road_network import *
from utils import GlobalDefinitions, dask_cluster_client, ForecastingToolbox, check_target, split_by_target

from tfs_eda import analyze_volume, volume_multicollinearity_test, analyze_mean_speed, mean_speed_multicollinearity_test


async def get_aiodbmanager_broker():
    return AIODBManagerBroker(
        superuser=DBConfig.SUPERUSER.value,
        superuser_password=DBConfig.SUPERUSER_PASSWORD.value,
        tfs_user=DBConfig.TFS_USER.value,
        tfs_password=DBConfig.TFS_PASSWORD.value,
        tfs_role=DBConfig.TFS_ROLE.value,
        tfs_role_password=DBConfig.TFS_ROLE_PASSWORD.value,
        hub_db=DBConfig.HUB_DB.value,
        maintenance_db=DBConfig.MAINTENANCE_DB.value,
        db_host=DBConfig.DB_HOST.value
    )


async def get_aiodb_broker():
    return AIODBBroker(
       db_user=DBConfig.TFS_USER.value,
       db_password=DBConfig.TFS_PASSWORD.value,
       db_name=(await (await get_aiodbmanager_broker()).get_current_project()).get("name", None),
       db_host=DBConfig.DB_HOST.value
    )


def get_db_broker():
    aiodbmanager_broker = asyncio.run(get_aiodbmanager_broker())
    return DBBroker(
        db_user=DBConfig.TFS_USER.value,
        db_password=DBConfig.TFS_PASSWORD.value,
        db_name=asyncio.run(aiodbmanager_broker.get_current_project()).get("name", None),
        db_host=DBConfig.DB_HOST.value
    )


async def initialize() -> None:
    os.makedirs(GlobalDefinitions.MEAN_SPEED_DIR.value, exist_ok=True) #The directory where mean speed files need to be placed
    await (await get_aiodbmanager_broker()).init()
    return None


async def manage_global(functionality: str) -> None:
    db_manager_broker_async = await get_aiodbmanager_broker()
    if functionality == "1.1":
        await db_manager_broker_async.create_project(name=await asyncio.to_thread(input, "Insert new project name: "), lang="en", auto_project_setup=True)

    elif functionality == "1.2":
        await db_manager_broker_async.set_current_project(
            await asyncio.to_thread(input, "Insert the project to set as current: "))

    elif functionality == "1.3":
        print("Current project: ", await db_manager_broker_async.get_current_project(), "\n\n")

    elif functionality == "1.4":
        await db_manager_broker_async.reset_current_project()

    elif functionality == "1.5":
        await db_manager_broker_async.delete_project(await asyncio.to_thread(input, "Insert the name of the project to delete: "))

    elif functionality == "1.6":
        print(await db_manager_broker_async.list_all_projects())

    else:
        print("\033[91mFunctionality not found, try again with a correct one\033[0m")
        print("\033[91mReturning to the main menu...\033[0m\n\n")
        main()

    return None


async def manage_downloads(functionality: str) -> None:
    if functionality == "2.1":
        print("\nDownloading traffic registration points information for the active operation...")
        await (await get_aiodbmanager_broker()).insert_trps(data=await fetch_trps(
            gql_client=await start_client_async()))
        print("Traffic registration points information downloaded successfully\n\n")

    elif functionality == "2.2":
        time_start = await asyncio.to_thread(input, "Insert starting datetime (of the time frame which you're interested in) - YYYY-MM-DDTHH: ") + ":00:00.000" + GlobalDefinitions.NORWEGIAN_UTC_TIME_ZONE.value
        time_end = await asyncio.to_thread(input, "Insert ending datetime (of the time frame which you're interested in) - YYYY-MM-DDTHH: ") + ":00:00.000" + GlobalDefinitions.NORWEGIAN_UTC_TIME_ZONE.value
        print("Downloading traffic volumes data for every registration point for the current project...")
        await volumes_to_db(
            gql_client=await start_client_async(),
            db_broker_async=await get_aiodb_broker(),
            time_start=time_start,
            time_end=time_end,
            n_async_jobs=5,
            max_retries=5
        )
    return None


async def mean_speeds_to_db() -> None:
    pipeline = MeanSpeedExtractionPipeline(db_broker_async=await get_aiodb_broker())
    all(await pipeline.ingest(fp=GlobalDefinitions.MEAN_SPEED_DIR.value / file, fields=GlobalDefinitions.MEAN_SPEED_INGESTION_FIELDS.value) for file in await asyncio.to_thread(os.listdir,))
    return None

#TODO CHECK HOW BEST PARAMETERS ARE INSERTED INTO DB AND IF THERE'S A DEFAULT IDX


async def manage_forecasting_horizon(functionality: str) -> None:
    ft = ForecastingToolbox(db_broker_async=await get_aiodb_broker())

    if functionality == "3.1.1":
        print("-- Forecasting horizon setter --")
        await ft.set_forecasting_horizon_async()

    elif functionality == "3.1.2":
        print("-- Forecasting horizon reader --")
        target = asyncio.to_thread(input, "V: Volumes | MS: Mean Speed")
        print("Target datetime: ", await ft.get_forecasting_horizon_async(target=target), "\n\n")

    elif functionality == "3.1.3":
        print("-- Forecasting horizon reset --")
        target = asyncio.to_thread(input, "V: Volumes | MS: Mean Speed")
        await ft.reset_forecasting_horizon_async(target=target)

    return None


def execute_eda() -> None:
    db_broker = get_db_broker()
    trps_data = db_broker.get_all_trps_metadata()

    for v in ...: #TODO TRPS WHICH ACTUALLY HAVE VOLUME DATA, CHECK METADATA VIEW
        volumes = ...
        analyze_volume(volumes)
        volume_multicollinearity_test(volumes)

    for s in ...: #TODO TRPS WHICH ACTUALLY HAVE MEAN SPEED DATA, CHECK METADATA VIEW
        speeds = ...
        analyze_mean_speed(speeds)
        mean_speed_multicollinearity_test(speeds)

    volumes_speeds = ... #TODO TRPS WHICH ACTUALLY HAVE BOTH VOLUME AND MEAN SPEED DATA
    # Determining the TRPs which have both traffic volumes and speed data

    print("\n\n")

    return None


# NOTE IN THE FUTURE WE COULD PREDICT percentile_85 AS WELL
def forecasts_warmup(functionality: str) -> None:
    db_broker = get_db_broker()
    loader = BatchStreamLoader(db_broker=db_broker)
    models = {m["name"]: {"binary_obj": m["binary_obj"],
                          "base_parameters": m["base_parameters"],
                          "volume_best_parameters": m["volume_best_parameters"],
                          "mean_speed_best_parameters": m["mean_speed_best_parameters"]} for m in db_broker.send_sql("""SELECT
                                                                                                                            m.name,
                                                                                                                            mo.pickle_object AS binary_obj,
                                                                                                                            m.base_params AS base_parameters,
                                                                                                                            m.volume_best_params AS volume_best_parameters
                                                                                                                            m.mean_speed_best_params AS mean_speed_best_parameters
                                                                                                                        FROM
                                                                                                                            MLModels m
                                                                                                                        JOIN
                                                                                                                            MLModelObjects mo ON m.id = mo.id;""")}
    actual_target: str | None = None


    def ml_gridsearch(X_train: dd.DataFrame, y_train: dd.DataFrame, learner: callable) -> None:
        gridsearch_result = learner.gridsearch(X_train, y_train)
        learner.export_gridsearch_results(gridsearch_result)
        print(f"============== {learner.model.name} grid search results ==============\n")
        print(gridsearch_result, "\n")
        return None


    def ml_training(X_train: dd.DataFrame, y_train: dd.DataFrame, learner: callable) -> None:
        learner.model.fit(X_train, y_train).export()
        print("Fitting phase ended")
        return None


    def ml_testing(X_test: dd.DataFrame, y_test: dd.DataFrame, learner: callable) -> None:
        model = learner.model
        y_pred = model.predict(X_test)
        print(model.evaluate_regression(y_test=y_test, y_pred=y_pred, scorer=learner.scorer))
        return None


    def process_functionality(func: callable) -> None:
        function_name = func.__name__

        for road_category, trp_ids in db_broker.get_trp_ids_by_road_category().items():

            print(f"\n********************* Executing data preprocessing for road category: {road_category} *********************\n")

            X_train, X_test, y_train, y_test = split_by_target(
                data=getattr(preprocessor := TFSPreprocessor(
                    data=getattr(loader, functionality_mapping[functionality]["loading_method"])(batch_size=5000, trp_list_filter=trp_ids, road_category_filter=road_category),
                    road_category=road_category,
                    client=client
                ), functionality_mapping[functionality]["preprocessing_method"]),
                target=actual_target,
                mode=0
            )
            print(f"Shape of the merged data for road category {road_category}: ", preprocessor.shape)

            for model, metadata in models:
                params = models[model][f"{actual_target}_best_params"] if function_name != "ml_gridsearch" else models[model]["base_params"]

                learner = TFSLearner(model=model(**params), road_category=road_category, target=actual_target, client=client, db_broker=db_broker)
                func(X_train if function_name in ["ml_gridsearch", "ml_training"] else X_test,
                     y_train if function_name in ["ml_gridsearch", "ml_training"] else y_test,
                     learner)

        return None

    with dask_cluster_client(processes=False) as client:
        functionality_mapping = {
            "3.2.1": {
                "func": ml_gridsearch,
                "target": "3.2.1",
                "preprocessing_method": "preprocess_volume",
                "loading_method": "get_volume"
            },
            "3.2.2": {
                "func": ml_gridsearch,
                "target": "3.2.2",
                "preprocessing_method": "preprocess_mean_speed",
                "loading_method": "get_mean_speed"
            },
            "3.2.3": {
                "func": ml_training,
                "target": "3.2.3",
                "preprocessing_method": "preprocess_volume",
                "loading_method": "get_volume"
            },
            "3.2.4": {
                "func": ml_training,
                "target": "3.2.4",
                "preprocessing_method": "preprocess_mean_speed",
                "loading_method": "get_mean_speed"
            },
            "3.2.5": {
                "func": ml_testing,
                "target": "3.2.5",
                "preprocessing_method": "preprocess_volume",
                "loading_method": "get_volume"
            },
            "3.2.6": {
                "func": ml_testing,
                "target": "3.2.6",
                "preprocessing_method": "preprocess_mean_speed",
                "loading_method": "get_mean_speed"
            }
        }

        if functionality not in functionality_mapping:
            raise ValueError(f"Unknown functionality: {functionality}")

        actual_target = functionality_mapping[functionality]["target"]
        process_functionality(functionality_mapping[functionality]["func"]) #Process the chosen operation

        print("Alive Dask cluster workers: ", dask.distributed.worker.Worker._instances)
        time.sleep(1)  # To cool down the system

    return None


def execute_forecasting(functionality: str) -> None:
    db_broker = get_db_broker()
    loader = BatchStreamLoader(db_broker=db_broker)
    ft = ForecastingToolbox(db_broker=db_broker)

    print("Enter target data to forecast: ")
    print("V: Volumes | MS: Mean Speed")
    option = input("Choice: ").upper()

    if not check_target(option):
        raise TargetDataNotAvailableError("Invalid target variable")

    if functionality == "3.3.1":

        with dask_cluster_client(processes=False) as client:
            trp_ids = db_broker.get_trp_ids()
            print("TRP IDs: ", trp_ids)
            trp_id = input("Insert TRP ID for forecasting: ")

            if trp_id not in trp_ids:
                raise TRPNotFoundError("TRP ID not in available TRP IDs list")

            trp_metadata = db_broker.get_trp_metadata(trp_id)
            trp_road_category = trp_metadata["road_category"]

            print("\nTRP road category: ", trp_road_category)

            forecaster = OnePointForecaster(
                trp_id=trp_id,
                road_category=trp_road_category,
                target=GlobalDefinitions.TARGET_DATA.value[option],
                client=client,
                db_broker=db_broker,
                loader=loader
            )
            future_records = forecaster.get_future_records(forecasting_horizon=ft.get_forecasting_horizon(target=GlobalDefinitions.TARGET_DATA.value[option]))  #Already preprocessed

            #TODO TEST training_mode = BOTH 0 AND 1
            model_training_dataset = forecaster.get_training_records(
                training_mode=0,
                limit=future_records.shape[0].compute() * 24
            )
            X, y = split_by_target(
                data=model_training_dataset,
                target=GlobalDefinitions.TARGET_DATA.value[option],
                mode=1
            )

            for name, data in db_broker.get_model_objects()["model_data"].items(): #Load model name and data (pickle object, best parameters and so on)

                model = pickle.load(data[name]["pickle_object"])
                best_params = data[name][f"{GlobalDefinitions.TARGET_DATA.value[option]}_best_params"]

                if best_params is None:
                    raise ModelBestParametersNotFound("Model's best parameters are None, check if the model has been trained or has best parameters set")

                learner = TFSLearner(
                    model=model(**best_params),
                    road_category=trp_road_category,
                    target=GlobalDefinitions.TARGET_DATA.value[option],
                    client=client,
                    db_broker=db_broker
                )
                data = learner.model.fit(X, y)
                predictions = data.predict(future_records)
                print(predictions)

    return None


def manage_road_network(functionality: str) -> None:

    def retrieve_edges(self) -> dict:
        with open(f"{self.get('folder_paths.rn_graph.edges.path')}/traffic-nodes-2024_2025-02-28.geojson", "r", encoding="utf-8") as e:
            return geojson.load(e)["features"]


    def retrieve_arches(self) -> dict:
        with open(f"{self.get('folder_paths.rn_graph.arches.path')}/traffic_links_2024_2025-02-27.geojson", "r", encoding="utf-8") as a:
            return geojson.load(a)["features"]


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
        "1.6": manage_global,
        "2.1": manage_downloads,
        "2.2": manage_downloads,
        "3.1.1": manage_forecasting_horizon,
        "3.1.2": manage_forecasting_horizon,
        "3.1.3": manage_forecasting_horizon,
        "3.2.1": forecasts_warmup,
        "3.2.2": forecasts_warmup,
        "3.2.3": forecasts_warmup,
        "3.2.4": forecasts_warmup,
        "3.2.5": forecasts_warmup,
        "3.2.6": forecasts_warmup,
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
    1.6 List all projects
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
