import os
import sys
import time
import pickle
import asyncio
import pandas as pd
import dask
import dask.dataframe as dd
import geojson

from exceptions import TRPNotFoundError, ModelBestParametersNotFound
from db_config import DBConfig, ProjectTables

from downloader import start_client_async, volumes_to_db, fetch_trps, fetch_trps_from_ids
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
    os.makedirs(GlobalDefinitions.MEAN_SPEED_DIR, exist_ok=True) #The directory where mean speed files need to be placed
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
        await (await get_aiodbmanager_broker()).insert_trps(data=await fetch_trps(gql_client=await start_client_async()))
        print("Traffic registration points information downloaded successfully\n\n")


    elif functionality == "2.2":
        await (await get_aiodbmanager_broker()).insert_trps(
            data=await fetch_trps_from_ids(gql_client=await start_client_async(), trp_ids=(await asyncio.to_thread(input, "Insert the TRP IDs which you want to ingest into the DB separated by commas: ")).strip().split(",")))


    elif functionality == "2.3":
        time_start = await asyncio.to_thread(input, "Insert starting datetime (of the time frame which you're interested in) - YYYY-MM-DDTHH: ") + ":00:00.000" + GlobalDefinitions.NORWEGIAN_UTC_TIME_ZONE
        time_end = await asyncio.to_thread(input, "Insert ending datetime (of the time frame which you're interested in) - YYYY-MM-DDTHH: ") + ":00:00.000" + GlobalDefinitions.NORWEGIAN_UTC_TIME_ZONE
        print("Downloading traffic volumes data for every registration point for the current project...")
        await volumes_to_db(
            db_broker_async=await get_aiodb_broker(),
            trp_ids=(trp_record["id"] for trp_record in await (await get_aiodb_broker()).get_trp_ids_async()),
            time_start=time_start,
            time_end=time_end,
            n_async_jobs=5,
            max_retries=5
        )


    elif functionality == "2.4":
        trp_ids = await asyncio.to_thread(lambda: input("Insert the TRP IDs for which you want to ingest data for separated by commas: ").strip().split(","))
        await volumes_to_db(
            db_broker_async=await get_aiodb_broker(),
            trp_ids=trp_ids,
            time_start=await asyncio.to_thread(input, "Insert starting datetime (of the time frame which you're interested in) - YYYY-MM-DDTHH: ") + ":00:00.000" + GlobalDefinitions.NORWEGIAN_UTC_TIME_ZONE,
            time_end=await asyncio.to_thread(input, "Insert ending datetime (of the time frame which you're interested in) - YYYY-MM-DDTHH: ") + ":00:00.000" + GlobalDefinitions.NORWEGIAN_UTC_TIME_ZONE,
            max_retries=5
        )
        print(f"Downloading traffic volumes data for TRPs: {list(trp_ids)}...")

    return None


async def mean_speeds_to_db(_: str) -> None:
    pipeline = MeanSpeedExtractionPipeline(db_broker_async=await get_aiodb_broker())
    semaphore = asyncio.Semaphore(10)

    async def limited_ingest(file: str) -> None:
        async with semaphore:
            await pipeline.ingest(fp=GlobalDefinitions.MEAN_SPEED_DIR / file, fields=GlobalDefinitions.MEAN_SPEED_INGESTION_FIELDS)
        return None

    await asyncio.gather(*(limited_ingest(file=file) for file in await asyncio.to_thread(os.listdir, GlobalDefinitions.MEAN_SPEED_DIR)))
    return None


async def manage_forecasting_horizon(functionality: str) -> None:
    ft = ForecastingToolbox(db_broker_async=await get_aiodb_broker())

    if functionality == "3.1.1":
        print("-- Forecasting horizon setter --")
        await ft.set_forecasting_horizon_async()

    elif functionality == "3.1.2":
        print("-- Forecasting horizon reader --")
        print("V: Volumes | MS: Mean Speed")
        target = await asyncio.to_thread(input, "Choice: ")
        print("Target datetime: ", await ft.get_forecasting_horizon_async(target=target.upper()), "\n\n")

    elif functionality == "3.1.3":
        print("-- Forecasting horizon reset --")
        print("V: Volumes | MS: Mean Speed")
        target = await asyncio.to_thread(input, "Choice: ")
        await ft.reset_forecasting_horizon_async(target=target.upper())

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


def forecasts_warmup(functionality: str) -> None:
    db_broker = get_db_broker()
    loader = BatchStreamLoader(db_broker=db_broker)


    def get_model_query(operation_type: str, target: str):
        return {
            "gridsearch": f"""
                            SELECT 
                                m.id as id,
                                m.name as name, 
                                m.base_params AS params,
                                bm.pickle_object AS pickle_object
                            FROM
                                "{ProjectTables.MLModels.value}" m
                            JOIN
                                "{ProjectTables.BaseModels.value}" bm ON m.id = bm.id;
                            """,  # models_base_params_query
            "training": f"""
                            SELECT 
                                bgr.model_id as id,
                                bgr.model_name as name,
                                bgr.best_{target}_params as params,
                                bm.pickle_object AS pickle_object
                            FROM 
                                "best_{target}_gridsearch_results" bgr
                            JOIN
                                "{ProjectTables.BaseModels.value}" bm ON bgr.model_id = bm.id;
                            """,
            # models_best_params_query #TODO ADD best_{target}_gridsearch_results TO ProjectViews
            "testing": f"""
                            SELECT 
                                m.id as id,
                                m.target as target,
                                m.name as name,
                                bm.pickle_object AS pickle_object
                            FROM 
                                "{ProjectTables.MLModels.value}" m
                            JOIN
                                "{ProjectTables.TrainedModels.value}" tm ON m.model_id = tm.id
                            WHERE m.target = {target};
                            """
            # models_best_params_query #TODO ADD best_{target}_gridsearch_results TO ProjectViews
        }.get(operation_type, None)


    def ml_gridsearch(X_train: dd.DataFrame, y_train: dd.DataFrame, learner: callable) -> None:

        gridsearch_results = learner.gridsearch(X_train, y_train)
        learner.export_gridsearch_results(gridsearch_results)

        print(f"============== {learner.model.name} grid search results ==============\n")
        print(gridsearch_results, "\n")

        return None


    def ml_training(X_train: dd.DataFrame, y_train: dd.DataFrame, learner: callable) -> None:
        print(f"Fitting phase for model: {learner.model.name} started")
        learner.model.fit(X_train, y_train).export()
        print("Fitting phase ended")
        return None


    def ml_testing(X_test: dd.DataFrame, y_test: dd.DataFrame, learner: callable) -> None:
        y_pred = learner.model.predict(X_test)
        learner.compute_fpe(y_true=y_test, y_pred=y_pred)
        return None


    def process_functionality(func: callable) -> None:

        models = {
            m["name"]: {
                "binary": pickle.loads(m["pickle_object"]),
                "params": m.get("params", None)
            }
            for m in db_broker.send_sql(functionality_mapping[functionality]["model_query"])
        }

        for road_category, trp_ids in db_broker.get_trp_ids_by_road_category(has_volumes=True if target == GlobalDefinitions.VOLUME else None,
                                                                             has_mean_speed=True if target == GlobalDefinitions.MEAN_SPEED else None).items():

            print(f"\n********************* Executing data preprocessing for road category: {road_category} *********************\n")

            X_train, X_test, y_train, y_test = split_by_target(
                data=getattr(preprocessor := TFSPreprocessor(
                    data=getattr(loader, functionality_mapping[functionality]["loading_method"])(
                        batch_size=50000,
                        trp_list_filter=trp_ids,
                        road_category_filter=road_category,
                        split_cyclical_features=False,
                        encoded_cyclical_features=True,
                        is_covid_year=True,
                        sort_by_date=True,
                        sort_ascending=True),
                    road_category=road_category,
                    client=client
                ), functionality_mapping[functionality]["preprocessing_method"])(),
                target=target,
                mode=0
            )
            print(f"Shape of the merged data for road category {road_category}: ", preprocessor.shape)

            for model, metadata in models:
                if functionality_mapping[functionality]["type"] == "gridsearch":
                    func(X_train, y_train, TFSLearner(
                            model=models[model]["binary"](models[model]["params"]),
                            road_category=road_category,
                            target=target,
                            client=client,
                            db_broker=db_broker
                        )
                    )
                elif functionality_mapping[functionality]["type"] == "training":
                    func(X_test, y_test, TFSLearner(
                            model=models[model]["binary"](models[model]["params"]),
                            road_category=road_category,
                            target=target,
                            client=client,
                            db_broker=db_broker
                        )
                    )
                elif functionality_mapping[functionality]["type"] == "testing":
                    func(X_test, y_test, TFSLearner(
                            model=models[model]["binary"],
                            road_category=road_category,
                            target=target,
                            client=client,
                            db_broker=db_broker
                        )
                    )

        return None


    with dask_cluster_client(processes=False) as client:

        functionality_mapping = {
            "3.2.1": {
                "func": ml_gridsearch,
                "type": "gridsearch",
                "target": GlobalDefinitions.VOLUME,
                "preprocessing_method": "preprocess_volume",
                "loading_method": "get_volume",
                "model_query": get_model_query(operation_type="gridsearch", target=GlobalDefinitions.VOLUME)
            },
            "3.2.2": {
                "func": ml_gridsearch,
                "type": "gridsearch",
                "target": GlobalDefinitions.MEAN_SPEED,
                "preprocessing_method": "preprocess_mean_speed",
                "loading_method": "get_mean_speed",
                "model_query": get_model_query(operation_type="gridsearch", target=GlobalDefinitions.MEAN_SPEED)
            },
            "3.2.3": {
                "func": ml_training,
                "type": "training",
                "target": GlobalDefinitions.VOLUME,
                "preprocessing_method": "preprocess_volume",
                "loading_method": "get_volume",
                "model_query": get_model_query(operation_type="training", target=GlobalDefinitions.VOLUME)
            },
            "3.2.4": {
                "func": ml_training,
                "type": "training",
                "target": GlobalDefinitions.MEAN_SPEED,
                "preprocessing_method": "preprocess_mean_speed",
                "loading_method": "get_mean_speed",
                "model_query": get_model_query(operation_type="training", target=GlobalDefinitions.MEAN_SPEED)
            },
            "3.2.5": {
                "func": ml_testing,
                "type": "testing",
                "target": GlobalDefinitions.VOLUME,
                "preprocessing_method": "preprocess_volume",
                "loading_method": "get_volume",
                "model_query": get_model_query(operation_type="testing", target=GlobalDefinitions.VOLUME)
            },
            "3.2.6": {
                "func": ml_testing,
                "type": "testing",
                "target": GlobalDefinitions.MEAN_SPEED,
                "preprocessing_method": "preprocess_mean_speed",
                "loading_method": "get_mean_speed",
                "model_query": get_model_query(operation_type="testing", target=GlobalDefinitions.MEAN_SPEED)
            }
        }

        if functionality not in functionality_mapping:
            raise ValueError(f"Unknown functionality: {functionality}")

        target = functionality_mapping[functionality]["target"]
        process_functionality(functionality_mapping[functionality]["func"]) #Process the chosen operation

        print("Alive Dask cluster workers: ", dask.distributed.worker.Worker._instances)
        time.sleep(1)  # To cool down the system

    return None


def manage_ml(functionality: str) -> None:

    if functionality == "5.1":
        db_broker = get_db_broker()

        print("Available models: ")
        db_broker.send_sql(f"""
            SELECT * FROM {ProjectTables.MLModels.value}
        """)

        model_id = input("Enter the ID of the model which you want to set the best parameters index for: ")

        print("\nV: Volumes | MS: Mean Speed")
        target = input("Enter the target variable for which the model has been trained for: ")
        check_target(target, errors=True)

        new_best_params_idx = int(input("Enter the new best parameters index for the model (integer value): "))


        db_broker.send_sql(f"""
                            UPDATE "{ProjectTables.MLModels.value}"
                            SET "{f"'best_{target}_gridsearch_params_idx'"}" = {new_best_params_idx}
                            WHERE "id" = '{model_id}';
        """)

        print(f"Best parameters for model: {model_id} ad target: {target} updated correctly")
        return None


    return None





def execute_forecasting(functionality: str) -> None:
    db_broker = get_db_broker()
    loader = BatchStreamLoader(db_broker=db_broker)
    ft = ForecastingToolbox(db_broker=db_broker)

    print("Enter target data to forecast: ")
    print("V: Volumes | MS: Mean Speed")
    option = input("Choice: ").upper()

    check_target(option, errors=True)

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
                target=GlobalDefinitions.TARGET_DATA[option],
                client=client,
                db_broker=db_broker,
                loader=loader
            )
            future_records = forecaster.get_future_records(forecasting_horizon=ft.get_forecasting_horizon(target=GlobalDefinitions.TARGET_DATA[option]))  #Already preprocessed

            #TODO TEST training_mode = BOTH 0 AND 1
            model_training_dataset = forecaster.get_training_records(
                training_mode=0,
                limit=future_records.shape[0].compute() * 24
            )
            X, y = split_by_target(
                data=model_training_dataset,
                target=GlobalDefinitions.TARGET_DATA[option],
                mode=1
            )

            for name, data in db_broker.get_model_objects()["model_data"].items(): #Load model name and data (pickle object, the best parameters and so on)

                model = pickle.load(data[name]["pickle_object"])
                best_params = data[name][f"{GlobalDefinitions.TARGET_DATA[option]}_best_params"]

                if best_params is None:
                    raise ModelBestParametersNotFound("Model's best parameters are None, check if the model has been trained or has best parameters set")

                learner = TFSLearner(
                    model=model(**best_params),
                    road_category=trp_road_category,
                    target=GlobalDefinitions.TARGET_DATA[option],
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
        "2.3": manage_downloads,
        "2.4": manage_downloads,
        "2.5": mean_speeds_to_db,
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
        "5.1": manage_ml,
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
    2.2 Traffic registration points information by IDs
    2.3 Traffic volumes for every TRP
    2.4 Volumes data for a single TRP by ID
    2.5 Ingest mean speed data
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
    5.1 Update best parameters for a model
    5.2 EDA (Exploratory Data Analysis)
    5.3 Erase all data about a project
    5.4 Analyze pre-existing road network graph
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
