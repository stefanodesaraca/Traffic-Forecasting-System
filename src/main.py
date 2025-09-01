import json
import os
import sys
import time
import pickle
from pprint import pprint
import asyncio
import dask
import dask.dataframe as dd

from exceptions import TRPNotFoundError
from definitions import GlobalDefinitions, DBConfig, ProjectTables
from downloader import start_client_async, volumes_to_db, fetch_trps, fetch_trps_from_ids
from brokers import AIODBManagerBroker, AIODBBroker, DBBroker
from pipelines import MeanSpeedIngestionPipeline, RoadGraphObjectsIngestionPipeline, MLPreprocessingPipeline, MLPredictionPipeline
from loaders import BatchStreamLoader
from ml import TFS
from road_network import RoadNetwork
from utils import dask_cluster_client, check_target, split_by_target

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
        await (await get_aiodbmanager_broker()).insert_trps(data=await fetch_trps(
            gql_client=await start_client_async()))
        print("Traffic registration points information downloaded successfully\n\n")


    elif functionality == "2.2":
        await (await get_aiodbmanager_broker()).insert_trps(
            data=await fetch_trps_from_ids(gql_client=await start_client_async(), trp_ids=(await asyncio.to_thread(input, "Insert the TRP IDs which you want to ingest into the DB separated by commas: ")).strip().split(",")))


    elif functionality == "2.3":
        time_start = await asyncio.to_thread(input, "Insert starting datetime (of the time frame which you're interested in) - YYYY-MM-DDTHH: ") + ":00:00.00" + GlobalDefinitions.NORWEGIAN_UTC_TIME_ZONE
        time_end = await asyncio.to_thread(input, "Insert ending datetime (of the time frame which you're interested in) - YYYY-MM-DDTHH: ") + ":00:00.00" + GlobalDefinitions.NORWEGIAN_UTC_TIME_ZONE
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
            time_start=await asyncio.to_thread(input, "Insert starting datetime (of the time frame which you're interested in) - YYYY-MM-DDTHH: ") + ":00:00.00" + GlobalDefinitions.NORWEGIAN_UTC_TIME_ZONE,
            time_end=await asyncio.to_thread(input, "Insert ending datetime (of the time frame which you're interested in) - YYYY-MM-DDTHH: ") + ":00:00.00" + GlobalDefinitions.NORWEGIAN_UTC_TIME_ZONE,
            max_retries=5
        )
        print(f"Downloading traffic volumes data for TRPs: {list(trp_ids)}...")

    return None


async def mean_speeds_to_db(_: str) -> None:
    pipeline = MeanSpeedIngestionPipeline(db_broker_async=await get_aiodb_broker())
    semaphore = asyncio.Semaphore(10)

    async def limited_ingest(file: str) -> None:
        async with semaphore:
            await pipeline.ingest(fp=GlobalDefinitions.MEAN_SPEED_DIR / file, fields=GlobalDefinitions.MEAN_SPEED_INGESTION_FIELDS)
        return None

    await asyncio.gather(*(limited_ingest(file=file) for file in await asyncio.to_thread(os.listdir, GlobalDefinitions.MEAN_SPEED_DIR)))
    return None


async def manage_forecasting_horizon(functionality: str) -> None:
    db_broker_async = await get_aiodb_broker()

    if functionality == "3.1.1":
        print("-- Forecasting horizon setter --")
        await db_broker_async.set_forecasting_horizon_async()

    elif functionality == "3.1.2":
        print("-- Forecasting horizon reader --")
        print("V: Volumes | MS: Mean Speed")
        target = await asyncio.to_thread(input, "Choice: ")
        print("Target datetime: ", await db_broker_async.get_forecasting_horizon_async(target=target.upper()), "\n\n")

    elif functionality == "3.1.3":
        print("-- Forecasting horizon reset --")
        print("V: Volumes | MS: Mean Speed")
        target = await asyncio.to_thread(input, "Choice: ")
        await db_broker_async.reset_forecasting_horizon_async(target=target.upper())

    return None


def eda() -> None:
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


def forecast_warmup(functionality: str) -> None:
    db_broker = get_db_broker()
    loader = BatchStreamLoader(db_broker=db_broker)
    preprocessing_pipeline = MLPreprocessingPipeline()

    #NOTE FOR A FUTURE UPDATE WE'LL INTEGRATE THE ABILITY TO PREDICT AT DIFFERENT TIME HORIZONS (LONG TERM PREDICTIONS AND SHORT TERM PREDICTIONS)
    #if long_term:
    #    lags = [24, 36, 48, 60, 72]  # One, two and three days in the past
    #else:
    #    lags = [8766, 17532, 26298]  # One, two and three years in the past


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


    def ml_gridsearch(X_train: dd.DataFrame, y_train: dd.DataFrame, learner: TFS) -> None:

        gridsearch_results = learner.gridsearch(X_train, y_train)
        learner.export_gridsearch_results(gridsearch_results)

        print(f"============== {learner.model.name} grid search results ==============\n")
        print(gridsearch_results, "\n")

        return None


    def ml_training(X_train: dd.DataFrame, y_train: dd.DataFrame, learner: TFS) -> None:
        print(f"Fitting phase for model: {learner.model.name} started...")
        learner.model.fit(X_train, y_train)
        print("Fitting phase ended")

        print("Exporting trained model...")
        learner.export_internal_model()
        print("Model exported correctly to the DB")
        return None


    def ml_testing(X_test: dd.DataFrame, y_test: dd.DataFrame, learner: TFS) -> None:
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
                                                                             has_mean_speed=True if target == GlobalDefinitions.MEAN_SPEED else None,
                                                                             county_ids_filter=[GlobalDefinitions.OSLO_COUNTY_ID]).items():

            print(f"\n********************* Executing data preprocessing for road category: {road_category} *********************\n")

            X_train, X_test, y_train, y_test = split_by_target(
                data=functionality_mapping[functionality]["preprocessing_pipeline"](
                    data=functionality_mapping[functionality]["loading_method"](
                        batch_size=50000,
                        county_ids_filter=[GlobalDefinitions.OSLO_COUNTY_ID],
                        trp_list_filter=trp_ids,
                        road_category_filter=[road_category],
                        encoded_cyclical_features=True,
                        is_mice=False,
                        is_covid_year=True,
                        trp_lat=True,
                        trp_lon=True,
                        sort_by_date=True,
                        sort_ascending=True
                    ),
                    lags=[24, 36, 48, 60, 72],
                    z_score=True
                ),
                target=target,
                mode=0
            )
            print(f"Shape of the merged data for road category {road_category}: ",
                  X_train.shape[0].compute() + X_test.shape[0].compute() + y_train.shape[0].compute() + y_test.shape[0].compute())

            for model, content in models.items():
                #print(content)
                if functionality_mapping[functionality]["type"] == "gridsearch":
                    func(X_train, y_train, TFS(
                            model=content["binary"](**content["params"]),
                            road_category=road_category,
                            target=target,
                            client=client,
                            db_broker=db_broker
                        )
                    )
                elif functionality_mapping[functionality]["type"] == "training":
                    func(X_test, y_test, TFS(
                            model=content["binary"](**content["params"]),
                            target=target, #Needed for model export
                            road_category=road_category, #Needed for model export
                            db_broker=db_broker
                        )
                    )
                elif functionality_mapping[functionality]["type"] == "testing":
                    func(X_test, y_test, TFS(
                            model=content["binary"],
                            target=target,
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
                "loading_method": loader.get_volume,
                "preprocessing_pipeline": preprocessing_pipeline.preprocess_volume,
                "model_query": get_model_query(operation_type="gridsearch", target=GlobalDefinitions.VOLUME)
            },
            "3.2.2": {
                "func": ml_gridsearch,
                "type": "gridsearch",
                "target": GlobalDefinitions.MEAN_SPEED,
                "loading_method": loader.get_mean_speed,
                "preprocessing_pipeline": preprocessing_pipeline.preprocess_mean_speed,
                "model_query": get_model_query(operation_type="gridsearch", target=GlobalDefinitions.MEAN_SPEED)
            },
            "3.2.3": {
                "func": ml_training,
                "type": "training",
                "target": GlobalDefinitions.VOLUME,
                "loading_method": loader.get_volume,
                "preprocessing_pipeline": preprocessing_pipeline.preprocess_volume,
                "model_query": get_model_query(operation_type="training", target=GlobalDefinitions.VOLUME)
            },
            "3.2.4": {
                "func": ml_training,
                "type": "training",
                "target": GlobalDefinitions.MEAN_SPEED,
                "loading_method": loader.get_mean_speed,
                "preprocessing_pipeline": preprocessing_pipeline.preprocess_mean_speed,
                "model_query": get_model_query(operation_type="training", target=GlobalDefinitions.MEAN_SPEED)
            },
            "3.2.5": {
                "func": ml_testing,
                "type": "testing",
                "target": GlobalDefinitions.VOLUME,
                "loading_method": loader.get_volume,
                "preprocessing_pipeline": preprocessing_pipeline.preprocess_volume,
                "model_query": get_model_query(operation_type="testing", target=GlobalDefinitions.VOLUME)
            },
            "3.2.6": {
                "func": ml_testing,
                "type": "testing",
                "target": GlobalDefinitions.MEAN_SPEED,
                "loading_method": loader.get_mean_speed,
                "preprocessing_pipeline": preprocessing_pipeline.preprocess_mean_speed,
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


    if functionality == "5.2":
        db_broker = get_db_broker()

        print("Models available: ")
        pprint(db_broker.get_ml_models())

        model_id = input("Enter the model ID: ")
        target = input("Enter the target variable of the grid you want to update: ")
        check_target(target=target, errors=True)
        grid_fp = input("Enter grid filepath: ")

        with open(grid_fp, "r", encoding="utf-8") as f:
            grid = json.load(f)

        db_broker.update_model_grid(
            model_id=model_id,
            target=target,
            grid=grid
        )

    if functionality == "5.3":
        db_broker = get_db_broker()

        grids_fp = input("Enter grids filepath: ")

        with open(grids_fp, "r", encoding="utf-8") as f:
            grids = json.load(f)

        for model_id, target_grids in grids.items():
            for target, grid in target_grids.items():
                if target != "base_parameters":
                    db_broker.update_model_grid(
                        model_id=model_id,
                        target=target,
                        grid=grid
                    )

    return None


def forecast(functionality: str) -> None:
    db_broker = get_db_broker()

    print("Enter target data to forecast: ")
    print("V: Volumes | MS: Mean Speed")
    target = input("Choice: ").upper()

    check_target(target, errors=True)
    target = GlobalDefinitions.TARGET_DATA[target]

    if functionality == "3.3.1":

        with dask_cluster_client(processes=False) as client:
            trps_with_data = db_broker.get_all_trps_metadata(**{f"has_{target}_filter": True}).keys()
            trp_ids = [d.get("id") for d in db_broker.get_trp_ids() if d.get("id") in trps_with_data]
            print("TRP IDs: ", trp_ids)
            trp_id = input("Insert TRP ID for forecasting: ")

            if trp_id not in trp_ids:
                raise TRPNotFoundError("TRP ID not in available TRP IDs list")

            trp_road_category = db_broker.get_trp_metadata(trp_id=trp_id)["road_category"]
            print("TRP road category: ", trp_road_category)

            for name, model in {m["name"]: pickle.loads(m["pickle_object"]) for m in db_broker.get_trained_model_objects(target=target, road_category=trp_road_category)}.items():  # Load model name and data (pickle object, the best parameters and so on)

                pipeline = MLPredictionPipeline(
                    trp_id=trp_id,
                    road_category=trp_road_category,
                    target=target,
                    db_broker=db_broker,
                    loader=BatchStreamLoader(db_broker),
                    preprocessing_pipeline=MLPreprocessingPipeline(),
                    model=model,
                )

                print(f"**************** {name}'s Predictions ****************")

                data = pipeline.start(trp_tuning=True) #TODO TO TRY WITH trp_tuning=True

                print(data.compute())



    return None


async def setup_road_network() -> None:
    pipeline = RoadGraphObjectsIngestionPipeline(db_broker_async=await get_aiodb_broker())
    print("Setting up road network data...")
    await pipeline.ingest_toll_stations(fp="data/road_network/toll_stations.json")
    await pipeline.ingest_nodes(fp="data/road_network/traffic-nodes-2024_2025-02-28.geojson")
    await pipeline.ingest_links(fp="data/road_network/traffic_links_2024_2025-02-27.geojson")
    print("Road network data successfully ingested into the DB")
    return None


def manage_road_network(functionality: str) -> None:

    if functionality == "4.1":
        asyncio.run(setup_road_network())

    elif functionality == "4.2":
        db_broker = get_db_broker()
        network = RoadNetwork(network_id="test", name="test", db_broker=db_broker,
                              loader=BatchStreamLoader(db_broker=db_broker))
        network.build(county_ids_filter=[GlobalDefinitions.OSLO_COUNTY_ID])
        network.find_route(source="633614", destination="633286")


        # TODO FILTER ONLY THE OSLO MUNICIPALITY!
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
        "3.2.1": forecast_warmup,
        "3.2.2": forecast_warmup,
        "3.2.3": forecast_warmup,
        "3.2.4": forecast_warmup,
        "3.2.5": forecast_warmup,
        "3.2.6": forecast_warmup,
        "3.3.1": forecast,
        "4.1": manage_road_network,
        "4.2": manage_road_network,
        "4.3": manage_road_network,
        "5.1": manage_ml,
        "5.2": manage_ml,
        "5.3": manage_ml,
        "5.4": eda
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
    4.1 Set up road network
    4.2 Graph generation
    4.3 Graph analysis
5. Other options
    5.1 Update best parameters for a model
    5.2 Update model grid
    5.3 Update multiple model grids
    5.4 EDA (Exploratory Data Analysis)
    5.5 Erase all data about a project
    5.6 Analyze pre-existing road network graph
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
