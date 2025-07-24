import json
from datetime import datetime, timezone, timedelta
from contextlib import contextmanager
from typing import Any, Literal
import asyncio
import aiofiles
import asyncpg
from asyncpg.exceptions import DuplicateDatabaseError
import psycopg
from psycopg.rows import tuple_row, dict_row
from cleantext import clean

from exceptions import ProjectDBNotFoundError, ProjectDBNotRegisteredError
from downloader import start_client_async, fetch_areas, fetch_road_categories, fetch_trps


@contextmanager
async def postgres_conn_async(user: str, password: str, dbname: str, host: str = 'localhost') -> asyncpg.connection:
    try:
        conn = await asyncpg.connect(
            user=user,
            password=password,
            database=dbname,
            host=host
        )
        yield conn
    finally:
        await conn.close()


@contextmanager
def postgres_conn(user: str, password: str, dbname: str, host: str = 'localhost', autocommit: bool = True, row_factory: Literal["tuple_row", "dict_row"] = "dict_row") -> psycopg.connection:
    row_factories = {
        "tuple_row": tuple_row,
        "dict_row": dict_row
    }
    try:
        conn=psycopg.connect(
            dbname=user,
            user=password,
            password=dbname,
            host=host,
            row_factory=row_factories.get(row_factory, tuple_row)
        )
        conn.autocommit = autocommit
        yield conn
    finally:
        conn.close()



class AIODBManager:

    def __init__(self, superuser: str, superuser_password: str, tfs_user: str, tfs_password: str, hub_db: str = "tfs_hub", maintenance_db: str = "postgres"):
        self._superuser = superuser
        self._superuser_password = superuser_password
        self._tfs_user = tfs_user
        self._tfs_password = tfs_password
        self._hub_db = hub_db
        self._maintenance_db = maintenance_db


    async def _check_db(self, dbname: str) -> bool:
        async with postgres_conn_async(user=self._superuser, password=self._superuser_password, dbname="postgres") as conn:
            project_db_check = await conn.fetchval(
                    "SELECT 1 FROM pg_database WHERE datname = $1",
                    dbname
                ) == 1 #First check layer
            project_record_check = bool(await conn.fetchrow(f"SELECT * FROM Projects LIMIT 1 WHERE name = {dbname}")) #Second check layer
            if not project_db_check:
                raise ProjectDBNotFoundError("Project DB doesn't exist")
            elif not project_record_check:
                raise ProjectDBNotRegisteredError("Project DB exists, but hasn't been registered within the ones available")
            return True


    @staticmethod
    async def _setup_project(conn: asyncpg.connection) -> None:
        # -- Fetch or import necessary data to work with during program usage --

        async def insert_areas(conn: asyncpg.connection, data: dict[str, Any]) -> None:
            all((await conn.execute(
                "INSERT INTO CountryParts (id, name) VALUES ($1, $2) ON CONFLICT (id) DO NOTHING",
                part["id"], part["name"]
            ),
                 all((await conn.execute(
                     "INSERT INTO Counties (number, name, country_part_id) VALUES ($1, $2, $3) ON CONFLICT (number) DO NOTHING",
                     county["number"], county["name"], part["id"]
                 ),
                      all(await conn.execute(
                          """
                          INSERT INTO Municipalities (number, name, county_number, country_part_id)
                          VALUES ($1, $2, $3, $4)
                          ON CONFLICT (number) DO NOTHING
                          """,
                          muni["number"], muni["name"], county["number"], part["id"]
                      ) for muni in county.get("municipalities", []))
                      ) for county in part["counties"])) for part in data["data"]["areas"]["countryParts"])
            return None

        async def insert_road_categories(conn: asyncpg.connection, data: dict[str, Any]) -> None:
            all(await conn.execute(
                """
                INSERT INTO RoadCategories (id, name)
                VALUES ($1, $2)
                ON CONFLICT (id) DO NOTHING
                """,
                cat["id"], cat["name"]
            ) for cat in data["data"]["roadCategories"])
            return None

        async def insert_trps(conn: asyncpg.connection, data: dict[str, Any]) -> None:
            all(await conn.execute(
                f"""
                            INSERT INTO TrafficRegistrationPoints (
                                id, name, lat, lon,
                                road_reference_short_form, road_category,
                                road_link_sequence, relative_position,
                                county, country_part_id, country_part_name,
                                county_number, geographic_number,
                                traffic_registration_type,
                                first_data, first_data_with_quality_metrics
                            )
                            VALUES (
                                {trp["id"]}, 
                                {trp["name"]}, 
                                {trp["location"]["coordinates"]["latLon"]["lat"]}, 
                                {trp["location"]["coordinates"]["latLon"]["lon"]},
                                {trp["location"].get("roadReference", {}).get("shortForm")}, 
                                {trp["location"].get("roadReference", {}).get("shortForm").get("roadCategory", {}).get("id")}, 
                                {trp["location"].get("roadLinkSequence", {}).get("roadLinkSequenceId")}, 
                                {trp["location"].get("roadLinkSequence", {}).get("relativePosition")},
                                {trp["location"].get("county", {}).get("name")}, 
                                {str(trp["location"].get("county", {}).get("countryPart", {}).get("id")) if trp["location"].get("county", {}).get("countryPart", {}).get("id") is not None else None}, 
                                {trp["location"].get("county", {}).get("countryPart", {}).get("name")}, 
                                {trp["location"].get("county", {}).get("number")},
                                {trp["location"].get("county", {}).get("geographicNumber")}, 
                                {trp.get("trafficRegistrationType")},
                                {trp.get("dataTimeSpan", {}).get("firstData")}, 
                                {trp.get("dataTimeSpan", {}).get("firstDataWithQualityMetrics")})
                            ON CONFLICT (id) DO NOTHING
                            """
            ) for trp in data["data"]["trafficRegistrationPoints"])
            return None

        fetch_funcs = (fetch_areas, fetch_road_categories, fetch_trps)
        insert_funcs = (insert_areas, insert_road_categories, insert_trps)
        try_desc = ("Trying to download areas data...",
                    "Trying to download road categories data...",
                    "Trying to download TRPs' data...")
        success_desc = ("Areas inserted correctly into project db",
                        "Road categories inserted correctly into project db",
                        "TRPs' data inserted correctly into project db")
        fail_desc = ("Areas download failed, load them from a JSON file",
                     "Road categories download failed, load them from a JSON file",
                     "TRPs' data download failed, load them from a JSON file")
        json_enter_desc = ("Enter json areas file path: ",
                           "Enter json road categories file path: ",
                           "Enter json TRPs' data file path: ")

        print("Setting up necessary data...")
        for fetch, insert, t, s, f, je in zip(fetch_funcs, insert_funcs, try_desc, success_desc, fail_desc, json_enter_desc, strict=True):
            print(t)
            data = await fetch(await start_client_async())
            if data:
                async with conn.transaction():
                    await insert(conn=conn, data=data)
                    print(s)
            else:
                print(f)
                json_file = input(je)
                async with aiofiles.open(json_file, "r", encoding="utf-8") as a:
                    await insert(conn=conn, data=json.load(a))

        return None


    async def create_project(self, name: str, lang: str, auto_project_setup: bool = True) -> None:

        # -- New Project DB Setup --
        async with postgres_conn_async(user=self._superuser, password=self._superuser_password, dbname=self._maintenance_db) as conn:
            # Accessing as superuser since some tools may require this configuration to create a new database
            async with conn.transaction():
                await conn.execute(f"""CREATE DATABASE {name};""")

        # -- Project Tables Setup --
        async with postgres_conn_async(user=self._tfs_user, password=self._tfs_password, dbname=name) as conn:
            async with conn.transaction():
                # Tables
                await conn.execute("""
                        CREATE TABLE IF NOT EXISTS RoadCategories (
                            id TEXT PRIMARY KEY,
                            name TEXT NOT NULL
                        );

                        CREATE TABLE IF NOT EXISTS CountryParts (
                            id INTEGER PRIMARY KEY,
                            name TEXT NOT NULL
                        );

                        CREATE TABLE IF NOT EXISTS Counties (
                            number INTEGER PRIMARY KEY,
                            name TEXT NOT NULL,
                            country_part_id TEXT NOT NULL,
                            FOREIGN KEY (country_part_id) REFERENCES CountryParts(id)
                        );

                        CREATE TABLE IF NOT EXISTS Municipalities (
                            number INTEGER PRIMARY KEY,
                            name TEXT NOT NULL,
                            county_number INTEGER NOT NULL,
                            country_part_id TEXT NOT NULL,
                            FOREIGN KEY (county_number) REFERENCES Counties(number),
                            FOREIGN KEY (country_part_id) REFERENCES CountryParts(id)
                        );

                        CREATE TABLE IF NOT EXISTS TrafficRegistrationPoints (
                            id TEXT PRIMARY KEY,
                            name TEXT NOT NULL,
                            lat FLOAT NOT NULL,
                            lon FLOAT NOT NULL,
                            road_reference_short_form TEXT NOT NULL,
                            road_category TEXT NOT NULL,
                            road_link_sequence INTEGER NOT NULL,
                            relative_position FLOAT NOT NULL,
                            county TEXT NOT NULL,
                            country_part_id TEXT NOT NULL,
                            country_part_name TEXT NOT NULL,
                            county_number INTEGER NOT NULL,
                            geographic_number INTEGER NOT NULL,
                            traffic_registration_type TEXT NOT NULL,
                            first_data TIMESTAMPTZ,
                            first_data_with_quality_metrics TIMESTAMPTZ,
                            FOREIGN KEY (road_category) REFERENCES RoadCategories(name)
                        );

                        CREATE TABLE IF NOT EXISTS Volume (
                            row_idx SERIAL PRIMARY KEY,
                            trp_id TEXT NOT NULL,
                            volume INTEGER NOT NULL,
                            coverage FLOAT NOT NULL,
                            is_mice BOOLEAN DEFAULT FALSE,
                            zoned_dt_iso TIMESTAMPTZ NOT NULL,
                            FOREIGN KEY (trp_id) REFERENCES TrafficRegistrationPoints(id)
                        );

                        CREATE TABLE IF NOT EXISTS MeanSpeed(
                            row_idx SERIAL PRIMARY KEY,
                            trp_id TEXT NOT NULL,
                            mean_speed FLOAT NOT NULL,
                            coverage FLOAT NOT NULL,
                            is_mice BOOLEAN DEFAULT FALSE,
                            percentile_85 FLOAT NOT NULL,
                            zoned_dt_iso TIMESTAMPTZ NOT NULL,
                            FOREIGN KEY (trp_id) REFERENCES TrafficRegistrationPoints(id)
                        );

                        CREATE TABLE IF NOT EXISTS TrafficRegistrationPointsMetadata (
                            trp_id TEXT PRIMARY KEY,
                            has_volume BOOLEAN DEFAULT FALSE,
                            has_mean_speed BOOLEAN DEFAULT FALSE,
                            volume_start_date TIMESTAMPTZ,
                            volume_end_date TIMESTAMPTZ,
                            mean_speed_start_date TIMESTAMPTZ,
                            mean_speed_end_date TIMESTAMPTZ,
                            FOREIGN KEY (trp_id) REFERENCES TrafficRegistrationPoints(id)
                        );
                        
                        CREATE TABLE IF NOT EXISTS MLModels (
                            id TEXT PRIMARY KEY,
                            name TEXT NOT NULL UNIQUE,
                            type TEXT DEFAULT 'Regression',
                            base_params JSON NOT NULL,
                            volume_grid JSON NOT NULL,
                            mean_speed_grid JSON NOT NULL,
                            best_volume_gridsearch_params_idx INT DEFAULT 1,
                            best_mean_speed_gridsearch_params_idx INT DEFAULT 1
                        );
                        
                        CREATE TABLE IF NOT EXISTS MLModelObjects (
                            id TEXT PRIMARY KEY,
                            joblib_object BYTEA,
                            pickle_object BYTEA NOT NULL,
                            FOREIGN KEY (id) REFERENCES MLModels(id)
                        );
                        
                        CREATE TABLE IF NOT EXISTS ModelGridSearchCVResults (
                            id SERIAL,
                            model_id TEXT REFERENCES MLModels(id) ON DELETE CASCADE,
                            road_category_id TEXT REFERENCES RoadCategories(id) ON DELETE CASCADE,
                            params JSON NOT NULL,
                            mean_fit_time FLOAT NOT NULL,
                            mean_test_r2 FLOAT NOT NULL,
                            mean_train_r2 FLOAT NOT NULL,
                            mean_test_mean_squared_error FLOAT NOT NULL,
                            mean_train_mean_squared_error FLOAT NOT NULL,
                            mean_test_root_mean_squared_error FLOAT NOT NULL,
                            mean_train_root_mean_squared_error FLOAT NOT NULL,
                            mean_test_mean_absolute_error FLOAT NOT NULL,
                            mean_train_mean_absolute_error FLOAT NOT NULL,
                            PRIMARY KEY (id, model_id, road_category_id)
                        );
                        
                        CREATE TABLE IF NOT EXISTS ForecastingSettings (
                            id BOOLEAN PRIMARY KEY CHECK (id = TRUE),
                            config JSONB DEFAULT {'volume_forecasting_horizon": NULL, 'mean_speed_forecasting_horizon": NULL}
                        );
                """)


                #Constraints
                await conn.execute("""
                            ALTER TABLE Volume
                            ADD CONSTRAINT unique_volume_per_trp_and_time
                            UNIQUE (trp_id, zoned_dt_iso);
                            
                            ALTER TABLE MeanSpeed
                            ADD CONSTRAINT unique_mean_speed_per_trp_and_time
                            UNIQUE (trp_id, zoned_dt_iso);
                """) #There can only be one registration at one specific time and location (where the location is the place where the TRP lies)


                # Views
                await conn.execute("""
                CREATE OR REPLACE VIEW TrafficRegistrationPointsMetadataView AS
                SELECT
                    trp.id AS trp_id,
                    BOOL_OR(v.volume IS NOT NULL) AS has_volume,
                    BOOL_OR(ms.mean_speed IS NOT NULL) AS has_mean_speed,
                    MIN(CASE WHEN v.volume IS NOT NULL THEN v.zoned_dt_iso END) AS volume_start_date,
                    MAX(CASE WHEN v.volume IS NOT NULL THEN v.zoned_dt_iso END) AS volume_end_date,
                    MIN(CASE WHEN ms.mean_speed IS NOT NULL THEN ms.zoned_dt_iso END) AS mean_speed_start_date,
                    MAX(CASE WHEN ms.mean_speed IS NOT NULL THEN ms.zoned_dt_iso END) AS mean_speed_end_date
                FROM
                    TrafficRegistrationPoints trp
                LEFT JOIN
                    Volume v ON trp.id = v.trp_id
                LEFT JOIN
                    MeanSpeed ms ON trp.id = ms.trp_id
                GROUP BY
                    trp.id;
                
                CREATE OR REPLACE VIEW VolumeMeanSpeedDateRangesView AS
                SELECT
                    MIN(v.zoned_dt_iso) AS volume_start_date,
                    MAX(v.zoned_dt_iso) AS volume_end_date,
                    MIN(ms.zoned_dt_iso) AS mean_speed_start_date,
                    MAX(ms.zoned_dt_iso) AS mean_speed_end_date
                FROM Volume v
                FULL OUTER JOIN MeanSpeed ms ON false;  -- force Cartesian for aggregation without joining
                """)

        # -- New Project Metadata Insertions --
        async with postgres_conn_async(user=self._tfs_user, password=self._tfs_password, dbname=self._hub_db) as conn:
            new_project = await conn.fetchrow(
                "INSERT INTO Projects (name, lang, is_current, creation_zoned_dt) VALUES ($1, $2, $3, $4) RETURNING *",
                name, lang, False, datetime.now(tz=timezone(timedelta(hours=1)))
            )
            print(f"New project created: {new_project}")

        # -- New Project Setup Insertions --
        async with postgres_conn_async(user=self._tfs_user, password=self._tfs_password, dbname=name) as conn:
            if auto_project_setup:
                await self._setup_project(conn=conn)

        return None


    async def delete_project(self, name: str) -> None:
        async with postgres_conn_async(user=self._tfs_user, password=self._tfs_password, dbname=self._hub_db) as conn:
            # Step 1: Deleting the actual project database
            await conn.execute(f"""
                DROP DATABASE IF EXISTS {name}
            """)
            # Step 2: Deleting the project record from the Projects table in the Hub DB
            # Creating a function that deletes a project by its name and executing a transaction directly from the SQL statement
            await conn.execute(f"""
                CREATE OR REPLACE FUNCTION delete_project_by_name(p_name TEXT)
                RETURNS BOOLEAN AS $$
                DECLARE
                    deleted_is_current BOOLEAN;
                BEGIN
                    DELETE FROM Projects
                    WHERE name = p_name
                    RETURNING is_current INTO deleted_is_current;
                
                    IF deleted_is_current THEN
                        UPDATE Projects
                        SET is_current = TRUE
                        WHERE id = (
                            SELECT id
                            FROM Projects
                            ORDER BY creation_zoned_dt DESC
                            LIMIT 1
                        );
                    END IF;
                
                    RETURN deleted_is_current;
                END;
                $$ LANGUAGE plpgsql;
                """)
            if (await conn.fetchval("SELECT delete_project_by_name($1)", name)) is True and (current_project := await self.get_current_project()): #If the deleted project was the current one then... (if the return statement of delete_project_by_name is True then the deleted project was the current one)
                print(f"The deleted project was the current one, now the current project is: {current_project}")
            else:
                print("The deleted project was the only one existing. Create a new one or exit the program? - 1: Yes | 0: Exit")
                if choice := await asyncio.to_thread(lambda: input("Choice: ")) == "1":
                    await self.init()
                elif choice == "0":
                    exit()
                else:
                    raise Exception(f"Wrong input {choice}")
        return None


    async def init(self, auto_project_setup: bool = True) -> None:

        # -- Initialize users and DBs --

        #Accessing as superuser and creating tfs user
        async with postgres_conn_async(user=self._superuser, password=self._superuser_password, dbname=self._maintenance_db) as conn:
            try:
                await conn.execute(f"CREATE USER '{self._tfs_user}' WITH PASSWORD '{self._tfs_password}'")
                print(f"User '{self._tfs_user}' created.")
            except asyncpg.DuplicateObjectError:
                print(f"User '{self._tfs_user}' already exists.")


        # -- Hub DB Initialization --
        if not await self._check_db(dbname=self._hub_db):
            # -- Hub DB Creation --
            async with postgres_conn_async(user=self._tfs_user, password=self._tfs_password, dbname=self._hub_db) as conn:
                try:
                    await conn.execute(f"""
                    CREATE DATABASE {self._hub_db}
                    """)
                except DuplicateDatabaseError:
                    pass

            # -- Hub DB Setup (If It Doesn't Exist) --
            async with postgres_conn_async(user=self._tfs_user, password=self._tfs_password, dbname=self._hub_db) as conn:
                #Hub DB Tables (Projects)
                await conn.execute("""
                        CREATE TABLE IF NOT EXISTS Projects (
                            id SERIAL PRIMARY KEY,
                            name TEXT NOT NULL UNIQUE,
                            lang TEXT,
                            is_current BOOL NOT NULL,
                            creation_zoned_dt TIMESTAMPTZ NOT NULL
                        )
                """)

                #Hub DB Constraints
                await conn.execute("""
                        CREATE UNIQUE INDEX one_current_project ON Projects (is_current)
                        WHERE is_current = TRUE;
                """)


        # -- Check if any projects exist --
        async with postgres_conn_async(user=self._tfs_user, password=self._tfs_password, dbname=self._hub_db) as conn:
            project_check = await conn.fetchrow("SELECT * FROM Projects LIMIT 1")

            #If there aren't any projects, let the user impute one and insert it into the Projects table
            if not project_check:
                print("Initialize the program. Create your first project!")
                name = clean(input("Enter project name: "), no_emoji=True, no_punct=True, no_emails=True, no_currency_symbols=True, no_urls=True, normalize_whitespace=True, lower=True)
                lang = input("Enter project language: ")
                print("Cleaned project DB name: ", name)
                print("Project language: ", lang)

                await self.create_project(name=name, lang=lang, auto_project_setup=auto_project_setup)


        #TODO IF SOME PROJECTS EXIST CHECK WHICH IS THE CURRENT ONE, IF THE RETURN IS NONE SET IT AS CURRENT. ALSO, ADD THE ABILITY TO DO THAT INSIDE CREATE_PROJECT VIA A PARAMETER (NOT AN INPUT)
        # LIKE auto_current_setup: bool = False
        # ALSO IF auto_current_setup IS TRUE, CHECK DO reset_current_project() FIRST AND THEN SET IT


        return None


    async def get_current_project(self) -> asyncpg.Record | None:
        async with postgres_conn_async(user=self._tfs_user, password=self._tfs_password, dbname=self._hub_db) as conn:
            async with conn.transaction():
                return await conn.fetchrow("""
                        SELECT *
                        FROM Projects
                        WHERE is_current = TRUE;
                """)


    async def set_current_project(self, name: str) -> None:
        async with postgres_conn_async(user=self._tfs_user, password=self._tfs_password, dbname=self._hub_db) as conn:
            await self._check_db(name) #If the project doesn't exist raise error
            async with conn.transaction(): #Needing to execute both of the operations in one transaction because otherwise the one_current_project constraint wouldn't be respected. Checkout the Hub DB Constraints sections to learn more
                await conn.execute("UPDATE Projects SET is_current = FALSE WHERE is_current = TRUE;")
                await conn.execute(f"""                
                    UPDATE Projects
                    SET is_current = TRUE
                    WHERE name = {name};
                """)
        return None


    async def reset_current_project(self, name: str) -> None:
        async with postgres_conn_async(user=self._tfs_user, password=self._tfs_password, dbname=self._hub_db) as conn:
            await self._check_db(name) #If the project doesn't exist raise error
            async with conn.transaction():
                await conn.execute("""
                    UPDATE Projects
                    SET is_current = FALSE
                    WHERE is_current = TRUE;
                """)
        return None



















































