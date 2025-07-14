import json
from contextlib import contextmanager
from typing import Any
import asyncio
import aiofiles
import asyncpg
from asyncpg.exceptions import DuplicateDatabaseError
from cleantext import clean

from db_config import DBConfig
from downloader import start_client_async, fetch_areas, fetch_road_categories, fetch_trps


@contextmanager
async def postgres_conn(user: str, password: str, dbname: str, host: str = 'localhost') -> asyncpg.connection:
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


class DBManager:

    def __init__(self, superuser: str, superuser_password: str, tfs_user: str, tfs_password: str, hub_db: str = "tfs_hub", maintenance_db: str = "postgres"):
        self._superuser = superuser
        self._superuser_password = superuser_password
        self._tfs_user = tfs_user
        self._tfs_password = tfs_password
        self._hub_db = hub_db
        self._maintenance_db = maintenance_db


    @staticmethod
    async def _check_db(dbname: str) -> bool:
        async with postgres_conn(user=DBConfig.SUPERUSER.value, password=DBConfig.SUPERUSER_PASSWORD.value, dbname="postgres") as conn:
            return await conn.fetchval(
                    "SELECT 1 FROM pg_database WHERE datname = $1",
                    dbname
                ) == 1


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
        async with postgres_conn(user=self._superuser, password=self._superuser_password, dbname=self._maintenance_db) as conn:
            # Accessing as superuser since some tools may require this configuration to create a new database
            async with conn.transaction():
                await conn.execute(f"""CREATE DATABASE {name};""")

        # -- Project Tables Setup --
        async with postgres_conn(user=self._tfs_user, password=self._tfs_password, dbname=name) as conn:
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
                            id SERIAL PRIMARY KEY,
                            name TEXT NOT NULL,
                            type TEXT DEFAULT 'Regression',
                            volume_grid JSON NOT NULL,
                            mean_speed_grid JSON NOT NULL,
                            best_volume_gridsearch_params INT DEFAULT 1,
                            best_mean_speed_gridsearch_params INT DEFAULT 1
                        );
                        
                        CREATE TABLE IF NOT EXISTS MLModelObjects (
                            id SERIAL PRIMARY KEY,
                            joblib_object BYTEA,
                            pickle_object BYTEA NOT NULL,
                            FOREIGN KEY (id) REFERENCES MLModels(id)
                        );
                        
                        CREATE TABLE IF NOT EXISTS ModelGridSearchCVResults (
                            id SERIAL,
                            model_id INT REFERENCES MLModels(id) ON DELETE CASCADE,
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
                """)

        # -- New Project Metadata Insertions --
        async with postgres_conn(user=self._tfs_user, password=self._tfs_password, dbname=DBConfig.HUB_DB.value) as conn:
            new_project = await conn.fetchrow(
                "INSERT INTO Projects (name, lang) VALUES ($1, $2) RETURNING *",
                name, lang
            )
            print(f"New project created: {new_project}")

            await conn.execute(
                "INSERT INTO Metadata (current_project_id, lang) VALUES ($1, $2)",
                new_project['id'], new_project['lang']
            )
            print(f"Metadata updated setting {new_project['name']} as current project.")

        # -- New Project Setup Insertions --
        async with postgres_conn(user=self._tfs_user, password=self._tfs_password, dbname=name) as conn:
            if auto_project_setup:
                await self._setup_project(conn=conn)

        return None


    async def init(self, auto_project_setup: bool = True) -> None:

        # -- Initialize users and DBs --

        #Accessing as superuser and creating tfs user
        async with postgres_conn(user=self._superuser, password=self._superuser_password, dbname=self._maintenance_db) as conn:
            try:
                await conn.execute(f"CREATE USER 'tfs' WITH PASSWORD 'tfs'")
                print(f"User 'tfs' created.")
            except asyncpg.DuplicateObjectError:
                print(f"User 'username' already exists.")

        async with postgres_conn(user=self._superuser, password=self._superuser_password, dbname=self._maintenance_db) as conn:
            try:
                await conn.execute(f"CREATE USER 'tfs' WITH PASSWORD 'tfs'")
                print(f"User 'tfs' created.")
            except asyncpg.DuplicateObjectError:
                print(f"User 'username' already exists.")


        if not await self._check_db(dbname=DBConfig.HUB_DB.value):
            async with postgres_conn(user=DBConfig.TFS_USER.value, password=DBConfig.TFS_PASSWORD.value, dbname=DBConfig.HUB_DB.value) as conn:
                try:
                    await conn.execute(f"""
                    CREATE DATABASE {DBConfig.HUB_DB.value}
                    """)
                except DuplicateDatabaseError:
                    pass

            # -- Hub DB Setup (If It Doesn't Exist) --
            async with postgres_conn(user=self._tfs_user, password=self._tfs_password, dbname=self._hub_db) as conn:

                #Projects
                await conn.execute("""
                        CREATE TABLE IF NOT EXISTS Projects (
                            id SERIAL PRIMARY KEY,
                            name TEXT,
                            lang TEXT
                         )
                         
                         CREATE TABLE IF NOT EXISTS Metadata (
                            id SERIAL PRIMARY KEY,
                            current_project_id TEXT,
                            lang TEXT,
                            FOREIGN KEY (current_project_id) REFERENCES Projects(id)
                         )
                """)


        # -- Check if any projects exist --
        async with postgres_conn(user=self._tfs_user, password=self._tfs_password, dbname=self._hub_db) as conn:
            project_check = await conn.fetchrow("SELECT * FROM Projects LIMIT 1")

            #If there aren't any projects, let the user impute one and insert it into the Projects table
            if not project_check:
                print("Initialize the program. Create your first project!")
                name = clean(input("Enter project name: "), no_emoji=True, no_punct=True, no_emails=True, no_currency_symbols=True, no_urls=True, normalize_whitespace=True, lower=True)
                lang = input("Enter project language: ")
                print("Cleaned project DB name: ", name)
                print("Project language: ", lang)

                await self.create_project(name=name, lang=lang, auto_project_setup=auto_project_setup)

        return None























































