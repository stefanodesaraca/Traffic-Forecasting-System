import json
from contextlib import contextmanager
from typing import Any
import asyncio
import aiofiles
from distributed.utils_test import async_wait_for
from gql.transport.exceptions import TransportServerError
import asyncpg
from asyncpg.exceptions import InvalidCatalogNameError, DuplicateDatabaseError

from db_config import DBConfig
from tfs_downloader import start_client_async, fetch_areas, fetch_road_categories, fetch_trps


@contextmanager
async def postgres_conn(user: str, password: str, dbname: str) -> asyncpg.connection:
    try:
        conn = await asyncpg.connect(
            user=user,
            password=password,
            database=dbname,
            host='localhost'
        )
        yield conn
    finally:
        await conn.close()


async def check_db(dbname: str) -> bool:
    async with postgres_conn(user=DBConfig.SUPERUSER.value, password=DBConfig.SUPERUSER_PASSWORD.value, dbname="postgres") as conn:
        return await conn.fetchval(
                "SELECT 1 FROM pg_database WHERE datname = $1",
                dbname
            ) == 1



async def init() -> None:

    # -- Initialize users and DBs --

    #Accessing as superuser and creating tfs user
    async with postgres_conn(user=DBConfig.SUPERUSER.value, password=DBConfig.SUPERUSER_PASSWORD.value, dbname=DBConfig.MAINTENANCE_DB.value) as conn:
        try:
            await conn.execute(f"CREATE USER 'tfs' WITH PASSWORD 'tfs'")
            print(f"User 'tfs' created.")
        except asyncpg.DuplicateObjectError:
            print(f"User 'username' already exists.")

    async with postgres_conn(user=DBConfig.SUPERUSER.value, password=DBConfig.SUPERUSER_PASSWORD.value, dbname=DBConfig.MAINTENANCE_DB.value) as conn:
        try:
            await conn.execute(f"CREATE USER 'tfs' WITH PASSWORD 'tfs'")
            print(f"User 'tfs' created.")
        except asyncpg.DuplicateObjectError:
            print(f"User 'username' already exists.")


    if not await check_db(dbname=DBConfig.HUB_DB.value):
        async with postgres_conn(user=DBConfig.TFS_USER.value, password=DBConfig.TFS_PASSWORD.value, dbname=DBConfig.HUB_DB.value) as conn:
            try:
                await conn.execute(f"""
                CREATE DATABASE {DBConfig.HUB_DB.value}
                """)
            except DuplicateDatabaseError:
                pass

    async with postgres_conn(user=DBConfig.TFS_USER.value, password=DBConfig.TFS_PASSWORD.value, dbname=DBConfig.HUB_DB.value) as conn:

        #Projects
        await conn.execute("""
                 CREATE TABLE IF NOT EXISTS Metadata (
                    id SERIAL PRIMARY KEY,
                    current_project_id TEXT,
                    lang TEXT,
                    FOREIGN KEY (current_project_id) REFERENCES Projects(id)
                 )
                 CREATE TABLE IF NOT EXISTS Projects (
                    id SERIAL PRIMARY KEY,
                    name TEXT,
                    lang TEXT
                 )
        """)

        #Tables
        await conn.execute("""
                CREATE TABLE IF NOT EXISTS RoadCategories (
                    id TEXT PRIMARY KEY,
                    name TEXT
                );
                
                CREATE TABLE IF NOT EXISTS CountryParts (
                    id INTEGER PRIMARY KEY,
                    name TEXT
                );
                
                CREATE TABLE IF NOT EXISTS Counties (
                    number INTEGER PRIMARY KEY,
                    name TEXT,
                    country_part_id TEXT,
                    FOREIGN KEY (country_part_id) REFERENCES CountryParts(id)
                );
                
                CREATE TABLE IF NOT EXISTS Municipalities (
                    number INTEGER PRIMARY KEY,
                    name TEXT,
                    county_number INTEGER,
                    country_part_id TEXT,
                    FOREIGN KEY (county_number) REFERENCES Counties(number),
                    FOREIGN KEY (country_part_id) REFERENCES CountryParts(id)
                );
                
                CREATE TABLE IF NOT EXISTS TrafficRegistrationPoints (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    lat FLOAT,
                    lon FLOAT,
                    road_reference_short_form TEXT,
                    road_category TEXT,
                    road_link_sequence INTEGER,
                    relative_position FLOAT,
                    county TEXT,
                    country_part_id TEXT,
                    country_part_name TEXT,
                    county_number INTEGER,
                    geographic_number INTEGER,
                    traffic_registration_type TEXT,
                    first_data TIMESTAMPTZ,
                    first_data_with_quality_metrics TIMESTAMPTZ,
                    FOREIGN KEY (road_category) REFERENCES RoadCategories(name)
                );
                
                CREATE TABLE IF NOT EXISTS Data (
                    row_idx SERIAL PRIMARY KEY,
                    trp_id TEXT,
                    volume INTEGER,
                    volume_coverage FLOAT,
                    volume_mice BOOLEAN,
                    mean_speed FLOAT,
                    mean_speed_coverage FLOAT,
                    mean_speed_mice BOOLEAN,
                    percentile_85 FLOAT,
                    zoned_dt_iso TIMESTAMPTZ,
                    FOREIGN KEY (trp_id) REFERENCES TrafficRegistrationPoints(id)
                );
                        
        """)


        # -- Check if any projects exist --

        project_check = await conn.fetchrow("SELECT * FROM Projects LIMIT 1")

        #If there aren't any projects, let the user impute one and insert it into the Projects table
        if not project_check:
            print("Initialize the program. Create your first project!")
            name = input("Enter project name: ")
            lang = input("Enter project language: ")

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


        print("Setting up necessary data...")

        print("Trying to download areas data...")
        areas = await fetch_areas(await start_client_async())
        if areas:
            async with conn.transaction():
                await insert_areas(conn=conn, data=areas)
                print("Areas inserted correctly into project db")

        else:
            print("Areas download failed, load them from a JSON file")
            json_file = input("Enter json areas file path: ")
            async with aiofiles.open(json_file, "r", encoding="utf-8") as a:
                await insert_areas(conn=conn, data=json.load(a))


        print("Trying to download road categories data...")
        road_categories = await fetch_road_categories(await (start_client_async()))
        if road_categories:
            async with conn.transaction():
                await insert_road_categories(conn=conn, data=road_categories)
                print("Road categories inserted correctly into project db")

        else:
            print("Road categories download failed, load them from a JSON file")
            json_file = input("Enter json road categories file path: ")
            async with aiofiles.open(json_file, "r", encoding="utf-8") as a:
                await insert_areas(conn=conn, data=json.load(a))


        print("Trying to download TRPs' data...")
        trps = await fetch_trps(await start_client_async())
        if trps:
            async with conn.transaction():
                await insert_trps(conn=conn, data=trps)
                print("TRPs' data inserted correctly into project db")

        else:
            print("TRPs' data download failed, load them from a JSON file")
            json_file = input("Enter json TRPs' data file path: ")
            async with aiofiles.open(json_file, "r", encoding="utf-8") as a:
                await insert_trps(conn=conn, data=json.load(a))

    return None




















































