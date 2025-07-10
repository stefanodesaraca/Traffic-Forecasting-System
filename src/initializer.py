from contextlib import contextmanager
import asyncpg
from asyncpg.exceptions import InvalidCatalogNameError, DuplicateDatabaseError
from db_config import DBConfig


@contextmanager
async def postgres_conn(user: str, password: str) -> asyncpg.connection:
    try:
        conn = await asyncpg.connect(
            user=user,
            password=password,
            database='postgres',
            host='localhost'
        )
        yield conn
    finally:
        await conn.close()

@contextmanager
async def tfs_db_conn(dbname: str) -> asyncpg.connection:
    try:
        conn = await asyncpg.connect(
            user=DBConfig.TFS_USER.value,
            password=DBConfig.TFS_PASSWORD.value,
            database=dbname,
            host='localhost'
        )
        yield conn
    finally:
        await conn.close()


async def check_db(dbname: str) -> bool:
    async with postgres_conn(user="postgres", password="") as conn:
        return await conn.fetchval(
                "SELECT 1 FROM pg_database WHERE datname = $1",
                dbname
            ) == 1


async def init() -> None:

    #Accessing as superuser and creating tfs user
    async with postgres_conn(user=DBConfig.SUPERUSER.value, password=DBConfig.SUPERUSER_PASSWORD.value) as conn:
        try:
            await conn.execute(f"CREATE USER 'tfs' WITH PASSWORD 'tfs'")
            print(f"User 'tfs' created.")
        except asyncpg.DuplicateObjectError:
            print(f"User 'username' already exists.")

    async with postgres_conn(user=DBConfig.SUPERUSER.value, password=DBConfig.SUPERUSER_PASSWORD.value) as conn:
        try:
            await conn.execute(f"CREATE USER 'tfs' WITH PASSWORD 'tfs'")
            print(f"User 'tfs' created.")
        except asyncpg.DuplicateObjectError:
            print(f"User 'username' already exists.")

    async with postgres_conn(user=DBConfig.TFS_USER.value, password=DBConfig.TFS_PASSWORD.value) as conn:
        try:
            await conn.execute("""
            CREATE DATABASE tfs_hub
            """)
        except DuplicateDatabaseError:
            pass

    await check_db(DBConfig.HUB_DB.value)


    async with tfs_db_conn(dbname=DBConfig.HUB_DB.value) as conn:

        await conn.execute("""
                 CREATE TABLE IF NOT EXISTS Metadata (
                    id TEXT PRIMARY KEY
                    current_project TEXT
                    lang TEXT
                 )
                 CREATE TABLE IF NOT EXISTS Projects (
                    id TEXT PRIMARY KEY
                    name TEXT
                    lang TEXT
                 )
        """)

        #Tables
        await conn.execute("""
                CREATE TABLE IF NOT EXISTS RoadCategories (
                    id INT PRIMARY KEY,
                    name TEXT
                );
                
                CREATE TABLE IF NOT EXISTS CountryParts (
                    id INT PRIMARY KEY,
                    name TEXT
                );
                
                CREATE TABLE IF NOT EXISTS Counties (
                    number INT PRIMARY KEY,
                    name TEXT,
                    country_part_id TEXT,
                    FOREIGN KEY (country_part_id) REFERENCES CountryParts(id)
                );
                
                CREATE TABLE IF NOT EXISTS TrafficRegistrationPoints (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    lat FLOAT,
                    lon FLOAT,
                    road_reference_short_form TEXT,
                    road_category TEXT,
                    road_link_sequence INT,
                    relative_position FLOAT,
                    county TEXT,
                    country_part_id TEXT,
                    country_part_name TEXT,
                    county_number INT,
                    geographic_number INT,
                    traffic_registration_type TEXT,
                    first_data TIMESTAMPTZ,
                    first_data_with_quality_metrics TIMESTAMPTZ,
                    FOREIGN KEY (road_category) REFERENCES RoadCategories(name)
                );
                
                CREATE TABLE IF NOT EXISTS Data (
                    row_idx INT PRIMARY KEY,
                    trp_id TEXT,
                    volume INT,
                    volume_coverage FLOAT,
                    mean_speed FLOAT,
                    mean_speed_coverage FLOAT,
                    percentile_85 FLOAT,
                    zoned_dt_iso TIMESTAMPTZ,
                    FOREIGN KEY (trp_id) REFERENCES TrafficRegistrationPoints(id)
                );
                        
        """)







    return None




















































