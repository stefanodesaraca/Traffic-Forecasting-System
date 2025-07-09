from contextlib import contextmanager
import aiosqlite


@contextmanager
async def aiosqlite_conn(dbname: str, auto_commit: bool = False):
    conn = await aiosqlite.connect(f"{dbname}.db")
    try:
        yield conn
    finally:
        if auto_commit:
            await conn.commit()
        await conn.close()


def init() -> None:
    with aiosqlite_conn(dbname="tfs_hub", auto_commit=True) as conn:
        conn.execute("""
                 CREATE TABLE IF NOT EXISTS metadata (
                    id TEXT PRIMARY KEY
                    current_project TEXT
                    lang TEXT
                 )
                 CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY
                    lang TEXT
                 )
        """)

        conn.execute("""
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
                    firstData TIMESTAMPTZ,
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




















































