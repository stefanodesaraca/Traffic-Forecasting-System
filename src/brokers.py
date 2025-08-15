from typing import Any, Literal, Iterator
from pydantic.types import PositiveInt
import asyncpg

from exceptions import WrongSQLStatement, MissingDataException
from dbmanager import AIODBManager, postgres_conn_async, postgres_conn
from db_config import HubDBTables, ProjectTables, ProjectViews, RowFactories



class AIODBBroker:
    #The AIODBBroker returns records directly from the db without any transformation to ensure
    # a standard query output format which is then treated differently by each function or method that requires db data

    def __init__(self, db_user: str, db_password: str, db_name: str, db_host: str):
        self._db_user = db_user
        self._db_password = db_password
        self._db_name = db_name
        self._db_host = db_host


    async def send_sql_async(self, sql: str, single: bool = False, many: bool = False, many_values: list[tuple[Any, ...]] | None = None) -> Any:
        async with postgres_conn_async(user=self._db_user, password=self._db_password, dbname=self._db_name, host=self._db_host) as conn:
            async with conn.transaction():
                if any(sql.lstrip().startswith(prefix) for prefix in ["SELECT", "select"]):
                    if single:
                        return await conn.fetchrow(sql)
                    return await conn.fetch(sql)
                elif any(sql.lstrip().startswith(prefix) for prefix in ["INSERT", "UPDATE", "DELETE", "insert", "update", "delete"]):
                    if many and many_values:
                        return await conn.executemany(sql, many_values)
                    elif many and not many_values:
                        raise MissingDataException("Missing data to insert")
                    return await conn.execute(sql)
                else:
                    raise WrongSQLStatement("The SQL query isn't correct")


    async def get_trp_ids_async(self, road_category_filter: list[str] | None = None) -> list[asyncpg.Record]:
        async with postgres_conn_async(user=self._db_user, password=self._db_password, dbname=self._db_name, host=self._db_host) as conn:
            async with conn.transaction():
                return await conn.fetch(f"""
                    SELECT id FROM "{ProjectTables.TrafficRegistrationPoints.value}";
                    {"WHERE road_category = ANY($1)" if road_category_filter else ""}
                """, *tuple(f for f in [road_category_filter] if f))


    async def get_trp_ids_by_road_category_async(self) -> list[asyncpg.Record]:
        async with postgres_conn_async(user=self._db_user, password=self._db_password, dbname=self._db_name, host=self._db_host) as conn:
            async with conn.transaction():
                return await conn.fetch(f"""SELECT json_object_agg(road_category, ids) AS result
                                           FROM (
                                               SELECT road_category, json_agg(id ORDER BY id) AS ids
                                               FROM "{ProjectTables.TrafficRegistrationPoints.value}"
                                               GROUP BY road_category
                                           ) AS sub;""")


    async def get_volume_date_boundaries_async(self) -> dict[str, Any]:
        async with postgres_conn_async(user=self._db_user, password=self._db_password, dbname=self._db_name, host=self._db_host) as conn:
            async with conn.transaction():
                result = await conn.fetchrow(f"""
                    SELECT volume_start_date, volume_end_date
                    FROM "{ProjectViews.VolumeMeanSpeedDateRangesView.value}"
                """)
                return {"min": result["volume_start_date"], "max": result["volume_end_date"]} #Respectively: min and max


    async def get_mean_speed_date_boundaries_async(self) -> dict[str, Any]:
        async with postgres_conn_async(user=self._db_user, password=self._db_password, dbname=self._db_name, host=self._db_host) as conn:
            async with conn.transaction():
                result = await conn.fetchrow(f"""
                    SELECT mean_speed_start_date, mean_speed_end_date
                    FROM "{ProjectViews.VolumeMeanSpeedDateRangesView.value}"
                """)
                return {"min": result["mean_speed_start_date"], "max": result["mean_speed_end_date"]} #Respectively: min and max



class DBBroker:
    #The SyncDBBroker returns records directly from the db without any transformation to ensure
    # a standard query output format which is then treated differently by each function or method that requires db data

    def __init__(self, db_user: str, db_password: str, db_name: str, db_host: str):
        self._db_user = db_user
        self._db_password = db_password
        self._db_name = db_name
        self._db_host = db_host


    def send_sql(self, sql: str, single: bool = False, many: bool = False, many_values: list[tuple[Any, ...]] | None = None, row_factory: Literal["tuple_row", "dict_row"] = "dict_row", execute_args: list[Any] | None = None) -> Any:
        with postgres_conn(user=self._db_user, password=self._db_password, dbname=self._db_name, host=self._db_host, row_factory=row_factory) as conn:
            with conn.cursor(row_factory=RowFactories.factories.get(row_factory)) as cur:
                if any(sql.lstrip().startswith(prefix) for prefix in ["SELECT", "select"]):
                    cur.execute(sql)
                    if single:
                        return cur.fetchone()
                    return cur.fetchall()
                elif any(sql.lstrip().startswith(prefix) for prefix in ["INSERT", "UPDATE", "DELETE", "insert", "update", "delete"]):
                    if many and many_values:
                        return cur.executemany(sql, many_values)
                    elif many and not many_values:
                        raise MissingDataException("Missing data to insert")
                    return cur.execute(sql, *execute_args) if execute_args else cur.execute(sql)
                else:
                    raise WrongSQLStatement("The SQL query isn't correct")


    def get_stream(self, sql: str, batch_size: PositiveInt, filters: tuple[Any] | tuple[list[Any], ...], row_factory: Literal["tuple_row", "dict_row"] = "dict_row") -> Iterator:
        if "SELECT" not in sql:
            raise WrongSQLStatement("Cannot return a data stream from a non selective statement (SELECT)")
        with postgres_conn(user=self._db_user, password=self._db_password, dbname=self._db_name, host=self._db_host, row_factory=row_factory) as conn:
            with conn.cursor().stream(query=sql, *filters, size=batch_size) as cursor:
                return cursor


    def get_trp_ids(self, road_category_filter: list[str] | None = None) -> list[tuple[Any, ...]]:
        with postgres_conn_async(user=self._db_user, password=self._db_password, dbname=self._db_name, host=self._db_host) as conn:
            with conn.transaction():
                return conn.fetchall(f"""
                    SELECT id FROM "{ProjectTables.TrafficRegistrationPoints.value}";
                    {"WHERE road_category = ANY(%s)" if road_category_filter else ""}
                """, *tuple(f for f in [road_category_filter] if f))


    def get_trp_ids_by_road_category(self) -> dict[Any, ...]:
        with postgres_conn_async(user=self._db_user, password=self._db_password, dbname=self._db_name, host=self._db_host) as conn:
            with conn.transaction():
                return conn.fetchall(f"""SELECT json_object_agg(road_category, ids) AS result
                                        FROM (
                                            SELECT road_category, json_agg(id ORDER BY id) AS ids
                                            FROM "{ProjectTables.TrafficRegistrationPoints.value}"
                                            GROUP BY road_category
                                        ) AS sub;
                                     """)
            #Output example:
            #{
            #    "E": ["17684V2460285", "17900V111222"],
            #    "R": ["03375V625405"]
            #}


    def get_volume_date_boundaries(self) -> dict[str, Any]:
        with postgres_conn(user=self._db_user, password=self._db_password, dbname=self._db_name, host=self._db_host) as conn:
            with conn.transaction():
                result = conn.fetchone(f"""
                    SELECT volume_start_date, volume_end_date
                    FROM "{ProjectViews.VolumeMeanSpeedDateRangesView.value}"
                """)
                return {"min": result["volume_start_date"], "max": result["volume_end_date"]} #Respectively: min and max


    def get_mean_speed_date_boundaries(self) -> dict[str, Any]:
        with postgres_conn(user=self._db_user, password=self._db_password, dbname=self._db_name, host=self._db_host) as conn:
            with conn.transaction():
                result = conn.fetchone(f"""
                    SELECT mean_speed_start_date, mean_speed_end_date
                    FROM "{ProjectViews.VolumeMeanSpeedDateRangesView.value}"
                """)
                return {"min": result["mean_speed_start_date"], "max": result["mean_speed_end_date"]} #Respectively: min and max


    def get_all_trps_metadata(self) -> dict[str, dict[str, Any]]:
        with postgres_conn(user=self._db_user, password=self._db_password, dbname=self._db_name, host=self._db_host) as conn:
            with conn.transaction():
                return conn.fetchone()(f"""
                    SELECT jsonb_object_agg(trp_id, to_jsonb(t) - 'trp_id') AS trp_metadata
                    FROM (
                        SELECT *
                        FROM "{ProjectViews.TrafficRegistrationPointsMetadataView.value}"
                    ) AS t;
                """)


    def get_trp_metadata(self, trp_id: str) -> dict[str, Any]:
        with postgres_conn(user=self._db_user, password=self._db_password, dbname=self._db_name, host=self._db_host) as conn:
            with conn.transaction():
                return conn.fetchone(f"""
                    SELECT TO_JSONB(t)
                    FROM "{ProjectViews.TrafficRegistrationPointsMetadataView.value}" t
                    WHERE trp_id = %s;
                """, trp_id)

    #TODO TRANSFER THIS OPERATION DIRECTLY INTO main.py AND USE send_sql()
    def get_model_objects(self) -> dict[str, dict[str, Any]]:
        with postgres_conn(user=self._db_user, password=self._db_password, dbname=self._db_name, host=self._db_host) as conn:
            with conn.transaction():
                return conn.fetchall(f"""
                    SELECT json_object_agg(
                        m.name,
                        json_build_object(
                            'pickle_object', o.pickle_object,
                            'volume_best_params', m.volume_best_params,
                            'mean_speed_best_params', m.mean_speed_best_params
                        )
                    ) AS model_data
                    FROM {ProjectTables.MLModels.value} m
                    JOIN {ProjectTables.MLModelObjects.value} o ON m.id = o.id;
                """)



class AIODBManagerBroker:

    def __init__(self,
                 superuser: str | None = None,
                 superuser_password: str | None = None,
                 tfs_user: str | None = None,
                 tfs_password: str | None = None,
                 tfs_role: str | None = None,
                 tfs_role_password: str | None = None,
                 hub_db: str | None = None,
                 maintenance_db: str | None = None,
                 db_host: str | None = None):
        self._superuser: str | None = superuser
        self._superuser_password: str | None = superuser_password
        self._tfs_user: str | None = tfs_user
        self._tfs_password: str | None = tfs_password
        self._tfs_role: str | None = tfs_role
        self._tfs_role_password: str | None = tfs_role_password
        self._hub_db: str | None = hub_db
        self._maintenance_db: str | None = maintenance_db
        self._db_host: str | None = db_host


    async def _get_db_manager_async(self) -> AIODBManager:
        if none_params := [name for name, value in locals().items() if value is None]:
            raise ValueError(f"Missing required parameters: {', '.join(none_params)}")
        return AIODBManager(
            superuser=self._superuser,
            superuser_password=self._superuser_password,
            tfs_user=self._tfs_user,
            tfs_password=self._tfs_password,
            tfs_role=self._tfs_password,
            tfs_role_password=self._tfs_password,
            hub_db=self._hub_db,
            maintenance_db=self._maintenance_db
        )


    async def init(self, auto_project_setup: bool = True) -> None:
        await (await self._get_db_manager_async()).init(auto_project_setup=auto_project_setup)
        return None


    async def create_project(self, name: str, lang: str = "en", auto_project_setup: bool = True) -> None:
        await (await self._get_db_manager_async()).create_project(name=name, lang=lang, auto_project_setup=auto_project_setup)
        return None


    async def delete_project(self, name: str) -> None:
        await (await self._get_db_manager_async()).delete_project(name=name)
        return None


    async def get_current_project(self) -> asyncpg.Record | None:
        return await (await self._get_db_manager_async()).get_current_project()


    async def set_current_project(self, name: str) -> None:
        await (await self._get_db_manager_async()).set_current_project(name=name)
        return None


    async def reset_current_project(self) -> None:
        await (await self._get_db_manager_async()).reset_current_project()
        return None


    async def list_all_projects(self) -> list[asyncpg.Record]:
        async with postgres_conn_async(user=self._superuser, password=self._superuser_password, dbname=self._hub_db, host=self._db_host) as conn:
            return await conn.fetch(f"""
                SELECT name
                FROM "{HubDBTables.Projects.value}"
            """)


    async def insert_trps(self, data: dict[str, Any]) -> None:
        async with postgres_conn_async(self._tfs_user, password=self._tfs_password, dbname=(await self.get_current_project()).get("name", None), host=self._db_host) as conn:
            await (await self._get_db_manager_async()).insert_trps(conn=conn, data=data)
        return None
