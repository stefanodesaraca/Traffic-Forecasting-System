from typing import Any, Literal, Generator, LiteralString, Sequence, Mapping
from contextlib import contextmanager
import psycopg
from pydantic.types import PositiveInt
import asyncpg

from exceptions import WrongSQLStatementError, MissingDataError
from definitions import GlobalDefinitions, HubDBTables, ProjectTables, ProjectViews, RowFactories
from dbmanager import AIODBManager, postgres_conn_async, postgres_conn
from utils import to_pg_array, cached, cached_async


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
                        raise MissingDataError("Missing data to insert")
                    return await conn.execute(sql)
                else:
                    raise WrongSQLStatementError("The SQL query isn't correct")


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


    @cached_async()
    async def get_road_categories_async(self, name_as_key: bool = False) -> dict[str, str]:
        async with postgres_conn_async(user=self._db_user, password=self._db_password, dbname=self._db_name, host=self._db_host) as conn:
            async with conn.transaction():
                if not name_as_key:
                    return {row['id']: row['name'] for row in (await conn.fetch(f'SELECT id, name FROM "{ProjectTables.RoadCategories.value}";'))}
                return {row['name']: row['id'] for row in (await conn.fetch(f'SELECT id, name FROM "{ProjectTables.RoadCategories.value}";'))}



class DBBroker:
    #The SyncDBBroker returns records directly from the db without any transformation to ensure
    # a standard query output format which is then treated differently by each function or method that requires db data

    def __init__(self, db_user: str, db_password: str, db_name: str, db_host: str):
        self._db_user = db_user
        self._db_password = db_password
        self._db_name = db_name
        self._db_host = db_host


    @contextmanager
    def PostgresConnectionCursor(self, conn: psycopg.Connection, query: LiteralString, row_factory: Literal["tuple_row", "dict_row"] = "dict_row", params: Sequence | Mapping[str, Any] | None = None) -> Generator[Any, Any, None]:
        try:
            with conn.cursor(row_factory=RowFactories.factories[row_factory]) as cursor:
                cursor.execute(query, params=params)
                yield cursor
        finally:
            pass


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
                        raise MissingDataError("Missing data to insert")
                    return cur.execute(sql, execute_args) if execute_args else cur.execute(sql)
                else:
                    raise WrongSQLStatementError("The SQL query isn't correct")


    def get_stream(self, sql: str, batch_size: PositiveInt, filters: tuple[Any] | tuple[str, ...] | tuple[list[Any], ...], row_factory: Literal["tuple_row", "dict_row"] = "dict_row") -> Generator:
        if "SELECT" not in sql:
            raise WrongSQLStatementError("Cannot return a data stream from a non selective statement (SELECT)")
        with postgres_conn(user=self._db_user, password=self._db_password, dbname=self._db_name, host=self._db_host, row_factory=row_factory) as conn:
            for row in conn.cursor().stream(query=sql, params=filters, size=batch_size):
                yield row


    def get_trp_ids(self, road_category_filter: list[str] | None = None) -> list[tuple[Any, ...] | dict[Any, ...]]:
        with postgres_conn(user=self._db_user, password=self._db_password, dbname=self._db_name, host=self._db_host) as conn:
            with self.PostgresConnectionCursor(query=f"""
                    SELECT "id" FROM "{ProjectTables.TrafficRegistrationPoints.value}"
                    {'WHERE "road_category" = ANY(%s)' if road_category_filter else ""};
                 """, params=tuple(to_pg_array(f) for f in [road_category_filter] if f), conn=conn) as cur:
                return cur.fetchall()


    def get_trp_ids_by_road_category(self, has_volumes: bool | None = None, has_mean_speed: bool | None = None) -> dict[Any, ...]:
        with postgres_conn(user=self._db_user, password=self._db_password, dbname=self._db_name, host=self._db_host) as conn:
            with self.PostgresConnectionCursor(query=f"""SELECT json_object_agg("road_category", "ids") AS result
                                                         FROM (
                                                             SELECT "road_category", json_agg("trp_id" ORDER BY "trp_id") AS ids
                                                             FROM "{ProjectViews.TrafficRegistrationPointsMetadataView.value}"
                                                             WHERE {f"has_{GlobalDefinitions.VOLUME} = TRUE" if has_volumes else "1=1"}
                                                             AND {f"has_{GlobalDefinitions.MEAN_SPEED} = TRUE" if has_mean_speed else "1=1"}
                                                             GROUP BY "road_category"
                                                         ) AS sub;
                                                      """, conn=conn) as cur:
                return cur.fetchone()["result"]
            #Output example:
            #{
            #    "E": ["17684V2460285", "17900V111222"],
            #    "R": ["03375V625405"]
            #}

    @cached()
    def get_volume_date_boundaries(self, trp_id_filter: list[str] | None = None) -> dict[str, Any]:
        with postgres_conn(user=self._db_user, password=self._db_password, dbname=self._db_name, host=self._db_host) as conn:
            with self.PostgresConnectionCursor(query=f"""
                    SELECT
                        MIN(zoned_dt_iso) AS {GlobalDefinitions.VOLUME}_start_date,
                        MAX(zoned_dt_iso) AS {GlobalDefinitions.VOLUME}_end_date
                    FROM "{ProjectTables.Volume.value}"
                    {f'''WHERE "trp_id" = ANY(%s)''' if trp_id_filter else ""}
                    ;
                    """, conn=conn, params=tuple(to_pg_array(f) for f in [list(trp_id_filter)])) as cur:
                result = cur.fetchone()
                return {"min": result[f"{GlobalDefinitions.VOLUME}_start_date"], "max": result[f"{GlobalDefinitions.VOLUME}_end_date"]} #Respectively: min and max

    @cached()
    def get_mean_speed_date_boundaries(self, trp_id_filter: list[str] | None = None) -> dict[str, Any]:
        with postgres_conn(user=self._db_user, password=self._db_password, dbname=self._db_name, host=self._db_host) as conn:
            with self.PostgresConnectionCursor(f"""
                    SELECT
                        MIN(zoned_dt_iso) AS {GlobalDefinitions.MEAN_SPEED}_start_date,
                        MAX(zoned_dt_iso) AS {GlobalDefinitions.MEAN_SPEED}_end_date
                    FROM "{ProjectTables.MeanSpeed.value}"
                    {f"WHERE m.trp_id = '{trp_id_filter}'" if trp_id_filter else ""}
                    ;
                    """, conn=conn, params=tuple(to_pg_array(f) for f in [list(trp_id_filter)])) as cur:
                result = cur.fetchone()
                return {"min": result[f"{GlobalDefinitions.MEAN_SPEED}_start_date"], "max": result[f"{GlobalDefinitions.MEAN_SPEED}_end_date"]} #Respectively: min and max


    def get_all_trps_metadata(self, has_volume_filter: bool = False, has_mean_speed_filter: bool = False) -> dict[str, dict[str, Any]]:
        with postgres_conn(user=self._db_user, password=self._db_password, dbname=self._db_name, host=self._db_host) as conn:
            with self.PostgresConnectionCursor(query=f"""
                        SELECT jsonb_object_agg("trp_id", to_jsonb(t) - 'trp_id') AS trp_metadata
                        FROM (
                            SELECT *
                            FROM "{ProjectViews.TrafficRegistrationPointsMetadataView.value}"
                            WHERE {f"has_{GlobalDefinitions.VOLUME} = TRUE" if has_volume_filter else "1=1"}
                            AND {f"has_{GlobalDefinitions.MEAN_SPEED} = TRUE" if has_mean_speed_filter else "1=1"}
                        ) AS t;
                    """, conn=conn) as cur:
                return cur.fetchone().get("trp_metadata")


    def get_trp_metadata(self, trp_id: str) -> dict[str, Any]:
        with postgres_conn(user=self._db_user, password=self._db_password, dbname=self._db_name, host=self._db_host) as conn:
            with self.PostgresConnectionCursor(query=f"""
                        SELECT TO_JSONB(t) AS metadata
                        FROM "{ProjectViews.TrafficRegistrationPointsMetadataView.value}" t
                        WHERE "trp_id" = %s;
                    """, conn=conn, params=[trp_id]) as cur:
                return cur.fetchone()["metadata"]

    #TODO TRANSFER THIS OPERATION DIRECTLY INTO main.py AND USE send_sql()
    def get_base_model_objects(self) -> list[dict[str, Any]]:
        with postgres_conn(user=self._db_user, password=self._db_password, dbname=self._db_name, host=self._db_host) as conn:
            with self.PostgresConnectionCursor(query=f"""
                            SELECT 
                                m.id as id,
                                m.name as name, 
                                m.base_params AS params,
                                m.best_{GlobalDefinitions.VOLUME}_gridsearch_params_idx AS best_{GlobalDefinitions.VOLUME}_params_idx,
                                m.best_{GlobalDefinitions.MEAN_SPEED}_gridsearch_params_idx AS best_{GlobalDefinitions.MEAN_SPEED}_params_idx,
                                bm.pickle_object AS pickle_object
                            FROM
                                "{ProjectTables.MLModels.value}" m
                            JOIN
                                "{ProjectTables.BaseModels.value}" bm ON m.id = bm.id;
                            """, conn=conn) as cur:
                return cur.fetchall()


    def get_trained_model_objects(self, target: str, road_category: str) -> list[dict[str, Any]]:
        with postgres_conn(user=self._db_user, password=self._db_password, dbname=self._db_name, host=self._db_host) as conn:
            with self.PostgresConnectionCursor(query=f"""
                            SELECT 
                                tm.id as id,
                                mm.name as name, 
                                tm.target,
                                tm.road_category,
                                tm.pickle_object AS pickle_object
                            FROM
                                "{ProjectTables.MLModels.value}" mm
                            JOIN
                                "{ProjectTables.TrainedModels.value}" tm ON mm.id=tm.id
                            WHERE tm.target = '{target}'
                            AND tm.road_category = '{road_category}';
                            """, conn=conn) as cur:
                return cur.fetchall()


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
