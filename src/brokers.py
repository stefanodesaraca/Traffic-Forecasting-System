from typing import Any
import asyncpg

from db_config import DBConfig
from exceptions import WrongSQLStatement, MissingDataException
from db_manager import postgres_conn


class DBBroker:
    #The DBBroker returns records directly from the db without any transformation to ensure
    # a standard query output format which is then treated differently by each function or method that requires db data

    def __init__(self, db_user: str, db_password: str, db_name: str, db_host: str):
        self._db_user = db_user
        self._db_password = db_password
        self._db_name = db_name
        self._db_host = db_host


    async def send_sql(self, sql: str, many: bool = False, many_values: list[tuple[Any, ...]] | None = None) -> Any:
        async with postgres_conn(user=self._db_user, password=self._db_password, name=self._db_name, host=self._db_host) as conn:
            async with conn.transaction():
                if any(sql.startswith(prefix) for prefix in ["SELECT", "select"]):
                    return await conn.fetch(sql)
                elif any(sql.startswith(prefix) for prefix in ["INSERT", "UPDATE", "DELETE", "insert", "update", "delete"]):
                    if many and many_values:
                        return conn.executemany(sql, many_values)
                    elif many and not many_values:
                        raise MissingDataException("Missing data to insert")
                    return await conn.execute(sql)
                else:
                    raise WrongSQLStatement("The SQL query isn't correct")


    async def get_trp_ids(self) -> list[asyncpg.Record]:
        async with postgres_conn(user=self._db_user, password=self._db_password, name=self._db_name, host=self._db_host) as conn:
            async with conn.transaction():
                return await conn.fetch("""SELECT id FROM TrafficRegistrationPoints;""")


    async def get_trp_ids_by_road_category(self) -> list[asyncpg.Record]:
        async with postgres_conn(user=self._db_user, password=self._db_password, name=self._db_name, host=self._db_host) as conn:
            async with conn.transaction():
                return await conn.fetch("""SELECT id FROM TrafficRegistrationPoints
                                           GROUP BY road_category;
                ;""")






























