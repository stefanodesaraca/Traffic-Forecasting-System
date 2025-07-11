from typing import Any
import asyncpg

from db_config import DBConfig
from db_manager import postgres_conn


class DBBroker:
    #The DBBroker returns records directly from the db without any transformation to ensure
    # a standard query output format which is then treated differently by each function or method that requires db data

    def __init__(self, db_user: str, db_password: str, db_name: str, db_host: str):
        self._db_user = db_user
        self._db_password = db_password
        self._db_name = db_name
        self._db_host = db_host


    async def execute_sql(self, sql: str) -> Any:
        async with postgres_conn(user=self._db_user, password=self._db_password, name=self._db_name, host=self._db_host) as conn:
            async with conn.transaction():
                return await conn.execute(sql)


    async def get_trp_ids(self) -> Any:
        async with postgres_conn(user=self._db_user, password=self._db_password, name=self._db_name, host=self._db_host) as conn:
            async with conn.transaction():
                return await conn.execute("""SELECT id FROM TrafficRegistrationPoints;""")







    async def send(self, data):
        #TODO INSTANTIATE PIPELINE AND SEND TO DB
        ...































