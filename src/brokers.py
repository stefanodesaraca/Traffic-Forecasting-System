from typing import Any
import asyncpg
from pyparsing import withClass

from db_config import DBConfig
from db_manager import DBManager, postgres_conn


class DBBroker:

    def __init__(self, db_user: str, db_password: str, db_name: str, db_host: str):
        self._db_user = db_user
        self._db_password = db_password
        self._db_name = db_name
        self._db_host = db_host
        self._db_manager = DBManager(self._db_user, self._db_password, self._db_name, self._db_host)


    async def execute_sql(self, sql: str) -> Any:
        async with postgres_conn() as conn:
            async with conn.transaction():
                return conn.execute(sql)









    async def send(self, data):
        #TODO INSTANTIATE PIPELINE AND SEND TO DB
        ...































