import asyncpg

from db_config import DBConfig
from db_manager import DBManager, postgres_conn


class DBBroker:

    def __init__(self, db_user: str, db_password: str, db_name: str, db_host: str):
        self._db_user = db_user
        self._db_password = db_password
        self._db_name = db_name
        self._db_host = db_host
        self._db_manager = DBManager(self._db_user, self._db_password, self._db_name, self._db_host)


    def send(self, data):
        #TODO INSTANTIATE PIPELINE AND SEND TO DB
        ...































