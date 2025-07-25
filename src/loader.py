import datetime
import asyncio
import dask.dataframe as dd
import pandas as pd
from typing import Any
from pydantic.types import PositiveInt


from utils import GlobalDefinitions
from brokers import DBBroker
from db_manager import postgres_conn



#Simple synchronous batch loader
class BatchStreamLoader:

    def __init__(self, db_broker: DBBroker):
        self._data: dd.DataFrame
        self._db_broker: DBBroker


    def _get_stream(self, sql: str, batch_size: PositiveInt):
        with postgres_conn() as conn:
            with conn.cursor() as cursor:
                cursor.stream(sql)







#NOTE TO BE USED TO LOAD AND TRANSFORM DATA TO BE COMPATIBLE WITH THE ML SECTION




































