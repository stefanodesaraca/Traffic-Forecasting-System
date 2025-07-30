import datetime
import asyncio
import dask.dataframe as dd
import pandas as pd
from typing import Any
from pydantic.types import PositiveInt


from utils import GlobalDefinitions
from brokers import DBBroker



#Simple synchronous batch loader
class BatchStreamLoader:

    def __init__(self, db_broker: DBBroker):
        self._data: dd.DataFrame
        self._db_broker: DBBroker = db_broker


    def get_volume(self,
                   batch_size: PositiveInt,
                   trp_list_filter: list[str] | None = None,
                   road_category_filter: list[str] | None = None
                   ) -> dd.DataFrame:
        df_partitions = []
        with self._db_broker.get_stream(sql=f"""
            SELECT 
                v.trp_id AS trp_id,
                v.volume AS volume,
                v.coverage AS coverage,
                v.is_mice AS is_mice,
                v.zoned_dt_iso AS zoned_dt_iso,
                t.road_category AS road_category
            FROM Volume v JOIN TrafficRegistrationPoints t ON v.trp_id = t.id
            WHERE {"v.trp_id = ANY(%s)" if trp_list_filter else "1=1"}
            AND {"t.road_category = ANY(%s)" if road_category_filter else "1=1"}
            ORDER BY zoned_dt_iso
        """, filters=tuple(f for f in [trp_list_filter, road_category_filter]), batch_size=batch_size, row_factory="tuple_row") as stream_cursor:
        # WARNING: the order of the filters in the list within the list comprehension must be the same as the order of conditions inside the sql query
            while rows := stream_cursor.fetchmany(500):
                df_partitions.append(dd.from_pandas(pd.DataFrame.from_records(rows)))

        return dd.concat(dfs=df_partitions, axis=1).sort_values(ascending=True, by=["zoned_dt_iso"])


    def get_mean_speed(self,
                       batch_size: PositiveInt,
                       trp_list_filter: list[str] | None = None,
                       road_category_filter: list[str] | None = None
                       ) -> dd.DataFrame:
        df_partitions = []
        with self._db_broker.get_stream(sql=f"""
            SELECT 
                ms.trp_id AS trp_id,
                ms.mean_speed AS mean_speed,
                ms.percentile_85 AS percentile_85,
                ms.coverage AS coverage,
                ms.is_mice AS is_mice,
                ms.zoned_dt_iso AS zoned_dt_iso,
                t.road_category AS road_category
           FROM MeanSpeed ms JOIN TrafficRegistrationPoints t ON ms.trp_id = t.id
           WHERE {"ms.trp_id = ANY(%s)" if trp_list_filter else "1=1"}
           AND {"t.road_category = ANY(%s)" if road_category_filter else "1=1"}
           ORDER BY zoned_dt_iso
        """, filters=tuple(f for f in [trp_list_filter, road_category_filter]), batch_size=batch_size, row_factory="tuple_row") as stream_cursor:
            # WARNING: the order of the filters in the list within the list comprehension must be the same as the order of conditions inside the sql query
            while rows := stream_cursor.fetchmany(500):
                df_partitions.append(dd.from_pandas(pd.DataFrame.from_records(rows)))

        return dd.concat(dfs=df_partitions, axis=1).sort_values(ascending=True, by=["zoned_dt_iso"])







#NOTE TO BE USED TO LOAD AND TRANSFORM DATA TO BE COMPATIBLE WITH THE ML SECTION




































