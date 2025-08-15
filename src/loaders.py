import dask.dataframe as dd
import pandas as pd
from pydantic.types import PositiveInt

from brokers import DBBroker
from utils import to_pg_array
from db_config import ProjectTables



#Simple synchronous batch loader
class BatchStreamLoader:

    def __init__(self, db_broker: DBBroker):
        self._data: dd.DataFrame
        self._db_broker: DBBroker = db_broker


    def get_volume(self,
                   batch_size: PositiveInt,
                   trp_list_filter: list[str] | None = None,
                   road_category_filter: list[str] | None = None,
                   limit: PositiveInt | None = None,
                   split_cyclical_features: bool = False,
                   encoded_cyclical_features: bool = False,
                   is_covid_year: bool = False,
                   ) -> dd.DataFrame:
        df_partitions = []
        stream = self._db_broker.get_stream(sql=f"""
            SELECT 
                v.trp_id AS trp_id,
                v.volume AS volume,
                v.coverage AS coverage,
                v.is_mice AS is_mice,
                v.zoned_dt_iso AS zoned_dt_iso,
                t.road_category AS road_category
            {'''
                ,
                EXTRACT(DAY FROM zoned_dt_iso) AS day_of_month,
                EXTRACT(HOUR FROM zoned_dt_iso) AS hour_of_day,
                EXTRACT(MONTH FROM zoned_dt_iso) AS month_of_year,
                EXTRACT(WEEK FROM zoned_dt_iso) AS week_of_year
            ''' if split_cyclical_features else ""
            }
            {'''    
                ,
                COS(2 * PI() * EXTRACT(DAY FROM zoned_dt_iso) / 31) AS day_cos,
                SIN(2 * PI() * EXTRACT(DAY FROM zoned_dt_iso) / 31) AS day_sin,
            
                COS(2 * PI() * EXTRACT(HOUR FROM zoned_dt_iso) / 24) AS hour_cos,
                SIN(2 * PI() * EXTRACT(HOUR FROM zoned_dt_iso) / 24) AS hour_sin,
            
                COS(2 * PI() * EXTRACT(MONTH FROM zoned_dt_iso) / 12) AS month_cos,
                SIN(2 * PI() * EXTRACT(MONTH FROM zoned_dt_iso) / 12) AS month_sin,
            
                COS(2 * PI() * EXTRACT(WEEK FROM zoned_dt_iso) / 53) AS week_cos,
                SIN(2 * PI() * EXTRACT(WEEK FROM zoned_dt_iso) / 53) AS week_sin'''
                if encoded_cyclical_features else ""
            }
            {'''
            ,
            CASE 
                WHEN EXTRACT(YEAR FROM zoned_dt_iso) IN (2020, 2021, 2022) THEN TRUE
                    ELSE FALSE
                END AS is_covid_year
            ''' if is_covid_year else ""
            }
            FROM "{ProjectTables.Volume.value}" v JOIN "{ProjectTables.TrafficRegistrationPoints.value}" t ON v.trp_id = t.id
            WHERE {"v.trp_id = ANY(%s)" if trp_list_filter else "1=1"}
            AND {"t.road_category = ANY(%s)" if road_category_filter else "1=1"}
            ORDER BY "zoned_dt_iso" DESC
            {f"LIMIT {limit}" if limit else ""}
        """, filters=tuple(to_pg_array(f) for f in [trp_list_filter, road_category_filter] if f), batch_size=batch_size, row_factory="tuple_row")
        # The ORDER BY (descending order) is necessary since in time series forecasting the order of the records is fundamental

        # WARNING: the order of the filters in the list within the list comprehension must be the same as the order of conditions inside the sql query
        batch = []
        for row in stream:
            batch.append(row)
            if len(batch) >= 100000:
                df_partitions.append(dd.from_pandas(pd.DataFrame.from_records(batch), npartitions=1))
                batch = []

        # Adding remaining rows
        if batch:
            df_partitions.append(dd.from_pandas(pd.DataFrame.from_records(batch), npartitions=1))

        return dd.concat(dfs=df_partitions, axis=1).sort_values(ascending=True, by=["zoned_dt_iso"])


    def get_mean_speed(self,
                       batch_size: PositiveInt,
                       trp_list_filter: list[str] | None = None,
                       road_category_filter: list[str] | None = None,
                       limit: PositiveInt | None = None,
                       split_cyclical_features: bool = False,
                       encoded_cyclical_features: bool = False,
                       is_covid_year: bool = False
                       ) -> dd.DataFrame:
        df_partitions = []
        stream = self._db_broker.get_stream(sql=f"""
             SELECT 
                 ms.trp_id AS trp_id,
                 ms.mean_speed AS mean_speed,
                 ms.percentile_85 AS percentile_85,
                 ms.coverage AS coverage,
                 ms.is_mice AS is_mice,
                 ms.zoned_dt_iso AS zoned_dt_iso,
                 t.road_category AS road_category
            {'''
                ,
                EXTRACT(DAY FROM zoned_dt_iso) AS day_of_month,
                EXTRACT(HOUR FROM zoned_dt_iso) AS hour_of_day,
                EXTRACT(MONTH FROM zoned_dt_iso) AS month_of_year,
                EXTRACT(WEEK FROM zoned_dt_iso) AS week_of_year
            ''' if split_cyclical_features else ""
            }
            {'''    
                ,
                COS(2 * PI() * EXTRACT(DAY FROM zoned_dt_iso) / 31) AS day_cos,
                SIN(2 * PI() * EXTRACT(DAY FROM zoned_dt_iso) / 31) AS day_sin,
            
                COS(2 * PI() * EXTRACT(HOUR FROM zoned_dt_iso) / 24) AS hour_cos,
                SIN(2 * PI() * EXTRACT(HOUR FROM zoned_dt_iso) / 24) AS hour_sin,
            
                COS(2 * PI() * EXTRACT(MONTH FROM zoned_dt_iso) / 12) AS month_cos,
                SIN(2 * PI() * EXTRACT(MONTH FROM zoned_dt_iso) / 12) AS month_sin,
            
                COS(2 * PI() * EXTRACT(WEEK FROM zoned_dt_iso) / 53) AS week_cos,
                SIN(2 * PI() * EXTRACT(WEEK FROM zoned_dt_iso) / 53) AS week_sin'''
                if encoded_cyclical_features else ""
            }
            {'''
            ,
            CASE 
                WHEN EXTRACT(YEAR FROM zoned_dt_iso) IN (2020, 2021, 2022) THEN TRUE
                    ELSE FALSE
                END AS is_covid_year
            ''' if is_covid_year else ""
            }
            FROM "{ProjectTables.MeanSpeed.value}" ms JOIN "{ProjectTables.TrafficRegistrationPoints.value}" t ON ms.trp_id = t.id
            WHERE {"ms.trp_id = ANY(%s)" if trp_list_filter else "1=1"}
            AND {"t.road_category = ANY(%s)" if road_category_filter else "1=1"}
            ORDER BY "zoned_dt_iso" DESC
            {f"LIMIT {limit}" if limit else ""}
        """, filters=tuple(to_pg_array(f) for f in [trp_list_filter, road_category_filter] if f), batch_size=batch_size, row_factory="tuple_row")
        # The ORDER BY (descending order) is necessary since in time series forecasting the order of the records is fundamental

        # WARNING: the order of the filters in the list within the list comprehension must be the same as the order of conditions inside the sql query
        batch = []
        for row in stream:
            batch.append(row)
            if len(batch) >= 100000:
                df_partitions.append(dd.from_pandas(pd.DataFrame.from_records(batch), npartitions=1))
                batch = []

        # Adding remaining rows
        if batch:
            df_partitions.append(dd.from_pandas(pd.DataFrame.from_records(batch), npartitions=1))

        return dd.concat(dfs=df_partitions, axis=1).sort_values(ascending=True, by=["zoned_dt_iso"])
