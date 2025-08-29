import datetime
from typing import Any, Generator
import dask.dataframe as dd
import numpy as np
import pandas as pd
from pydantic.types import PositiveInt

from definitions import GlobalDefinitions, ProjectTables
from brokers import DBBroker
from utils import to_pg_array



#Simple synchronous batch loader
class BatchStreamLoader:

    def __init__(self, db_broker: DBBroker):
        self._data: dd.DataFrame
        self._db_broker: DBBroker = db_broker
        self._dask_partition_size: str = GlobalDefinitions.DEFAULT_DASK_DF_PARTITION_SIZE


    def _load_from_stream(self, stream: Generator[Any, None, None], df_partitions_size: PositiveInt) -> dd.DataFrame:
        df_partitions = []
        # WARNING: the order of the filters in the list within the list comprehension must be the same as the order of conditions inside the sql query
        batch = []
        for row in stream:
            batch.append(row)
            if len(batch) >= df_partitions_size:
                df_partitions.append(dd.from_pandas(pd.DataFrame.from_records(batch), npartitions=1))
                batch = []

        # Adding remaining rows
        if batch:
            df_partitions.append(dd.from_pandas(pd.DataFrame.from_records(batch), npartitions=1))

        # If df_partitions has only 1 element inside then there aren't any dataframes to concatenate since there's only one, so just return it
        if len(df_partitions) == 1:
            return df_partitions[0].repartition(partition_size=self._dask_partition_size)
        return dd.concat(dfs=df_partitions, axis=0).repartition(partition_size=self._dask_partition_size)


    def get_volume(self,
                   batch_size: PositiveInt = 50000,
                   trp_list_filter: list[str] | None = None,
                   road_category_filter: list[str] | None = None,
                   county_ids_filter: list[str] | None = None,
                   limit: PositiveInt | None = None,
                   split_cyclical_features: bool = False,
                   year: bool = True,
                   encoded_cyclical_features: bool = False,
                   is_covid_year: bool = False,
                   is_mice: bool = False,
                   zoned_dt_start: datetime.datetime | None = None,
                   zoned_dt_end: datetime.datetime | None = None,
                   trp_lat: bool | None = False,
                   trp_lon: bool | None = False,
                   sort_by_date: bool = True,
                   sort_ascending: bool = True,
                   df_partitions_size: PositiveInt = 100000
                   ) -> dd.DataFrame:
        return self._load_from_stream(stream=self._db_broker.get_stream(sql=f"""
            SELECT 
                v.trp_id AS trp_id,
                v.{GlobalDefinitions.VOLUME} AS {GlobalDefinitions.VOLUME},
                v.coverage AS coverage,
                {"v.is_mice AS is_mice," if is_mice else ""}
                v.zoned_dt_iso AS zoned_dt_iso
                {",EXTRACT(YEAR FROM zoned_dt_iso) as year" if year else ""}
                {f",t.lat AS lat" if trp_lat else ""}
                {f",t.lon AS lon" if trp_lon else ""}
            {'''
                ,
                EXTRACT(DAY FROM zoned_dt_iso) AS day_of_month,
                EXTRACT(HOUR FROM zoned_dt_iso) AS hour_of_day,
                EXTRACT(MONTH FROM zoned_dt_iso) AS month_of_year,
                EXTRACT(WEEK FROM zoned_dt_iso) AS week_of_year
            ''' if split_cyclical_features else ""}
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
            if encoded_cyclical_features else ""}
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
            AND {f'''"zoned_dt_iso" >= '{str(zoned_dt_start)}'::timestamptz''' if zoned_dt_start else "1=1"}
            AND {f'''"zoned_dt_iso" <= '{str(zoned_dt_end)}'::timestamptz''' if zoned_dt_end else "1=1"}
            AND {f"t.county_number = ANY(%s)" if county_ids_filter else "1=1"}
            {f'''
            ORDER BY "zoned_dt_iso" {"ASC" if sort_ascending else "DESC"}
            ''' if sort_by_date else ""
            }
            {f"LIMIT {limit}" if limit else ""}
        """, filters=tuple(to_pg_array(f) for f in [trp_list_filter, road_category_filter, county_ids_filter] if f), batch_size=batch_size, row_factory="dict_row"), df_partitions_size=df_partitions_size)
        # The ORDER BY (descending order) is necessary since in time series forecasting the order of the records is fundamental


    def get_mean_speed(self,
                       batch_size: PositiveInt = 50000,
                       trp_list_filter: list[str] | None = None,
                       road_category_filter: list[str] | None = None,
                       county_ids_filter: list[str] | None = None,
                       limit: PositiveInt | None = None,
                       split_cyclical_features: bool = False,
                       year: bool = True,
                       encoded_cyclical_features: bool = False,
                       is_covid_year: bool = False,
                       is_mice: bool = False,
                       zoned_dt_start: datetime.datetime | None = None,
                       zoned_dt_end: datetime.datetime | None = None,
                       trp_lat: bool | None = False,
                       trp_lon: bool | None = False,
                       sort_by_date: bool = True,
                       sort_ascending: bool = True,
                       df_partitions_size: PositiveInt = 100000
                       ) -> dd.DataFrame:
        return self._load_from_stream(self._db_broker.get_stream(sql=f"""
             SELECT 
                 ms.trp_id AS trp_id,
                 ms.{GlobalDefinitions.MEAN_SPEED} AS {GlobalDefinitions.MEAN_SPEED},
                 ms.percentile_85 AS percentile_85,
                 ms.coverage AS coverage,
                 {"ms.is_mice AS is_mice," if is_mice else ""}
                 ms.zoned_dt_iso AS zoned_dt_iso
                {",EXTRACT(YEAR FROM zoned_dt_iso) as year" if year else ""}
                {f",t.lat AS lat" if trp_lat else ""}
                {f",t.lon AS lon" if trp_lon else ""}
            {'''
                ,
                EXTRACT(DAY FROM zoned_dt_iso) AS day_of_month,
                EXTRACT(HOUR FROM zoned_dt_iso) AS hour_of_day,
                EXTRACT(MONTH FROM zoned_dt_iso) AS month_of_year,
                EXTRACT(WEEK FROM zoned_dt_iso) AS week_of_year''' 
            if split_cyclical_features else ""}
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
            if encoded_cyclical_features else ""}
            {'''
            ,
            CASE 
                WHEN EXTRACT(YEAR FROM zoned_dt_iso) IN (2020, 2021, 2022) THEN TRUE
                    ELSE FALSE
                END AS is_covid_year''' 
            if is_covid_year else ""
            }
            FROM "{ProjectTables.MeanSpeed.value}" ms JOIN "{ProjectTables.TrafficRegistrationPoints.value}" t ON ms.trp_id = t.id
            WHERE {"ms.trp_id = ANY(%s)" if trp_list_filter else "1=1"}
            AND {"t.road_category = ANY(%s)" if road_category_filter else "1=1"}
            AND {f'''"zoned_dt_iso" >= '{str(zoned_dt_start)}'::timestamptz''' if zoned_dt_start else "1=1"}
            AND {f'''"zoned_dt_iso" <= '{str(zoned_dt_end)}'::timestamptz''' if zoned_dt_end else "1=1"}
            AND {f"t.county_number = ANY(%s)" if county_ids_filter else "1=1"}
            {f'''
            ORDER BY "zoned_dt_iso" {"ASC" if sort_ascending else "DESC"}''' 
            if sort_by_date else ""
            }            
            {f"LIMIT {limit}" if limit else ""}
        """, filters=tuple(to_pg_array(f) for f in [trp_list_filter, road_category_filter, county_ids_filter] if f), batch_size=batch_size, row_factory="dict_row"), df_partitions_size=df_partitions_size)
        # The ORDER BY (descending order) is necessary since in time series forecasting the order of the records is fundamental


    def get_nodes(self,
                  batch_size: PositiveInt = 50000,
                  node_ids_filter: list[str] | None = None,
                  link_ids_filter: list[str] | None = None,
                  limit: PositiveInt | None = None,
                  df_partitions_size: PositiveInt = 100000
                  ) -> dd.DataFrame:
        return self._load_from_stream(self._db_broker.get_stream(sql=f"""
            SELECT 
                "node_id",
                "type",
                ST_AsText("geom") AS geom,
                "road_node_ids",
                "is_roundabout",
                "number_of_incoming_links",
                "number_of_outgoing_links",
                "number_of_undirected_links",
                "legal_turning_movements",
                "road_system_references",
                "raw_properties"
            FROM "{ProjectTables.RoadGraphNodes.value}"
            WHERE {f'''"road_node_ids" && ANY(%s)'''
                    if node_ids_filter else "1=1"
            }
            AND {f'''"connected_traffic_link_ids" && ANY(%s)'''
                    if link_ids_filter else "1=1"
            }
            {f"LIMIT {limit}" if limit else ""}
            ;
        """, filters=tuple(to_pg_array(f) for f in [node_ids_filter, link_ids_filter] if f), batch_size=batch_size, row_factory="dict_row"), df_partitions_size=df_partitions_size)


    def get_links(self,
                  batch_size: PositiveInt = 50000,
                  link_id_filter: list[str] | None = None,
                  road_category_filter: list[str] | None = None,
                  municipality_ids_filter: list[str] | None = None,
                  county_ids_filter: list[str] | None = None,
                  node_ids_filter: list[str] | None = None,
                  link_ids_filter: list[str] | None = None,
                  has_only_public_transport_lanes_filter: bool | None = False,
                  limit: PositiveInt | None = None,
                  df_partitions_size: PositiveInt = 100000) -> dd.DataFrame:
        return self._load_from_stream(self._db_broker.get_stream(sql=f"""
            SELECT 
                rl.link_id,
                rl.type,
                ST_AsText("geom") AS geom,
                rl.year_applies_to,
                rl.candidate_ids,
                rl.road_system_references,
                rl.road_category,
                rl.road_placements,
                rl.functional_road_class,
                rl.function_class,
                rl.start_traffic_node_id,
                rl.end_traffic_node_id,
                rl.subsumed_traffic_node_ids,
                rl.road_link_ids,
                rl.highest_speed_limit,
                rl.lowest_speed_limit,
                rl.max_lanes,
                rl.min_lanes,
                rl.has_only_public_transport_lanes,
                rl.length,
                rl.traffic_direction_wrt_metering_direction,
                rl.is_norwegian_scenic_route,
                rl.is_ferry_route,
                rl.is_ramp,
                rl.traffic_volumes,
                rl.urban_ratio,
                rl.number_of_establishments,
                rl.number_of_employees,
                rl.number_of_inhabitants,
                rl.has_anomalies,
                rl.anomalies,
                rl.raw_properties
                {''',ARRAY_AGG(m.municipality_id) AS municipality_ids''' if municipality_ids_filter else ""}
                {''',ARRAY_AGG(c.county_id) AS county_ids''' if county_ids_filter else ""}
            FROM "{ProjectTables.RoadGraphLinks.value}" rl 
            {f'LEFT JOIN "{ProjectTables.RoadLink_Municipalities.value}" m ON rl.link_id = m.link_id' if municipality_ids_filter else ""}
            {f'LEFT JOIN "{ProjectTables.RoadLink_Counties.value}" c ON rl.link_id = c.link_id' if county_ids_filter else ""}
            WHERE {f'''"rl.link_id" = ANY(%s)'''
                if link_id_filter else "1=1"
            }
            AND {f'''"rl.road_category" = ANY(%s)'''
                if road_category_filter else "1=1"
            }
            AND {f'''"municipality_id" = ANY(%s)'''
                if municipality_ids_filter else "1=1"
            }
            AND {f'''"county_id" = ANY(%s)'''
                if county_ids_filter else "1=1"
            }
            AND {f'''"road_node_ids" && ANY(%s)'''
                if node_ids_filter else "1=1"
            }
            AND {f'''"road_link_ids" && ANY(%s)'''
                if link_ids_filter else "1=1"
            }
            AND {f'"has_only_public_transport_lanes" = FALSE' 
                if has_only_public_transport_lanes_filter is False else "1=1"
            }
            {'''
            GROUP BY
                rl.id,
                rl.link_id,
                rl.type,
                rl.geom,
                rl.year_applies_to,
                rl.candidate_ids,
                rl.road_system_references,
                rl.road_category,
                rl.road_placements,
                rl.functional_road_class,
                rl.function_class,
                rl.start_traffic_node_id,
                rl.end_traffic_node_id,
                rl.subsumed_traffic_node_ids,
                rl.road_link_ids,
                rl.highest_speed_limit,
                rl.lowest_speed_limit,
                rl.max_lanes,
                rl.min_lanes,
                rl.has_only_public_transport_lanes,
                rl.length,
                rl.traffic_direction_wrt_metering_direction,
                rl.is_norwegian_scenic_route,
                rl.is_ferry_route,
                rl.is_ramp,
                rl.traffic_volumes,
                rl.urban_ratio,
                rl.number_of_establishments,
                rl.number_of_employees,
                rl.number_of_inhabitants,
                rl.has_anomalies,
                rl.anomalies,
                rl.raw_properties
            ''' if municipality_ids_filter or county_ids_filter else ""}
            {f"LIMIT {limit}" if limit else ""}
            ;
            """, filters=tuple(to_pg_array(f) for f in [
                    link_id_filter,
                    road_category_filter,
                    municipality_ids_filter,
                    county_ids_filter,
                    node_ids_filter,
                    link_ids_filter
                ] if f), batch_size=batch_size, row_factory="dict_row"), df_partitions_size=df_partitions_size)
