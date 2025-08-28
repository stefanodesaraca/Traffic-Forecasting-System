import math
from itertools import pairwise
import datetime
from datetime import timedelta
import json
from pathlib import Path
import datetime
from zoneinfo import ZoneInfo
import asyncio
import aiofiles
import numpy as np
import dask.dataframe as dd
import pandas as pd
from typing import Any, Literal, cast
from pydantic import BaseModel
from pydantic.types import PositiveInt

from sklearn.linear_model import Lasso, GammaRegressor, QuantileRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklego.meta import ZeroInflatedRegressor

from dask_ml.preprocessing import MinMaxScaler, LabelEncoder

import geojson
from shapely.geometry import shape
from shapely import wkt

from definitions import GlobalDefinitions, ProjectTables, ProjectConstraints
from utils import ZScore, cos_encoder, sin_encoder, check_target, split_by_target, get_n_items_from_gen


pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)


class RegressorTypes(BaseModel):
    lasso: str = "lasso"
    gamma: str = "gamma"
    quantile: str = "quantile"

    class Config:
        frozen=True



class IngestionPipelineMixin:

    # Executing multiple imputation to get rid of NaNs using the MICE method (Multiple Imputation by Chained Equations)
    @staticmethod
    async def _impute_missing_values(data: pd.DataFrame | dd.DataFrame, r: str = "gamma") -> pd.DataFrame:
        """
        This function should only be supplied with numerical columns-only dataframes

        Parameters:
            data: the data with missing values
            r: the regressor kind. Has to be within a specific list or regressors available
        """
        if r not in cast(dict, RegressorTypes.model_fields).keys(): #cast is used to silence static type checker warnings
            raise ValueError(f"Regressor type '{r}' is not supported. Must be one of: {cast(dict, RegressorTypes.model_fields).keys()}")

        reg = None
        if r == "lasso":
            reg = ZeroInflatedRegressor(
                regressor=Lasso(random_state=100, fit_intercept=True),
                classifier=DecisionTreeClassifier(random_state=100)
            )  # Using Lasso regression (L1 Penalization) to get better results in case of non-informative columns present in the data (coverage data, because their values all the same)
        elif r == "gamma":
            reg = ZeroInflatedRegressor(
                regressor=GammaRegressor(fit_intercept=True, verbose=0),
                classifier=DecisionTreeClassifier(random_state=100)
            )  # Using Gamma regression to address for the zeros present in the data (which will need to be predicted as well)
        elif r == "quantile":
            reg = ZeroInflatedRegressor(
                regressor=QuantileRegressor(fit_intercept=True),
                classifier=DecisionTreeClassifier(random_state=100)
            )

        mice_imputer = IterativeImputer(
            estimator=reg,
            random_state=100,
            verbose=0,
            imputation_order="roman",
            initial_strategy="mean"
        )  # Imputation order is set to arabic so that the imputations start from the right (so from the traffic volume columns)

        return await asyncio.to_thread(lambda: pd.DataFrame(mice_imputer.fit_transform(data), columns=data.columns)) # Fitting the imputer and processing all the data columns except the date one #TODO BOTTLENECK. MAYBE USE POLARS LazyFrame or PyArrow?


    @staticmethod
    async def _is_empty_async(data: dict[Any, Any]) -> bool:
        return len(data) == 0



class VolumeIngestionPipeline(IngestionPipelineMixin):

    def __init__(self, db_broker_async: Any, data: dict[str, Any] | None = None):
        self.data: dict[str, Any] | pd.DataFrame | dd.DataFrame | None = data
        self._db_broker_async: Any = db_broker_async


    @staticmethod
    async def _get_missing(zoned_datetimes: set[datetime.datetime]) -> set[datetime.datetime]:
        return await asyncio.to_thread(lambda: set(pd.date_range(min(zoned_datetimes), max(zoned_datetimes), ambiguous=True, nonexistent="shift_forward")).difference(zoned_datetimes)) # Finding all zoned datetimes which should exist (calculated above in all_dts), but that aren't withing the ones available.


    @staticmethod
    async def _get_dfs_diff_mask(df1: pd.DataFrame, df2: pd.DataFrame, comparison_cols: list[str]):
        # Create a mask for rows in data that also exist in contextd
        return await asyncio.to_thread(lambda: df1[comparison_cols].isin(df2[comparison_cols].to_dict(orient='list')).all(axis=1))


    async def _parse_by_hour_async(self, data: dict[str, Any]) -> pd.DataFrame | dd.DataFrame | None:

        trp_id = data["trafficData"]["trafficRegistrationPoint"]["id"]
        data = data["trafficData"]["volume"]["byHour"]["edges"]

        if await self._is_empty_async(data):
            return None

        by_hour = {
            "trp_id": [],
            "volume": [],
            "coverage": [],
            "is_mice": [],
            "zoned_dt_iso": []
        }

        all((
            by_hour["trp_id"].append(trp_id),
            by_hour["volume"].append(None),
            by_hour["coverage"].append(None),
            by_hour["is_mice"].append(True),
            by_hour["zoned_dt_iso"].append(m)
            ) for m in await self._get_missing(set(datetime.datetime.fromisoformat(node["node"]["from"]).replace(tzinfo=ZoneInfo("Europe/Oslo")) for node in data)))

        all((
            by_hour["trp_id"].append(trp_id),
            by_hour["volume"].append(edge["node"]["total"]["volumeNumbers"]["volume"] if edge["node"]["total"]["volumeNumbers"] is not None else None),  # In some cases the volumeNumbers key could have null as value, so the "volume" key won't be present. In that case we'll directly insert None as value with an if statement
            by_hour["coverage"].append(edge["node"]["total"]["coverage"]["percentage"] or None),  # For less recent data it's possible that sometimes coverage can be null, so we'll address this problem like so
            by_hour["is_mice"].append(False if edge["node"]["total"]["volumeNumbers"] else True),  # For less recent data it's possible that sometimes coverage can be null, so we'll address this problem like so
            by_hour["zoned_dt_iso"].append(datetime.datetime.fromisoformat(edge["node"]["from"]).replace(tzinfo=ZoneInfo("Europe/Oslo"))))
        for edge in data)

        return await asyncio.to_thread(lambda: pd.DataFrame(by_hour).sort_values(by=["zoned_dt_iso"], ascending=True))


    async def _clean_async(self, data: pd.DataFrame, mice_past_window: PositiveInt, fields: list[str] = GlobalDefinitions.VOLUME_INGESTION_FIELDS) -> pd.DataFrame | None:

        #If all columns which need to be fed to the MICE algorithm are None then just skip this batch
        if data[GlobalDefinitions.MICE_COLS].isna().all().all():
            return None

        #print("Shape before MICE: ", data.shape)
        #print("Number of zeros before MICE: ", len(data[data["volume"] == 0]))


        #TODO MAYBE IN THE FUTURE WE'LL USE POLARS LazyFrame FOR ALL OF THIS


        contextd = await asyncio.to_thread((await asyncio.to_thread(
            pd.concat, [
                data,
                (await asyncio.to_thread(pd.DataFrame,
                    await self._db_broker_async.send_sql_async(sql=f"""
                            SELECT {', '.join(fields)}
                            FROM "{ProjectTables.Volume.value}"
                            ORDER BY "zoned_dt_iso" DESC
                            LIMIT {mice_past_window};
                        """), columns=fields
                    )),
            ], axis=0, ignore_index=True
        )).sort_values, by=["zoned_dt_iso"])  # Extracting data from the past to improve MICE regression model performances

        mice_treated_data = await asyncio.to_thread(pd.concat, [
            contextd[["trp_id", "is_mice", "zoned_dt_iso"]],
            await self._impute_missing_values(contextd.drop(columns=["trp_id", "is_mice", "zoned_dt_iso"], axis=1), r="gamma")
        ], axis=1)
        #Once having completed the MICE part, we'll concatenate back the columns which were dropped before (since they can't be fed to the MICE algorithm)

        data = mice_treated_data[await self._get_dfs_diff_mask(mice_treated_data, data, comparison_cols=list(set(fields).difference(set(GlobalDefinitions.MICE_COLS))))]
        #Getting the intersection between the data that has been treated with MICE and the original data records.
        #By doing so, we'll get all the records which already had data AND the rows which were supposed to be MICEd filled with synthetic data

        #print("Shape after MICE: ", data.shape)
        #print("Number of zeros after MICE: ", len(data[data["volume"] == 0]))
        #print("Number of negative values (after MICE): ", len(data[data["volume"] < 0]))

        data["volume"] = data["volume"].astype("int") #Re-converting volume to int after MICE
        data["coverage"] = data["coverage"].round(3) #Re-converting volume to int after MICE

        return data


    async def ingest(self, payload: dict[str, Any], trp_id: str, fields: list[str] | None = None) -> None:
        self.data = await self._parse_by_hour_async(payload)
        if self.data is None:
            print(f"\033[91mNo data to parse for TRP: {trp_id}\033[0m")
            return None

        self.data = await self._clean_async(data=self.data, mice_past_window=max(1, math.ceil(len(self.data) / 2)), fields=fields)
        if self.data is None:
            print(f"\033[91mNo data available for TRP: {trp_id}\033[0m")
            return None

        if fields:
            self.data = self.data[fields]

        await self._db_broker_async.send_sql_async(f"""
            INSERT INTO "{ProjectTables.Volume.value}" ({', '.join(fields)})
            VALUES ({', '.join(f'${nth_field}' for nth_field in range(1, len(fields) + 1))})
            ON CONFLICT DO NOTHING;
        """, many=True, many_values=list(self.data.itertuples(index=False, name=None)))

        print(f"Successfully inserted TRP: {trp_id} data")

        return None



class MeanSpeedIngestionPipeline(IngestionPipelineMixin):

    def __init__(self, db_broker_async: Any, data: dict[str, Any] | None = None):
        self.data: dict[str, Any] | pd.DataFrame | dd.DataFrame | None = data
        self._db_broker_async: Any = db_broker_async


    async def _parse_mean_speed_async(self, speeds: pd.DataFrame) -> pd.DataFrame | dd.DataFrame | None:
        if speeds.empty:
            return None

        speeds["coverage"] = speeds["coverage"].replace(",", ".", regex=True).astype("float") * 100
        speeds["mean_speed"] = speeds["mean_speed"].replace(",", ".", regex=True).astype("float")
        speeds["percentile_85"] = speeds["percentile_85"].replace(",", ".", regex=True).astype("float")

        speeds["zoned_dt_iso"] = pd.to_datetime(speeds["date"] + "T" + speeds["hour_start"] + GlobalDefinitions.NORWEGIAN_UTC_TIME_ZONE)
        speeds = speeds.drop(columns=["date", "hour_start"])

        speeds["is_mice"] = speeds["mean_speed"].notna()

        try:
            #print("Shape before MICE:", speeds.shape)
            #print("Number of zeros before MICE:", len(speeds[speeds["mean_speed"] == 0]))
            #print("Negative values (mean speed) before MICE:", len(speeds[speeds["mean_speed"] < 0]))

            speeds = await asyncio.to_thread(pd.concat, [
                speeds[["trp_id", "is_mice", "zoned_dt_iso"]],
                await self._impute_missing_values(speeds.drop(columns=["trp_id", "is_mice", "zoned_dt_iso"]), r="gamma")
            ], axis=1)

            #print("Shape after MICE:", speeds.shape, "\n")
            #print("Number of zeros after MICE:", len(speeds[speeds["mean_speed"] == 0]))
            #print("Negative values (mean speed) after MICE:", len(speeds[speeds["mean_speed"] < 0]))

        except ValueError as e:
            print(f"\033[91mValueError: {e}. Skipping...\033[0m")
            return None

        return speeds


    async def ingest(self, fp: str | Path, fields: list[str]) -> None:
        print(f"""Starting TRP: {fp} cleaning""")
        self.data = await self._parse_mean_speed_async(
            await asyncio.to_thread(pd.read_csv, fp, sep=";", **{"engine": "c"})) #TODO OR dd.read_csv()
        if self.data is None:
            return None
        elif self.data.empty:
            return None

        if fields:
            self.data = self.data[fields]
        await self._db_broker_async.send_sql_async(f"""
            INSERT INTO "{ProjectTables.MeanSpeed.value}" ({', '.join(f'"{f}"' for f in fields)})
            VALUES ({', '.join(f'${nth_field}' for nth_field in range(1, len(fields) + 1))})
            ON CONFLICT ON CONSTRAINT "{ProjectConstraints.UNIQUE_MEAN_SPEED_PER_TRP_AND_TIME.value}" DO NOTHING;
        """, many=True, many_values=list(self.data.itertuples(index=False, name=None)))

        print(f"""Ended TRP: {fp} cleaning""")

        return None



class RoadGraphObjectsIngestionPipeline:

    def __init__(self, db_broker_async: Any):
        self._db_broker_async: Any = db_broker_async


    @staticmethod
    async def _load_geojson_async(fp: str) -> dict[str, Any]:
        async with aiofiles.open(fp, "r", encoding="utf-8") as geo:
            return geojson.loads(await geo.read())


    async def ingest_nodes(self, fp: str | Path, batch_size: PositiveInt = 200, n_async_jobs: PositiveInt = 10) -> None:
        semaphore = asyncio.Semaphore(n_async_jobs)

        nodes = (await self._load_geojson_async(fp=fp)).get("features", [])
        ing_query = f"""
            INSERT INTO "{ProjectTables.RoadGraphNodes.value}" (
                "node_id",
                "type",
                "geom",
                "road_node_ids",
                "is_roundabout",
                "number_of_incoming_links",
                "number_of_outgoing_links",
                "number_of_undirected_links",
                "legal_turning_movements",
                "road_system_references",
                "raw_properties"
            ) VALUES (
                $1, $2, ST_GeomFromText($3, {GlobalDefinitions.COORDINATES_REFERENCE_SYSTEM}), $4, $5, $6, $7, $8, $9, $10, $11
            )
            ON CONFLICT DO NOTHING;
        """

        batches = get_n_items_from_gen(gen=((
            feature.get("properties").get("id"),
            feature.get("type"),
            shape(feature.get("geometry")).wkt, # Convertion of the geometry to WKT for PostGIS compatibility (so that PostGIS can read the actual shape of the feature)
            feature.get("properties").get("roadNodeIds"),
            feature.get("properties").get("isRoundabout"),
            feature.get("properties").get("numberOfIncomingLinks"),
            feature.get("properties").get("numberOfOutgoingLinks"),
            feature.get("properties").get("numberOfUndirectedLinks"),
            json.dumps(feature.get("properties").get("legalTurningMovements", [])),
            feature.get("properties").get("roadSystemReferences"),
            json.dumps(feature)  # keep raw properties for flexibility
        ) for feature in nodes), n=batch_size)

        async def limited_ingest(batch: list[tuple[Any]]) -> None:
            async with semaphore:
                return await self._db_broker_async.send_sql_async(sql=ing_query, many=True, many_values=batch)

        await asyncio.gather(*(limited_ingest(batch) for batch in batches))

        return None


    async def ingest_links(self, fp: str | Path, batch_size: PositiveInt = 200, n_async_jobs: PositiveInt = 10) -> None:
        semaphore = asyncio.Semaphore(n_async_jobs)
        road_categories = await self._db_broker_async.get_road_categories_async(enable_cache=True, name_as_key=True)

        links = (await self._load_geojson_async(fp=fp)).get("features", [])
        links_ing_query = f"""
                    INSERT INTO "{ProjectTables.RoadGraphLinks.value}" (
                        "link_id",
                        "type",
                        "geom",
                        "year_applies_to",
                        "candidate_ids",
                        "road_system_references",
                        "road_category",
                        "road_placements",
                        "functional_road_class",
                        "function_class",
                        "start_traffic_node_id",
                        "end_traffic_node_id",
                        "subsumed_traffic_node_ids",
                        "road_link_ids",
                        "road_node_ids",
                        "highest_speed_limit",
                        "lowest_speed_limit",
                        "max_lanes",
                        "min_lanes",
                        "has_only_public_transport_lanes",
                        "length",
                        "traffic_direction_wrt_metering_direction",
                        "is_norwegian_scenic_route",
                        "is_ferry_route",
                        "is_ramp",
                        "traffic_volumes",
                        "urban_ratio",
                        "number_of_establishments",
                        "number_of_employees",
                        "number_of_inhabitants",
                        "has_anomalies",
                        "anomalies",
                        "raw_properties"
                    ) VALUES (
                        $1, $2, ST_GeomFromText($3, {GlobalDefinitions.COORDINATES_REFERENCE_SYSTEM}), $4, $5, $6, $7, $8, $9, $10,
                        $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23,
                        $24, $25, $26, $27, $28, $29, $30, $31, $32, $33
                    )
                    ON CONFLICT DO NOTHING;
                """

        link_municipalities_ing_query = f"""
                    INSERT INTO "{ProjectTables.RoadLink_Municipalities.value}" (
                        "link_id",
                        "municipality_id"
                    )
                    VALUES ($1, $2);
        """
        link_counties_ing_query = f"""
                    INSERT INTO "{ProjectTables.RoadLink_Counties.value}" (
                        "link_id",
                        "county_id"
                    )
                    VALUES ($1, $2);
        """
        link_toll_stations_ing_query = f"""
                    INSERT INTO "{ProjectTables.RoadLink_TollStations.value}" (
                        "link_id",
                        "toll_station_id"
                    )
                    VALUES ($1, $2);
        """
        link_trps_ing_query = f"""
                    INSERT INTO "{ProjectTables.RoadLink_TrafficRegistrationPoints.value}" (
                        "link_id",
                        "trp_id"
                    )
                    VALUES ($1, $2);
        """

        link_batches = get_n_items_from_gen(gen=((
            feature.get("properties").get("id"),
            feature.get("type"),
            shape(feature.get("geometry")).wkt, # Convert geometry to WKT with shapely's shape() and then extracting the wkt
            feature.get("properties").get("yearAppliesTo"),
            feature.get("properties").get("candidateIds"),
            feature.get("properties").get("roadSystemReferences"),
            road_categories.get(feature.get("properties").get("roadCategory"), None),
            json.dumps(feature.get("properties").get("roadPlacements", [])),
            feature.get("properties").get("functionalRoadClass"),
            feature.get("properties").get("functionClass"),
            feature.get("properties").get("startTrafficNodeId"),
            feature.get("properties").get("endTrafficNodeId"),
            feature.get("properties").get("subsumedTrafficNodeIds"),
            feature.get("properties").get("roadLinkIds"),
            feature.get("properties").get("roadNodeIds"),
            feature.get("properties").get("highestSpeedLimit"),
            feature.get("properties").get("lowestSpeedLimit"),
            feature.get("properties").get("maxLanes"),
            feature.get("properties").get("minLanes"),
            feature.get("properties").get("hasOnlyPublicTransportLanes"),
            feature.get("properties").get("length"),
            feature.get("properties").get("trafficDirectionWrtMeteringDirection"),
            feature.get("properties").get("isNorwegianScenicRoute"),
            feature.get("properties").get("isFerryRoute"),
            feature.get("properties").get("isRamp"),
            json.dumps(feature.get("properties").get("trafficVolumes", [])),
            feature.get("properties").get("urbanRatio"),
            feature.get("properties").get("numberOfEstablishments"),
            feature.get("properties").get("numberOfEmployees"),
            feature.get("properties").get("numberOfInhabitants"),
            feature.get("properties").get("hasAnomalies"),
            json.dumps(feature.get("properties").get("anomalies", [])),
            json.dumps(feature)
        ) for feature in links), n=batch_size)
        link_municipalities_matches = get_n_items_from_gen(gen=((
            feature.get("properties").get("id"),
            feature.get("properties").get("municipalityIds"),
        ) for feature in links), n=batch_size)
        link_counties_matches = get_n_items_from_gen(gen=((
            feature.get("properties").get("id"),
            feature.get("properties").get("countyIds")
        ) for feature in links), n=batch_size)
        link_toll_stations_matches = get_n_items_from_gen(gen=((
            feature.get("properties").get("id"),
            feature.get("properties").get("tollStationIds")
        ) for feature in links), n=batch_size)
        link_trps_matches = get_n_items_from_gen(gen=((
            feature.get("properties").get("id"),
            feature.get("properties").get("associatedTrpIds")
        ) for feature in links), n=batch_size)

        async def limited_ingest(batch: list[tuple[Any]]) -> None:
            async with semaphore:
                return await self._db_broker_async.send_sql_async(sql=links_ing_query, many=True, many_values=batch)

        await asyncio.gather(*(asyncio.wait_for(limited_ingest(batch), timeout=30) for batch in link_batches)) #Setting a timer so if the tasks fail for whatever reason they won't hang forever

        await asyncio.gather(*(asyncio.wait_for(limited_ingest(batch), timeout=30) for batch in link_municipalities_matches))
        await asyncio.gather(*(asyncio.wait_for(limited_ingest(batch), timeout=30) for batch in link_counties_matches))
        await asyncio.gather(*(asyncio.wait_for(limited_ingest(batch), timeout=30) for batch in link_toll_stations_matches))
        await asyncio.gather(*(asyncio.wait_for(limited_ingest(batch), timeout=30) for batch in link_trps_matches))

        return None



class MLPreprocessingPipeline:

    def __init__(self):
        self._scaler: MinMaxScaler

    @property
    def scaler(self) -> MinMaxScaler | None:
        return self._scaler or None

    @staticmethod
    def _add_lag_features(data: dd.DataFrame, target: str, lags: list[PositiveInt]) -> dd.DataFrame:
        for lag in lags:
            data[f"{target}_lag{lag}h"] = data.groupby("trp_id")[target].shift(lag)  # N hours shift
            data[f"{target}_rolling_mean_{lag}h"] = (
                data.groupby("trp_id")["volume"]
                .shift(lag)  # exclude current hour
                .rolling(window=lag, min_periods=1) # min_periods=1 ensures that the rolling statistic is computed even if fewer than *lag (value)* valid values exist.
                .mean()
                .reset_index(level=0, drop=True)  # Drop group index, align back
            )
            data[f"{target}_rolling_std_{lag}h"] = (
                data.groupby("trp_id")[target]
                .shift(lag)  # exclude current hour
                .rolling(window=lag, min_periods=1)
                .std()
                .reset_index(level=0, drop=True)  # Drop group index, align back
            )
            #IMPORTANT: we're grouping because in the dataframe there's data from multiple TRPs which have completely different values
            data[f"coverage_{target}_lag_{lag}h"] = data["coverage"] * data[f"{target}_lag_{lag}h"]

        for lag_t1, lag_t2 in pairwise(lags):
            data[f"{target}_delta_{lag_t1}h_{lag_t2}h"] = data[f"{target}_lag_{lag_t1}h"] - data[f"{target}_lag_{lag_t2}h"]
            # Delta value between the first lag and the second one per cycle. Lags are taken pairwise from the lags list, if the number of lags is odd the last element won't be included. If the lags list has only one element no delta can be calculated, so the pairwise() function will just return an empty iterator
            data[f"{target}_pct_change_{lag_t1}h_{lag_t2}h"] = (data[f"{target}_lag_{lag_t1}h"] - data[f"{target}_lag_{lag_t2}h"]) / data[f"{target}_lag_{lag_t2}h"] # Percentual change in the period between the first lag to the second
        return data.persist()


    def _scale_features(self, data: dd.DataFrame, feats: list[str], scaler: MinMaxScaler | type[callable] = MinMaxScaler) -> dd.DataFrame:
        self._scaler = scaler()
        data[feats] = self.scaler.fit_transform(data[feats])
        return data.persist()

    @staticmethod
    def _encode_categorical_features(data: dd.DataFrame, feats: list[str]) -> dd.DataFrame:
        # Using a label encoder to encode TRP IDs to include the effect of the non-independence of observations from each other inside the forecasting models
        for feat in feats:
            data[feat] = data[feat].astype("category")
        # Encode each feature
        for feat in feats:
            le = LabelEncoder(use_categorical=True)
            data[feat] = le.fit_transform(data[feat])
        return data.persist()


    def preprocess_volume(self,
                          data: dd.DataFrame,
                          lags: list[PositiveInt],
                          z_score: bool = False) -> dd.DataFrame:
        if z_score:
            data = ZScore(data, column=GlobalDefinitions.VOLUME)
        data = self._add_lag_features(data=data, target=GlobalDefinitions.VOLUME, lags=lags)
        data = self._encode_categorical_features(data=data, feats=GlobalDefinitions.ENCODED_FEATURES)
        data = self._scale_features(data=data, feats=GlobalDefinitions.VOLUME_SCALED_FEATURES, scaler=MinMaxScaler)
        data = data.sort_values(by=["trp_id", "zoned_dt_iso"], ascending=True).persist()
        return data.drop(columns=GlobalDefinitions.NON_PREDICTORS, axis=1).persist()


    def preprocess_mean_speed(self,
                              data: dd.DataFrame,
                              lags: list[PositiveInt],
                              z_score: bool = False) -> dd.DataFrame:
        if z_score:
            data = ZScore(data, column=GlobalDefinitions.MEAN_SPEED)
        data = self._add_lag_features(data=data, target=GlobalDefinitions.MEAN_SPEED, lags=lags)
        data = self._encode_categorical_features(data=data, feats=GlobalDefinitions.ENCODED_FEATURES)
        data = self._scale_features(data=data, feats=GlobalDefinitions.MEAN_SPEED_SCALED_FEATURES, scaler=MinMaxScaler)
        data = data.sort_values(by=["trp_id", "zoned_dt_iso"], ascending=True).persist()
        return data.drop(columns=GlobalDefinitions.NON_PREDICTORS, axis=1).persist()



class MLPredictionPipeline:
    def __init__(self,
                 trp_id: str,
                 road_category: str,
                 target: str,
                 db_broker: Any,
                 loader: Any,
                 preprocessing_pipeline: MLPreprocessingPipeline,
                 model: Any
                 ):
        from brokers import DBBroker
        from loaders import BatchStreamLoader
        from ml import ModelWrapper

        self._trp_id: str = trp_id
        self._road_category: str = road_category
        self._target: str = target
        self._db_broker: DBBroker = db_broker
        self._loader: BatchStreamLoader = loader
        self._pipeline: MLPreprocessingPipeline = preprocessing_pipeline
        self._model: ModelWrapper = model

        check_target(target=self._target, errors=True)


    def _get_training_records(self, training_mode: Literal[0, 1], cache_latest_dt_collection: bool = True) -> dd.DataFrame:
        """
        Parameters:
            training_mode: the training mode we want to use.
                0 - Stands for single-point training, so only the data from the TRP we want to predict future records for is used
                1 - Stands for multipoint training, where data from all TRPs of the same road category as the one we want to predict future records for is used
        """

        def get_volume_training_data_start():
            return (self._db_broker.get_volume_date_boundaries(trp_id_filter=trp_id_filter, enable_cache=cache_latest_dt_collection)["max"]
                    - timedelta(hours=((self._db_broker.get_forecasting_horizon(target=self._target) - self._db_broker.get_volume_date_boundaries(trp_id_filter=trp_id_filter, enable_cache=cache_latest_dt_collection)["max"]).days * 24) * 2))

        def get_mean_speed_training_data_start():
            return (self._db_broker.get_mean_speed_date_boundaries(trp_id_filter=trp_id_filter, enable_cache=cache_latest_dt_collection)["max"]
                    - timedelta(hours=((self._db_broker.get_forecasting_horizon(target=self._target) - self._db_broker.get_mean_speed_date_boundaries(trp_id_filter=trp_id_filter, enable_cache=cache_latest_dt_collection)["max"]).days * 24) * 2))

        trp_id_filter = (self._trp_id,) if training_mode == 0 else None
        training_functions_mapping = {
            GlobalDefinitions.VOLUME: {
                "loader": self._loader.get_volume,
                "training_data_start": get_volume_training_data_start,
                "date_boundaries": self._db_broker.get_volume_date_boundaries
            },
            GlobalDefinitions.MEAN_SPEED: {
                "loader": self._loader.get_mean_speed,
                "training_data_start": get_mean_speed_training_data_start,
                "date_boundaries": self._db_broker.get_mean_speed_date_boundaries
            }
        }
        return training_functions_mapping[self._target]["loader"](
            road_category_filter=[self._road_category] if self._road_category else None,
            trp_list_filter=trp_id_filter,
            encoded_cyclical_features=True,
            year=True,
            is_covid_year=True,
            is_mice=False,
            zoned_dt_start=training_functions_mapping[self._target]["training_data_start"](),
            zoned_dt_end=training_functions_mapping[self._target]["date_boundaries"](trp_id_filter=trp_id_filter, enable_cache=cache_latest_dt_collection)["max"]
        ).assign(is_future=False).repartition(partition_size=GlobalDefinitions.DEFAULT_DASK_DF_PARTITION_SIZE).persist()


    def _generate_future_records(self, forecasting_horizon: datetime.datetime) -> dd.DataFrame | None:
        """
        Generate records of the future to predict.

        Parameters
        ----------
        forecasting_horizon : datetime
            The target datetime which we want to predict data for and before.

        Returns
        -------
        dd.DataFrame
            A dask dataframe of empty records for future predictions.
        """

        attr = {GlobalDefinitions.VOLUME: np.nan} if self._target == GlobalDefinitions.VOLUME else {GlobalDefinitions.MEAN_SPEED: np.nan, "percentile_85": np.nan}  # TODO ADDRESS THE FACT THAT WE CAN'T PREDICT percentile_85, SO WE HAVE TO REMOVE IT FROM HERE

        last_available_data_dt = self._db_broker.get_volume_date_boundaries(trp_id_filter=tuple([self._trp_id]), enable_cache=False)["max"] if self._target == GlobalDefinitions.VOLUME else \
        self._db_broker.get_mean_speed_date_boundaries(trp_id_filter=tuple([self._trp_id]), enable_cache=False)["max"]
        rows_to_predict = ({
            "trp_id": self._trp_id,
            **attr,
            "coverage": 100.0,
            # We'll assume it's 100 since we won't know the coverage until the measurements are actually made
            "zoned_dt_iso": dt,
            "day_cos": cos_encoder(dt.day, timeframe=31),
            "day_sin": sin_encoder(dt.day, timeframe=31),
            "hour_cos": cos_encoder(dt.hour, timeframe=24),
            "hour_sin": sin_encoder(dt.hour, timeframe=24),
            "month_cos": cos_encoder(dt.month, timeframe=12),
            "month_sin": sin_encoder(dt.month, timeframe=12),
            "year": dt.year,
            "week_cos": cos_encoder(dt.isocalendar().week, timeframe=53),
            "week_sin": sin_encoder(dt.isocalendar().week, timeframe=53),
            "is_covid_year": False
        } for dt in pd.date_range(start=last_available_data_dt, end=forecasting_horizon, freq="1h"))
        # The start parameter contains the last date for which we have data available, the end one contains the target date for which we want to predict data

        return dd.from_pandas(pd.DataFrame(list(rows_to_predict))).assign(is_future=True).repartition(partition_size=GlobalDefinitions.DEFAULT_DASK_DF_PARTITION_SIZE).persist()


    def _get_future_records(self):
        past_data = self._get_training_records(
            training_mode=0,
            cache_latest_dt_collection=True
        )
        future_records = self._generate_future_records(forecasting_horizon=self._db_broker.get_forecasting_horizon(target=self._target))
        return getattr(self._pipeline, f"preprocess_{self._target}")(data=dd.concat([past_data, future_records], axis='columns').repartition(partition_size=GlobalDefinitions.DEFAULT_DASK_DF_PARTITION_SIZE), lags=[24, 36, 48, 60, 72], z_score=False)


    def start(self, trp_tuning: bool = False) -> dd.DataFrame:
        # trp_tuning defines if we want to train the already trained model on data exclusively from the TRP which we want to forecast data for to improve prediction accuracy
        scaling_mapping = {
            GlobalDefinitions.VOLUME: GlobalDefinitions.VOLUME_SCALED_FEATURES,
            GlobalDefinitions.MEAN_SPEED: GlobalDefinitions.MEAN_SPEED_SCALED_FEATURES
        }
        scaled_cols = scaling_mapping[self._target]
        data = self._get_future_records()
        if trp_tuning:
            X_tune, y_tune = split_by_target(
                data=data[data["is_future"] != True].drop(columns=["is_future"]).persist(),
                target=self._target,
                mode=1
            )
            self._model.fit(X_tune, y_tune)

        data = data[data["is_future"] != False].drop(columns=["is_future"]).persist()
        X_predict, _ = split_by_target(
            data=data,
            target=self._target,
            mode=1
        )
        data[self._target] = dd.from_array(self._model.predict(X_predict))
        data[scaled_cols] = self._pipeline.scaler.inverse_transform(data[scaled_cols])
        return data



















