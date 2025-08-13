import math
import datetime
import asyncio
import dask.dataframe as dd
import pandas as pd
from typing import Any, cast
from pydantic import BaseModel
from pydantic.types import PositiveInt

from sklearn.linear_model import Lasso, GammaRegressor, QuantileRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklego.meta import ZeroInflatedRegressor

from utils import GlobalDefinitions, localize_datetimes_async
from db_config import ProjectTables

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)



class RegressorTypes(BaseModel):
    lasso: str = "lasso"
    gamma: str = "gamma"
    quantile: str = "quantile"

    class Config:
        frozen=True



class ExtractionPipelineMixin:

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



class VolumeExtractionPipeline(ExtractionPipelineMixin):

    def __init__(self, db_broker_async: Any, data: dict[str, Any] | None = None):
        self.data: dict[str, Any] | pd.DataFrame | dd.DataFrame | None = data
        self._db_broker_async: Any = db_broker_async


    @staticmethod
    async def _get_missing(zoned_datetimes: set[datetime.datetime]) -> set[datetime.datetime]:
        return await asyncio.to_thread(lambda: set(pd.date_range(min(zoned_datetimes), max(zoned_datetimes))).difference(zoned_datetimes)) # Finding all zoned datetimes which should exist (calculated above in all_dts), but that aren't withing the ones available.


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
            ) for m in await self._get_missing(set(localize_datetimes_async((node["node"]["from"] for node in data), timezone_literal="Europe/Oslo"))))

        all((
            by_hour["trp_id"].append(trp_id),
            by_hour["volume"].append(edge["node"]["total"]["volumeNumbers"]["volume"] if edge["node"]["total"]["volumeNumbers"] is not None else None),  # In some cases the volumeNumbers key could have null as value, so the "volume" key won't be present. In that case we'll directly insert None as value with an if statement
            by_hour["coverage"].append(edge["node"]["total"]["coverage"]["percentage"] or None),  # For less recent data it's possible that sometimes coverage can be null, so we'll address this problem like so
            by_hour["is_mice"].append(False if edge["node"]["total"]["volumeNumbers"] else True),  # For less recent data it's possible that sometimes coverage can be null, so we'll address this problem like so
            by_hour["zoned_dt_iso"].append(datetime.datetime.fromisoformat(edge["node"]["from"])))
        for edge in data)

        return await asyncio.to_thread(lambda: pd.DataFrame(by_hour).sort_values(by=["zoned_dt_iso"], ascending=True))


    async def _clean_async(self, data: pd.DataFrame, mice_past_window: PositiveInt, fields: list[str] = GlobalDefinitions.VOLUME_INGESTION_FIELDS.value) -> pd.DataFrame | None:

        #If all columns which need to be fed to the MICE algorithm are None then just skip this batch
        if data[GlobalDefinitions.MICE_COLS.value].isna().all().all():
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
                            ORDER BY zoned_dt_iso DESC
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

        data = mice_treated_data[await self._get_dfs_diff_mask(mice_treated_data, data, comparison_cols=list(set(fields).difference(set(GlobalDefinitions.MICE_COLS.value))))]
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
            print(f"\033[91mNo data for TRP: {trp_id}\033[0m")
            return None

        self.data = await self._clean_async(data=self.data, mice_past_window=max(1, math.ceil(len(self.data) / 2)), fields=fields)
        if self.data is None:
            print(f"\033[91mNo data for TRP: {trp_id}\033[0m")
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



class MeanSpeedExtractionPipeline(ExtractionPipelineMixin):

    def __init__(self, db_broker_async: Any, data: dict[str, Any] | None = None):
        self.data: dict[str, Any] | pd.DataFrame | dd.DataFrame | None = data
        self._db_broker_async: Any = db_broker_async


    async def _parse_mean_speed_async(self, speeds: pd.DataFrame) -> pd.DataFrame | dd.DataFrame | None:
        if speeds.empty:
            return None

        speeds["coverage"] = speeds["coverage"].replace(",", ".", regex=True).astype("float") * 100
        speeds["mean_speed"] = speeds["mean_speed"].replace(",", ".", regex=True).astype("float")
        speeds["percentile_85"] = speeds["percentile_85"].replace(",", ".", regex=True).astype("float")

        speeds["zoned_dt_iso"] = speeds["date"] + "T" + speeds["hour_start"] + GlobalDefinitions.NORWEGIAN_UTC_TIME_ZONE.value
        speeds = speeds.drop(columns=["date"])

        try:
            print("Shape before MICE:", speeds.shape)
            print("Number of zeros before MICE:", len(speeds[speeds["mean_speed"] == 0]))
            print("Negative values (mean speed) before MICE:", len(speeds[speeds["mean_speed"] < 0]))

            speeds = await asyncio.to_thread(pd.concat, [
                speeds[["trp_id", "zoned_dt_iso"]],
                await self._impute_missing_values(speeds.drop(columns=["trp_id", "zoned_dt_iso"]), r="gamma")
            ], axis=1) #TODO IF THIS DOESN'T WORK TRY AS IT WAS BEFORE ... .to_thread(lambda: pd.concat(...)

            print("Shape after MICE:", speeds.shape, "\n")
            print("Number of zeros after MICE:", len(speeds[speeds["mean_speed"] == 0]))
            print("Negative values (mean speed) after MICE:", len(speeds[speeds["mean_speed"] < 0]))

        except ValueError as e:
            print(f"\033[91mValueError: {e}. Skipping...\033[0m")
            return None

        return speeds


    async def ingest(self, fp: str, fields: list[str]) -> None:
        self.data = await self._parse_mean_speed_async(
            await asyncio.to_thread(pd.read_csv, fp, sep=";", **{"engine": "c"})) #TODO OR dd.read_csv()
        if not self.data:
            pass
        if fields:
            self.data = self.data[[fields]]
        await self._db_broker_async.send_sql_async(f"""
            INSERT INTO "{ProjectTables.MeanSpeed.value}" ({', '.join(fields)})
            VALUES ({', '.join(f'${nth_field}' for nth_field in range(1, len(fields) + 1))})
            ON CONFLICT ON CONSTRAINT unique_mean_speed_per_trp_and_time DO NOTHING;
        """, many=True, many_values=list(self.data.itertuples(index=False, name=None)))

        return None















