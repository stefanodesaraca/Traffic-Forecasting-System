import datetime
import asyncio
import dask.dataframe as dd
import pandas as pd
import numpy
from typing import Any, Generator
from pydantic import BaseModel
from pydantic.types import PositiveInt, PositiveFloat

from sklearn.linear_model import Lasso, GammaRegressor, QuantileRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklego.meta import ZeroInflatedRegressor

from tfs_utils import GlobalDefinitions
from brokers import DBBroker

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)



class RegressorTypes(BaseModel):
    lasso = "lasso"
    gamma = "gamma"
    quantile = "quantile"

    class Config:
        frozen=True




class ExtractionPipelineMixin:

    # Executing multiple imputation to get rid of NaNs using the MICE method (Multiple Imputation by Chained Equations)
    @staticmethod
    def _impute_missing_values(data: pd.DataFrame | dd.DataFrame, r: str = "gamma") -> pd.DataFrame:
        """
        This function should only be supplied with numerical columns-only dataframes

        Parameters:
            data: the data with missing values
            r: the regressor kind. Has to be within a specific list or regressors available
        """
        if r not in RegressorTypes.model_fields.keys():
            raise ValueError(f"Regressor type '{r}' is not supported. Must be one of: {RegressorTypes.model_fields.keys()}")

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

        return pd.DataFrame(mice_imputer.fit_transform(data), columns=data.columns) # Fitting the imputer and processing all the data columns except the date one #TODO BOTTLENECK. MAYBE USE POLARS LazyFrame or PyArrow?


    @staticmethod
    async def _is_empty_async(data: dict[Any, Any]) -> bool:
        return len(data) == 0



class VolumeExtractionPipeline(ExtractionPipelineMixin):

    def __init__(self, db_broker: DBBroker, data: dict[str, Any] | None = None):
        self.data: dict[str, Any] | pd.DataFrame | dd.DataFrame | None = data
        self.db_broker: DBBroker = db_broker


    @staticmethod
    def _get_missing(zoned_datetimes: set[datetime.datetime]) -> set[datetime.datetime]:
        return set(pd.date_range(min(zoned_datetimes), max(zoned_datetimes))).difference(zoned_datetimes) #Finding all zoned datetimes which should exist (calculated above in all_dts), but that aren't withing the ones available.


    async def _parse_by_hour_async(self) -> pd.DataFrame | dd.DataFrame | None:

        trp_id = self.data["trafficData"]["trafficRegistrationPoint"]["id"]
        self.data = self.data["trafficData"]["volume"]["byHour"]["edges"]

        if await self._is_empty_async(self.data):
            print(f"\033[91mNo data found for TRP: {trp_id}\033[0m\n\n")
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
            ) for m in self._get_missing(set(datetime.datetime.fromisoformat(node["node"]["from"]) for node in self.data)))

        all((
            by_hour["trp_id"].append(trp_id),
            by_hour["volume"].append(edge["node"]["total"]["volumeNumbers"]["volume"] if edge["node"]["total"]["volumeNumbers"] is not None else None),  # In some cases the volumeNumbers key could have null as value, so the "volume" key won't be present. In that case we'll directly insert None as value with an if statement
            by_hour["coverage"].append(edge["node"]["total"]["coverage"]["percentage"] or None),  # For less recent data it's possible that sometimes coverage can be null, so we'll address this problem like so
            by_hour["is_mice"].append(False if edge["node"]["total"]["volumeNumbers"] else True),  # For less recent data it's possible that sometimes coverage can be null, so we'll address this problem like so
            by_hour["zoned_dt_iso"].append(datetime.datetime.fromisoformat(edge["node"]["from"])))
        for edge in self.data)

        return pd.DataFrame(by_hour).sort_values(by=["zoned_dt_iso"], ascending=True)


    async def _clean_async(self, mice_past_window: PositiveInt) -> pd.DataFrame:
        print("Shape before MICE: ", len(self.data), len(self.data.columns))
        print("Number of zeros before MICE: ", len(self.data[self.data["volume"] == 0]))


        #TODO FOR TESTING PURPOSES
        context = pd.DataFrame(await self.db_broker.send_sql(sql=f"""SELECT *
                                                                        FROM Volume
                                                                        ORDER BY zoned_dt_iso DESC
                                                                        LIMIT {mice_past_window};
                                                                     """))
        print(context)
        print(context.shape)
        print(context.describe())



        self.data = pd.concat([
                                self.data[["trp_id", "is_mice", "zoned_dt_iso"]],
                                await asyncio.to_thread(pd.DataFrame, await self.db_broker.send_sql(sql=f"""SELECT *
                                                                                                                     FROM Volume
                                                                                                                     ORDER BY zoned_dt_iso DESC
                                                                                                                     LIMIT {mice_past_window};
                                                                                                                 """)), #Extracting data from the past to improve MICE regression model performances
                                await asyncio.to_thread(self._impute_missing_values,self.data.drop(columns=["trp_id", "is_mice", "zoned_dt_iso"], axis=1), r="gamma")
                                ], axis=1)
        #Duplicates aren't a problem since PostgresSQL inserts are set to do nothing on conflict

        print("Shape after MICE: ", len(self.data), len(self.data.columns))
        print("Number of zeros after MICE: ", len(self.data[self.data["volume"] == 0]))
        print("Number of negative values (after MICE): ", len(self.data[self.data["volume"] < 0]))

        self.data["volume"] = self.data["volume"].astype("int") #Re-converting volume to int after MICE

        return self.data


    async def ingest(self, payload: dict[str, Any], fields: list[str] | None = None) -> None:
        self.data = payload
        self.data = await self._clean_async(self.data)
        if fields:
            self.data = self.data[[fields]]

        await self.db_broker.send_sql(f"""
            INSERT INTO Volume ({', '.join(fields)})
            VALUES ({', '.join(f'${nth_field}' for nth_field in range(1, len(fields)+1))})
            ON CONFLICT ON CONSTRAINT unique_volume_per_trp_and_time DO NOTHING;
        """, many=True)

        return None



class MeanSpeedExtractionPipeline(ExtractionPipelineMixin):

    def __init__(self, db_broker: DBBroker, data: dict[str, Any] | None = None):
        self.data: dict[str, Any] | pd.DataFrame | dd.DataFrame | None = data
        self.db_broker: DBBroker = db_broker


    async def _parse_speeds_async(self, speeds: pd.DataFrame) -> pd.DataFrame | dd.DataFrame | None:
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
                self._impute_missing_values(speeds.drop(columns=["trp_id", "zoned_dt_iso"]), r="gamma")
            ], axis=1) #TODO IF THIS DOESN'T WORK TRY AS IT WAS BEFORE ... .to_thread(lambda: pd.concat(...)

            print("Shape after MICE:", speeds.shape, "\n")
            print("Number of zeros after MICE:", len(speeds[speeds["mean_speed"] == 0]))
            print("Negative values (mean speed) after MICE:", len(speeds[speeds["mean_speed"] < 0]))

        except ValueError as e:
            print(f"\033[91mValueError: {e}. Skipping...\033[0m")
            return None

        return speeds


    async def ingest(self, fp: str, fields: list[str]) -> None:
        self.data = await self._parse_speeds_async(
            await asyncio.to_thread(pd.read_csv, fp, sep=";", **{"engine": "c"}))

        if not self.data:
            ...


    async def clean_async(self, fp: str) -> pd.DataFrame | dd.DataFrame | None:
        # Using to_thread since this is a CPU-bound operation which would otherwise block the event loop until it's finished executing
        #TODO TO GET ALL mean_speed FILES JSUT USE os.listdir() UPSTREAM
        ...












