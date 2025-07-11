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


    async def _clean_async(self, mice_past_window: PositiveInt) -> None:
        print("Shape before MICE: ", len(self.data), len(self.data.columns))
        print("Number of zeros before MICE: ", len(self.data[self.data["volume"] == 0]))


        #TODO FOR TESTING PURPOSES
        context = pd.DataFrame(await self.db_broker.execute_sql(sql=f"""SELECT *
                                                                        FROM Volume
                                                                        ORDER BY zoned_dt_iso DESC
                                                                        LIMIT {mice_past_window};
                                                                     """))
        print(context)
        print(context.shape)
        print(context.describe())



        self.data = pd.concat([
                                self.data[["trp_id", "is_mice", "zoned_dt_iso"]],
                                await asyncio.to_thread(pd.DataFrame, await self.db_broker.execute_sql(sql=f"""SELECT *
                                                                                                                     FROM Volume
                                                                                                                     ORDER BY zoned_dt_iso DESC
                                                                                                                     LIMIT {mice_past_window};
                                                                                                                 """)), #Extracting data from the past to improve MICE regression model performances
                                await asyncio.to_thread(self._impute_missing_values,self.data.drop(columns=["trp_id", "is_mice", "zoned_dt_iso"], axis=1), r="gamma")
                                ], axis=1)

        print("Shape after MICE: ", len(self.data), len(self.data.columns))
        print("Number of zeros after MICE: ", len(self.data[self.data["volume"] == 0]))
        print("Number of negative values (after MICE): ", len(self.data[self.data["volume"] < 0]))

        self.data["volume"] = self.data["volume"].astype("int") #Re-converting volume to int after MICE

        return None


    async def ingest(self, payload: dict[str, Any]):
        self.data = payload
        ...


        #TODO FOR INSERTS USE "ON CONFLICT (id) DO NOTHING"


























