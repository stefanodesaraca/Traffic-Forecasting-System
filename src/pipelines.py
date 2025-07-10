import datetime
from datetime import datetime
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

    def __init__(self, data: dict[str, Any] | None = None):
        self.data = data


    @staticmethod
    def _get_registration_datetimes(edges: dict[Any, Any]) -> set[str]:
        return set(datetime.fromisoformat(node["node"]["from"]).replace(tzinfo=None).isoformat() for node in edges) # Only keeping the datetime without the +00:00 at the end
               # Removing duplicates with set() at the end


    def _get_missing_data(self, edges: dict[Any, Any]) -> dict[str, list[str]]:

        reg_datetimes = self._get_registration_datetimes(edges)

        # The available_day_hours dict will have as key-value pairs: the day and a list with all hours which do have registrations (so that have data)
        available_day_hours = {str(datetime.fromisoformat(dt).date().isoformat()): [datetime.strptime(rd, "%Y-%m-%dT%H:%M:%S").strftime("%H") for rd in reg_datetimes]
                               for dt in (str(datetime.fromisoformat(dt).date().isoformat()) for dt in reg_datetimes)}  # These dict will have a dictionary for each day with an empty list


        # ------------------ Addressing missing days and hours problem ------------------

        missing_hours_by_day = {
            d: [h for h in (f"{i:02}" for i in range(24)) if h not in available_day_hours[d]]
            for d in available_day_hours.keys()
        }  # This dictionary comprehension goes like this: we'll create a day key with a list of hours for each day in the available days.
        # Each day's list will only include registration hours (h) which SHOULD exist, but are missing in the available dates in the data

        #Determining the missing days in the first generator: for missing_day in (1° generator)
        #Then simply creating the key-value pair in the missing_hours_by_day with all 24 hours as values of the list which includes missing hours for that day
        for missing_day in (
            str(d.date().isoformat())
            for d in pd.date_range(start=min(reg_datetimes), end=max(reg_datetimes), freq="d")
            if str(d.date().isoformat()) not in available_day_hours.keys()
        ):
            missing_hours_by_day[missing_day] = [f"{i:02}" for i in range(24)]  # If a whole day is missing we'll just create it and say that all hours of that day are missing

        return {d: l for d, l in missing_hours_by_day.items() if len(l) != 0}  # Removing elements with empty lists (the days which don't have missing hours)


    async def _parse_by_hour_async(self) -> pd.DataFrame | dd.DataFrame | None:

        trp_id = self.data["trafficData"]["trafficRegistrationPoint"]["id"]
        self.data = self.data["trafficData"]["volume"]["byHour"]["edges"]

        if await self._is_empty_async(self.data):
            print(f"\033[91mNo data found for TRP: {trp_id}\033[0m\n\n")
            return None

        by_hour_structured = {
            "trp_id": [],
            "volume": [],
            "coverage": [],
            "year": [],
            "month": [],
            "week": [],
            "day": [],
            "hour": [],
            "date": [],
        }

        missing_days_cnt = 0
        for d, mh in self._get_missing_data(self.data).items():
            for h in mh:
                by_hour_structured["trp_id"].append(trp_id)
                by_hour_structured["volume"].append(None)
                by_hour_structured["coverage"].append(None)
                by_hour_structured["year"].append(datetime.strptime(d, "%Y-%m-%d").strftime("%Y"))
                by_hour_structured["month"].append(datetime.strptime(d, "%Y-%m-%d").strftime("%m"))
                by_hour_structured["week"].append(datetime.strptime(d, "%Y-%m-%d").strftime("%V"))
                by_hour_structured["day"].append(datetime.strptime(d, "%Y-%m-%d").strftime("%d"))
                by_hour_structured["hour"].append(h)
                by_hour_structured["date"].append(d)

            missing_days_cnt += 1

        print("Missing days: ", missing_days_cnt)


        for edge in payload:
            # ---------------------- Fetching registration datetime ----------------------

            # This is the datetime which will be representative of a volume, specifically, there will be multiple datetimes with the same day
            # to address this fact we'll just re-format the data to keep track of the day, but also maintain the volume values for each hour
            reg_datetime = datetime.strptime(datetime.fromisoformat(edge["node"]["from"]).replace(tzinfo=None).isoformat(), "%Y-%m-%dT%H:%M:%S").strftime("%Y-%m-%dT%H")  # Only keeping the datetime without the +00:00 at the end

            # ----------------------- Total volumes section -----------------------

            by_hour_structured["trp_id"].append(trp_id)
            by_hour_structured["year"].append(datetime.strptime(reg_datetime, "%Y-%m-%dT%H").strftime("%Y"))
            by_hour_structured["month"].append(datetime.strptime(reg_datetime, "%Y-%m-%dT%H").strftime("%m"))
            by_hour_structured["week"].append(datetime.strptime(reg_datetime, "%Y-%m-%dT%H").strftime("%V"))
            by_hour_structured["day"].append(datetime.strptime(reg_datetime, "%Y-%m-%dT%H").strftime("%d"))
            by_hour_structured["hour"].append(datetime.strptime(reg_datetime, "%Y-%m-%dT%H").strftime("%H"))
            by_hour_structured["volume"].append(edge["node"]["total"]["volumeNumbers"]["volume"] if edge["node"]["total"]["volumeNumbers"] is not None else None) # In some cases the volumeNumbers key could have null as value, so the "volume" key won't be present. In that case we'll directly insert None as value with an if statement
            by_hour_structured["coverage"].append(edge["node"]["total"]["coverage"]["percentage"] or None) # For less recent data it's possible that sometimes coverage can be null, so we'll address this problem like so
            by_hour_structured["date"].append(datetime.strptime(reg_datetime, "%Y-%m-%dT%H").date().isoformat())


        return pd.DataFrame(by_hour_structured).sort_values(by=["date", "hour"], ascending=True)



    async def ingest(self, payload: dict[str, Any]):
        self.data = payload
        ...





























