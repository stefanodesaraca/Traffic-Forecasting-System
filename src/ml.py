import io
import gc
import pickle
import json
import warnings
from warnings import simplefilter
import hashlib
import datetime
from typing import Any
from pydantic import BaseModel
import pandas as pd
import matplotlib.pyplot as plt
import joblib

import dask.dataframe as dd
from dask.distributed import Client

from dask_ml.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit

from sklearn.metrics import (
    make_scorer,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    root_mean_squared_error,
    PredictionErrorDisplay
)

from sktime.base import BaseEstimator as SktimeBaseEstimator
from pytorch_forecasting.models.base_model import BaseModel as PyTorchForecastingBaseModel

from exceptions import ModelNotSetError, ScoringNotFoundError
from definitions import GlobalDefinitions, ProjectTables, ProjectConstraints

from brokers import DBBroker
from utils import check_target


simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)





class ModelWrapper(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    model_obj: Any
    target: str

    # noinspection PyNestedDecorators
    #@field_validator("model_obj", mode="after")
    #@classmethod
    #def validate_model_obj(cls, model_obj) -> Any:
    #    if not issubclass(model_obj.__class__, ScikitLearnBaseEstimator | PyTorchForecastingBaseModel | SktimeBaseEstimator):
    #        raise ValueError(f"Object passed is not an estimator accepted from this class. Type of the estimator received: {type(model_obj)}")
    #    return model_obj


    @property
    def name(self) -> str:
        """
        Get the name of the model.

        Returns
        -------
        str
            The class name of the model object.
        """
        return self.model_obj.__class__.__name__


    @property
    def model_id(self) -> str:
        return hashlib.sha256(self.name.encode('utf-8')).hexdigest()


    @property
    def params(self) -> dict[Any, Any]:
        """
        Get the current model's parameters.

        Returns
        -------
        dict[Any, Any]
            The current model's parameters at the time of the call of this method.
        """
        return self.model_obj.get_params()


    @property
    def fit_state(self):
        """
        Get the fitting state of the model.

        Returns
        -------
        bool
            Whether the model has been fitted.
        """
        return self.model_obj.__sklearn_is_fitted__()


    def set(self, model_object: Any) -> None:
        """
        Set the model object in the wrapper. Basically just imputing a model object (not an instance) inside the wrapper

        Parameters
        ----------
        model_object : Any
            The model object to wrap.

        Returns
        -------
        None
        """
        setattr(self, "model_obj", model_object)
        return None


    def get(self) -> Any:
        """
        Get the model object.

        Returns
        -------
        Any
            The model object.

        Raises
        ------
        ModelNotSetError
            If no model has been passed to the wrapper class.
        """
        #For future development of this function:
        #When self.model_obj is a scikit-learn model (for example: RandomForestRegressor), using "if not self.model_obj" internally tries to evaluate the object in a boolean context.
        # For scikit-learn models like RandomForestRegressor, this can trigger special methods like __len__ or others
        # that assume the model has already been fitted (which would populate attributes like estimators_), leading to the AttributeError.
        if self.model_obj is None: #WARNING: keep the condition with "is None", setting the condition like "if not self.model_obj" will raise an error. Read above
            raise ModelNotSetError("Model not passed to the wrapper class")
        return self.model_obj


    def fit(self, X: dd.DataFrame, y: dd.DataFrame | None = None) -> Any:
        """
        Train the model on the input data.

        Parameters
        ----------
        X : dd.DataFrame
            Predictor variables' data for training.
        y : dd.DataFrame
            The target variable's data for training.

        Returns
        -------
        Any
            The trained model object.
        """
        with joblib.parallel_backend("dask"):
            return self.model_obj.fit(X.compute(), y.compute())


    def predict(self, X: dd.DataFrame) -> Any:
        """
        Make predictions using the trained model.

        Parameters
        ----------
        X : dd.DataFrame
            Input features for prediction.

        Returns
        -------
        Any
            Predicted values.

        Raises
        ------
        TypeError
            If the model type is unsupported.

        Notes
        -----
        Currently supports ScikitLearnBaseEstimator. PyTorchForecastingBaseModel
        and SktimeBaseEstimator support is still to be implemented.
        """
        if isinstance(self.model_obj, (RandomForestRegressor, DecisionTreeRegressor, HistGradientBoostingRegressor)):
            with joblib.parallel_backend("dask"):
                return self.model_obj.predict(X.compute()) # type: ignore[attr-defined] # <- WARNING: this comment is used to avoid seeing a useless warning since the model will indeed have a predict method, but the scikit-learn BaseEstimator class doesn't
        elif isinstance(self.model_obj, PyTorchForecastingBaseModel):
            return ... #NOTE STILL TO IMPLEMENT
        elif isinstance(self.model_obj, SktimeBaseEstimator):
            return ... #NOTE STILL TO IMPLEMENT
        else:
            raise TypeError(f"Unsupported model type: {type(self.model_obj)}")



class TFSLearner:
    """
    Base class for models that learn to predict traffic volumes, average speed,
    or other traffic-related data using machine learning or statistical methods.

    Parameters
    ----------
    model : callable (class instance)
        The model **class instance** (not an object). For example:
        `TFSLearner(model=EstimatorClass(), ...)` and **not**
        `TFSLearner(model=EstimatorClass, ...)`.

    road_category : str
        The category of the road where the TRP that recorded the data was located

    target : str
        The target variable to predict (e.g., 'traffic_volumes', 'average_speed').

    client : dask.distributed.Client
        A Dask distributed client used to parallelize computation.
    """

    def __init__(self, model: callable, target: str, db_broker: DBBroker, client: Client | None = None, road_category: str | None = None):
        self._scoring_functions: dict[str, callable] = {
            "r2": r2_score,
            "mean_squared_error": mean_squared_error,
            "root_mean_squared_error": root_mean_squared_error,
            "mean_absolute_error": mean_absolute_error
        }
        self._scorer: dict[str, Any] = {
            func_name: make_scorer(func) for func_name, func in self._scoring_functions.items()
        }
        self._client: Client | None = client
        self._target: str = target
        self._road_category: str | None = road_category
        self._model: ModelWrapper = ModelWrapper(model_obj=model, target=self._target)
        self._db_broker: DBBroker = db_broker

        check_target(self._target, errors=True)


    @property
    def model(self) -> ModelWrapper:
        return self._model

    @property
    def scorer(self) -> dict[str, Any]:
        return self._scorer


    def _get_grid(self) -> dict[str, Any]:
        """
        Get the model's custom hyperparameter grid.

        Returns
        -------
        dict[str, Any]
            The model's grid for hyperparameter tuning.
        """
        return self._db_broker.send_sql(sql=f"""SELECT "{f"{self._target}_grid"}"
                                                FROM "{ProjectTables.MLModels.value}"
                                                WHERE "id" = '{self._model.model_id}';""", single=True)[f'{self._target}_grid']


    def gridsearch(self, X_train: dd.DataFrame, y_train: dd.DataFrame) -> pd.DataFrame | None:
        """
        Perform grid search cross-validation for hyperparameter tuning.

        Parameters
        ----------
        X_train : dd.DataFrame
            Predictors' training data.
        y_train : dd.DataFrame
            Target variable's training data.

        Returns
        -------
        None

        Raises
        ------
        TargetVariableNotFoundError
            If wrong target variable is specified.
        ScoringNotFoundError
            If scoring metric is not found.

        Notes
        -----
        Uses TimeSeriesSplit for cross-validation to respect temporal relationships
        in the data. The grid search uses multiple scoring metrics and refits on
        'mean_absolute_error'.
        """

        t_start = datetime.datetime.now()
        print(f"{self._model.name} GridSearchCV started at {t_start}\n")

        gridsearch = GridSearchCV(
            self._model.get(),
            param_grid=self._get_grid(),
            scoring=self._scorer,
            refit="mean_absolute_error",
            return_train_score=True,
            n_jobs=GlobalDefinitions.ML_CPUS,
            scheduler=self._client,
            cv=TimeSeriesSplit(n_splits=5)  # A time series splitter for cross validation (for time series cross validation) is necessary since there's a relationship between the rows, thus we cannot use classic cross validation which shuffles the data because that would lead to a data leakage and incorrect predictions
        )  # The models_gridsearch_parameters is obtained from the tfs_models file

        with joblib.parallel_backend("dask"):
            gridsearch.fit(X=X_train, y=y_train)

        t_end = datetime.datetime.now()
        print(f"{self._model.name} GridSearchCV finished at {t_end}")
        print(f"Time passed: {t_end - t_start}\n")

        try:
            return pd.DataFrame(gridsearch.cv_results_)[
                [
                    "params",
                    "mean_fit_time",
                    "mean_test_r2",
                    "mean_train_r2",
                    "mean_test_mean_squared_error",
                    "mean_train_mean_squared_error",
                    "mean_test_root_mean_squared_error",
                    "mean_train_root_mean_squared_error",
                    "mean_test_mean_absolute_error",
                    "mean_train_mean_absolute_error",
                ]
            ]

        except KeyError as e:
            raise ScoringNotFoundError(f"\033[91mScoring not found. Parent error: {e}")

        finally:
            gc.collect()


    def compute_fpe(self, y_true: pd.DataFrame | dd.DataFrame, y_pred: pd.DataFrame | dd.DataFrame) -> dict[str, float | int]:
        return {func_name: scoring_function(**{"y_true": y_true, "y_pred": y_pred}) for func_name, scoring_function in self._scoring_functions.items()}


    def export_gridsearch_results(self, results: pd.DataFrame) -> None:
        results["model_id"] = self._model.model_id
        results["road_category"] = self._road_category
        results["target"] = self._target
        results["params"] = results["params"].apply(lambda x: json.dumps(x, sort_keys=True)) #Binarizing parameters' dictionary. sort_keys=True ensures that the dictionary is always serialized in a consistent manner
        results["params_hash"] = results["params"].apply(lambda x: hashlib.sha256(x.encode('utf-8')).hexdigest()) #Encodes the JSON string and then calculates the sha256 hash of it
        results = results.reindex(columns=["model_id",
                                           "road_category",
                                           "target",
                                           "params",
                                           "params_hash",
                                           "mean_fit_time",
                                           "mean_test_r2",
                                           "mean_train_r2",
                                           "mean_test_mean_squared_error",
                                           "mean_train_mean_squared_error",
                                           "mean_test_root_mean_squared_error",
                                           "mean_train_root_mean_squared_error",
                                           "mean_test_mean_absolute_error",
                                           "mean_train_mean_absolute_error"]) #Changing the columns order to match the one in the SQL query below
        self._db_broker.send_sql(f'''
            INSERT INTO "{ProjectTables.ModelGridSearchCVResults.value}" (
                "result_id",
                "model_id",
                "road_category_id",
                "target",
                "params",
                "params_hash",
                "mean_fit_time",
                "mean_test_r2",
                "mean_train_r2",
                "mean_test_mean_squared_error",
                "mean_train_mean_squared_error",
                "mean_test_root_mean_squared_error",
                "mean_train_root_mean_squared_error",
                "mean_test_mean_absolute_error",
                "mean_train_mean_absolute_error"
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            ON CONFLICT ON CONSTRAINT {ProjectConstraints.UNIQUE_MODEL_ROAD_TARGET_PARAMS.value} DO UPDATE
            SET
                "road_category_id" = EXCLUDED.road_category_id,
                "target"  = EXCLUDED.target,
                "params" = EXCLUDED.params,
                "params_hash" = EXCLUDED.params_hash,
                "mean_fit_time" = EXCLUDED.mean_fit_time,
                "mean_test_r2" = EXCLUDED.mean_test_r2,
                "mean_train_r2" = EXCLUDED.mean_train_r2,
                "mean_test_mean_squared_error" = EXCLUDED.mean_test_mean_squared_error,
                "mean_train_mean_squared_error" = EXCLUDED.mean_train_mean_squared_error,
                "mean_test_root_mean_squared_error" = EXCLUDED.mean_test_root_mean_squared_error,
                "mean_train_root_mean_squared_error" = EXCLUDED.mean_train_root_mean_squared_error,
                "mean_test_mean_absolute_error" = EXCLUDED.mean_test_mean_absolute_error,
                "mean_train_mean_absolute_error" = EXCLUDED.mean_train_mean_absolute_error;
        ''', many=True, many_values=[tuple(row) for row in results.itertuples(name=None)])


        #TODO EXPORT PARAMETERES TO JSON FOR DEEPER ANALYSES


        return None


    def export_internal_model(self) -> None:
        joblib_bytes = io.BytesIO() #Serializing model into a joblib object directly in memory through the BytesIO class
        joblib.dump(self._model, joblib_bytes)
        joblib_bytes.seek(0)
        self._db_broker.send_sql(f"""
                INSERT INTO "{ProjectTables.TrainedModels.value}" ("id", "target", "road_category", "joblib_object", "pickle_object")
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT ("id", "target", "road_category") DO UPDATE
                SET 
                "joblib_object" = EXCLUDED."joblib_object",
                "pickle_object" = EXCLUDED."pickle_object";
            """, execute_args=[self._model.model_id, self._target, self._road_category, joblib_bytes.getvalue(), pickle.dumps(self._model.model_obj)])
        return None

















# TODO CREATE THE A2BForecaster CLASS
