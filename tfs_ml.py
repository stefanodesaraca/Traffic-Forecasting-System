from tfs_utilities import *
from tfs_models import *
import os
import numpy as np
import pickle
import warnings
from warnings import simplefilter
from datetime import datetime
import pandas as pd
import dask.dataframe as dd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats
from scipy.special import softmax
import time

from dask.distributed import Client
import joblib

from dask_ml.preprocessing import MinMaxScaler
from dask_ml.model_selection import GridSearchCV

from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import make_scorer, r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error



simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)


def sin_transformer(timeframe: int, data: [pd.Series | pd.DataFrame]):
    """
    The timeframe indicates a number of days
    """
    return np.sin(data * (2. * np.pi / timeframe))

def cos_transformer(timeframe: int, data: [pd.Series | pd.DataFrame]):
    """
    The timeframe indicates a number of days
    """
    return np.cos((data-1)*(2.*np.pi/timeframe))



class TrafficVolumesForecaster:

    def __init__(self, volumes_file_path):
        self.volumes_file_path = volumes_file_path
        self.scorer = {"r2": make_scorer(r2_score),
                       "mean_squared_error": make_scorer(mean_squared_error),
                       "root_mean_squared_error": make_scorer(root_mean_squared_error),
                       "mean_absolute_error": make_scorer(mean_absolute_error),
                       "mean_absolute_percentage_error": make_scorer(mean_absolute_percentage_error)}


    def get_volumes_data(self):
        volumes = dd.read_csv(self.volumes_file_path)
        return volumes


    def volumes_ml_preprocessing_pipeline(self):

        volumes = self.get_volumes_data()

        # ------------------ Cyclical variables encoding ------------------

        volumes["hour_sin"] = sin_transformer(data=volumes["hour"], timeframe=24)
        volumes["hour_cos"] = sin_transformer(data=volumes["hour"], timeframe=24)

        volumes["week_sin"] = sin_transformer(data=volumes["week"], timeframe=52)
        volumes["week_cos"] = sin_transformer(data=volumes["week"], timeframe=52)

        volumes["day_sin"] = sin_transformer(data=volumes["day"], timeframe=31)
        volumes["day_cos"] = sin_transformer(data=volumes["day"], timeframe=31)

        volumes["month_sin"] = sin_transformer(data=volumes["month"], timeframe=12)
        volumes["month_cos"] = sin_transformer(data=volumes["month"], timeframe=12)

        print("\n\n")

        #------------------ Outliers filtering with Z-Score ------------------

        volumes = ZScore(volumes, "volume")

        volumes = volumes.sort_values(by=["year", "month", "day"], ascending=True)


        # ------------------ Variables normalization ------------------

        scaler = MinMaxScaler()
        volumes[["volume", "coverage"]] = scaler.fit_transform(volumes[["volume", "coverage"]])


        #------------------ Creating lag features ------------------

        lag_column_names = ["volumes_lag1", "volumes_lag2", "volumes_lag3", "volumes_lag4", "volumes_lag5", "volumes_lag6", "volumes_lag7"]

        for idx, n in enumerate(lag_column_names): volumes[n] = volumes["volume"].shift(idx + 1)

        #print(volumes.head(10))
        #print(volumes.dtypes)

        volumes = volumes.drop(columns=["year", "month", "week", "day", "trp_id"], axis=1).persist()

        #print("Volumes dataframe head: ")
        #print(volumes.head(5), "\n")

        #print("Volumes dataframe tail: ")
        #print(volumes.tail(5), "\n")


        return volumes


    @staticmethod
    def split_data(volumes_preprocessed: dd.DataFrame):

        X = volumes_preprocessed.drop(columns=["volume"])
        y = volumes_preprocessed[["volume"]]

        #print("X shape: ", X.shape, "\n")
        #print("y shape: ", y.shape, "\n")

        p_75 = int((len(y) / 100) * 70)

        X_train = X.loc[:p_75].persist()
        X_test = X.loc[p_75:].persist()

        y_train = y.loc[:p_75].persist()
        y_test = y.loc[p_75:].persist()


        #print(X_train.tail(5), "\n", X_test.tail(5), "\n", y_train.tail(5), "\n", y_test.tail(5), "\n")
        #print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


        return X_train, X_test, y_train, y_test


    def gridsearch_for_model(self, X_train, y_train, model_name):

        ops_name = get_active_ops_name()

        parameters_grid = models_gridsearch_parameters[model_name]
        model = model_names_and_functions[model_name]()  # Finding the function which returns the model and executing it

        ml_parameters_folder_path = get_ml_model_parameters_folder_path()
        model_filename = ops_name + "_" + model_name + "_" + "parameters"

        if model_name != "XGBRegressor":

            client = Client(processes=False)

            gridsearch = GridSearchCV(model, param_grid=parameters_grid, scoring=self.scorer, refit="mean_absolute_error", return_train_score=True, n_jobs=-1, scheduler="threads", cv=5)  # The models_gridsearch_parameters is obtained from the tfs_models file

            with joblib.parallel_backend('dask'):
                gridsearch.fit(X=X_train, y=y_train)


            gridsearch_results = pd.DataFrame(gridsearch.cv_results_)[["params", "mean_fit_time", "mean_test_r2", "mean_train_r2",
                                                                       "mean_test_mean_squared_error", "mean_train_mean_squared_error",
                                                                       "mean_test_root_mean_squared_error", "mean_train_root_mean_squared_error",
                                                                       "mean_test_mean_absolute_error", "mean_train_mean_absolute_error",
                                                                       "mean_test_mean_absolute_percentage_error", "mean_train_mean_absolute_percentage_error"]]

            print(f"============== {model_name} grid search results ==============\n")
            print(gridsearch_results, "\n")

            print("GridSearchCV best estimator: ", gridsearch.best_estimator_)
            print("GridSearchCV best parameters: ", gridsearch.best_params_)
            print("GridSearchCV best score: ", gridsearch.best_score_)

            print("GridSearchCV best combination index (in the results dataframe): ", gridsearch.best_index_, "\n")

            #print(gridsearch.scorer_, "\n")

            client.close()

            best_parameters_by_model = {"RandomForestRegressor": 11,
                                        "BaggingRegressor": 4,
                                        "DecisionTreeRegressor": 3}

            true_best_parameters = {model_name: gridsearch_results["params"].loc[best_parameters_by_model[model_name]]}

            print(f"True best parameters for {model_name}: ", true_best_parameters, "\n")


            with open(ml_parameters_folder_path + model_filename + ".json", "w") as parameters_file:
                json.dump(true_best_parameters, parameters_file)

        else:

            client = Client(processes=False)
            with joblib.parallel_backend('dask'):
                model.fit(X=X_train, y=y_train)

            client.close()

            joblib.dump(model, ml_parameters_folder_path + model_filename + ".joblib")

            with open(ml_parameters_folder_path + model_filename + ".pkl", "wb") as ml_pkl_file:
                pickle.dump(model, ml_pkl_file)

        # TODO EXPORT TRAINED MODELS HERE WITH JOBLIB AND PICKLE

        return None




    def train_model(self, model_name: str):

        ops_name = get_active_ops_name()
        model_parameters_filepath = ops_name + "_" + model_name + "_" + "parameters" + ".json"

        with open(model_parameters_filepath, "r") as parameters_file:
            model_parameters_value = json.load(parameters_file)









        joblib.dump(gridsearch.best_estimator_, ml_parameters_folder_path + model_filename + ".joblib")

        with open(ml_parameters_folder_path + model_filename + ".pkl", "wb") as ml_pkl_file:
            pickle.dump(gridsearch.best_estimator_, ml_pkl_file)




        return None





    def test_model(self, X_test, y_test, model_name):

        ops_name = get_active_ops_name()

        ml_folder_path = get_ml_models_folder_path()
        model_filename = ops_name + "_" + model_name

        model = joblib.load(ml_folder_path + model_filename + ".joblib")

        y_pred = model.predict(X=X_test)

        print("R^2: ", r2_score(y_true=y_test, y_pred=y_pred))
        print("Mean Absolute Error: ", mean_absolute_error(y_true=y_test, y_pred=y_pred))
        print("Mean Squared Error: ", mean_squared_error(y_true=y_test, y_pred=y_pred))
        print("Root Mean Squared Error: ", root_mean_squared_error(y_true=y_test, y_pred=y_pred))
        print("Mean Absolute Percentage Error: ", mean_absolute_percentage_error(y_true=y_test, y_pred=y_pred))





        #TODO IMPORT MODELS HERE TO TEST THEM WITH TEST DATA




        return None



    def forecast_volumes(self):





        return None










#TODO CREATE THE AvgSpeedForecaster CLASS AND THEN CREATE THE OnePointForecaster AND A2BForecaster CLASSES, THEY WILL JUST RETRIEVE AND USE THE PRE-MADE, TESTED AND EXPORTED MODELS


























