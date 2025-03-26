from tfs_utilities import ZScore, get_active_ops_name, get_ml_models_folder_path
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

        p_75 = int((len(y) / 100) * 75)

        X_train = X.loc[:p_75].persist()
        X_test = X.loc[p_75:].persist()

        y_train = y.loc[:p_75].persist()
        y_test = y.loc[p_75:].persist()


        #print(X_train.tail(5), "\n", X_test.tail(5), "\n", y_train.tail(5), "\n", y_test.tail(5), "\n")
        #print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


        return X_train, X_test, y_train, y_test


    def train_model(self, X_train, y_train, model_name):

        ops_name = get_active_ops_name()

        parameters_grid = models_gridsearch_parameters[model_name]
        model = model_names_and_functions[model_name]() #Finding the function which returns the model and executing it

        ml_folder_path = get_ml_models_folder_path()
        model_filename = ops_name + "_" + model_name


        if model_name != "XGBRegressor":

            client = Client(processes=False)

            gridsearch = GridSearchCV(model, param_grid=parameters_grid, return_train_score=False, n_jobs=-1, scheduler="threads") #The models_gridsearch_parameters is obtained from the tfs_models file

            with joblib.parallel_backend('dask'):
                gridsearch.fit(X=X_train, y=y_train)


            print(f"============== {model_name} grid search results ==============")
            print(pd.DataFrame(gridsearch.cv_results_), "\n")

            print("Best estimator: ", gridsearch.best_estimator_, "\n")
            print("Best parameters: ", gridsearch.best_params_, "\n")
            print("Best score: ", gridsearch.best_score_, "\n")

            print("Best index: ", gridsearch.best_index_, "\n")

            #print(gridsearch.scorer_, "\n")

            client.close()

            joblib.dump(gridsearch.best_estimator_, ml_folder_path + model_filename + ".joblib")

            with open(ml_folder_path + model_filename + ".pkl", "wb") as ml_pkl_file:
                pickle.dump(gridsearch.best_estimator_, ml_pkl_file)


        else:

            client = Client(processes=False)
            with joblib.parallel_backend('dask'):
                model.fit(X=X_train, y=y_train)

            client.close()

            joblib.dump(model, ml_folder_path + model_filename + ".joblib")

            with open(ml_folder_path + model_filename + ".pkl", "wb") as ml_pkl_file:
                pickle.dump(model, ml_pkl_file)



        #TODO EXPORT TRAINED MODELS HERE WITH JOBLIB AND PICKLE




        return None




















    def test_models(self):


        #TODO IMPORT MODELS HERE TO TEST THEM WITH TEST DATA




        return None



    def forecast_volumes(self):





        return None










#TODO CREATE THE AvgSpeedForecaster CLASS AND THEN CREATE THE OnePointForecaster AND A2BForecaster CLASSES, THEY WILL JUST RETRIEVE AND USE THE PRE-MADE, TESTED AND EXPORTED MODELS


























