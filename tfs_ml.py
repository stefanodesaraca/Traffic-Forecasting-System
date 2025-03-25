from bokeh.layouts import column

from tfs_utilities import ZScore
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

from feature_engine.creation import CyclicalFeatures

from dask_ml.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold, cross_val_score, train_test_split
from sklearn.metrics import make_scorer, r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor


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



























































