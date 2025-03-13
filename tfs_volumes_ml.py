import os
import pickle
import joblib
import warnings
from warnings import simplefilter
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats
from scipy.special import softmax

from feature_engine.creation import CyclicalFeatures

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
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


class TrafficVolumesForecaster:

    def __init__(self, volumes_file_path):
        self.volumes_file_path = volumes_file_path


    def get_volumes_data(self):
        volumes = pd.read_csv(self.volumes_file_path)
        return volumes


    #TODO CHANGE NAME HERE
    def preprocessing_pipeline(self):

        volumes = self.get_volumes_data()

        print(volumes.columns)
        print(volumes.dtypes)
        print(volumes.isna().sum())
        print(volumes.shape)
        print(volumes.corr(numeric_only=True))
        print(volumes.cov(numeric_only=True))


        #TODO ENCODE CYCLICAL VARIABLES HERE


        return None #TODO RETURN THE PREPROCESSED DATA



























































