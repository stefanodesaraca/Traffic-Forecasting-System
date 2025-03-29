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

from dask_ml.wrappers import ParallelPostFit
from dask_ml.preprocessing import MinMaxScaler
from dask_ml.model_selection import GridSearchCV

from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import make_scorer, r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error
from sklearn.model_selection import cross_validate



simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)

#n_cpu = os.cpu_count()
#ml_dedicated_cores = n_cpu // 80


#TODO CONVERT pd.Series AND pd.DataFrame to dd. etc.
def sin_transformer(timeframe: int, data: [pd.Series | pd.DataFrame]) -> [pd.Series | pd.DataFrame]:
    """
    The timeframe indicates a number of days
    """
    return np.sin(data * (2. * np.pi / timeframe))

def cos_transformer(timeframe: int, data: [pd.Series | pd.DataFrame]) -> [pd.Series | pd.DataFrame]:
    """
    The timeframe indicates a number of days
    """
    return np.cos((data-1)*(2.*np.pi/timeframe))



#TODO CREATE A FATHER CLASS Forecaster WHICH WILL INTEGRATE TRAIN, TEST AND EXPORT METHODS



class TrafficVolumesForecaster:

    def __init__(self, volumes_file_path):
        self.volumes_file_path = volumes_file_path
        self.scorer = {"r2": make_scorer(r2_score),
                       "mean_squared_error": make_scorer(mean_squared_error),
                       "root_mean_squared_error": make_scorer(root_mean_squared_error),
                       "mean_absolute_error": make_scorer(mean_absolute_error),
                       "mean_absolute_percentage_error": make_scorer(mean_absolute_percentage_error)}


    def get_volumes_data(self) -> dd.DataFrame:
        volumes = dd.read_csv(self.volumes_file_path)
        return volumes


    def volumes_ml_preprocessing_pipeline(self) -> dd.DataFrame:

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
    def split_data(volumes_preprocessed: dd.DataFrame, return_pandas: bool = False):

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

        if return_pandas is True:
            return pd.DataFrame(X_train), pd.DataFrame(X_test), pd.DataFrame(y_train), pd.DataFrame(y_test)
        else:
            return X_train, X_test, y_train, y_test


    def gridsearch_for_model(self, X_train, y_train, model_name) -> None:

        ops_name = get_active_ops_name()

        parameters_grid = models_gridsearch_parameters[model_name]
        model = model_names_and_functions[model_name]()  # Finding the function which returns the model and executing it

        ml_parameters_folder_path = get_ml_model_parameters_folder_path()
        model_filename = ops_name + "_" + model_name + "_" + "parameters"


        client = Client(processes=True)

        gridsearch = GridSearchCV(model, param_grid=parameters_grid, scoring=self.scorer, refit="mean_absolute_error", return_train_score=True, n_jobs=-1, scheduler="multiprocessing", cv=5)  # The models_gridsearch_parameters is obtained from the tfs_models file

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

        #The best_parameters_by_model variable is obtained from the tfs_models file
        true_best_parameters = {model_name: gridsearch_results["params"].loc[best_parameters_by_model[model_name]] if gridsearch_results["params"].loc[best_parameters_by_model[model_name]] is not None else {}}
        auxiliary_parameters = model_auxiliary_parameters[model_name]

        #This is just to add the classic parameters which are necessary to get both consistent results and maximise the CPU usage to minimize training time. Also, these are the parameters that aren't included in the grid for the grid search algorithm
        for par, val in auxiliary_parameters.items():
            true_best_parameters[model_name][par] = val

        true_best_parameters["best_GridSearchCV_model_index"] = best_parameters_by_model[model_name]
        true_best_parameters["best_GridSearchCV_model_scores"] = gridsearch_results.loc[best_parameters_by_model[model_name]].to_dict() #to_dict() is used to convert the resulting series into a dictionary (which is a data type that's serializable by JSON)


        print(f"True best parameters for {model_name}: ", true_best_parameters, "\n")


        with open(ml_parameters_folder_path + model_filename + ".json", "w") as parameters_file:
            json.dump(true_best_parameters, parameters_file, indent=4)


        return None


    def train_model(self, X_train, y_train, model_name: str) -> None:

        # -------------- Filenames, etc. --------------

        ops_name = get_active_ops_name()

        models_parameters_folder_path = get_ml_model_parameters_folder_path()
        models_folder_path = get_ml_models_folder_path()

        model_filename = ops_name + "_" + model_name

        model_parameters_filename = ops_name + "_" + model_name + "_" + "parameters" + ".json"
        model_parameters_filepath = models_parameters_folder_path + model_parameters_filename


        # -------------- Parameters extraction --------------

        with open(model_parameters_filepath, "r") as parameters_file:
            parameters = json.load(parameters_file)

        parameters = parameters[model_name] #Extracting the model parameters


        # -------------- Training --------------

        model = model_names_and_class_objects[model_name](**parameters) #Unpacking the dictionary to set all parameters to instantiate the model's class object

        model.fit(X_train, y_train)


        print(f"Successfully trained {model_name} with parameters: {parameters}")


        # -------------- Model exporting --------------

        try:
            joblib.dump(model, models_folder_path + model_filename + ".joblib", protocol=pickle.HIGHEST_PROTOCOL)

            with open(models_folder_path + model_filename + ".pkl", "wb") as ml_pkl_file:
                pickle.dump(model, ml_pkl_file, protocol=pickle.HIGHEST_PROTOCOL)

        except Exception as e:
            print(f"\033[91mCouldn't export trained model. Safely exited the program. Error: {e}\033[0m")
            exit()


        print("\n\n")


        return None


    def test_model(self, X_test, y_test, model_name) -> None:

        ops_name = get_active_ops_name()

        ml_folder_path = get_ml_models_folder_path()
        model_filename = ops_name + "_" + model_name


        # -------------- Model loading --------------

        model = joblib.load(ml_folder_path + model_filename + ".joblib")
        #print(model.get_params())

        y_pred = model.predict(X_test)


        print(f"================= {model_name} testing metrics =================")
        print("R^2: ", r2_score(y_true=y_test, y_pred=y_pred))
        print("Mean Absolute Error: ", mean_absolute_error(y_true=y_test, y_pred=y_pred))
        print("Mean Squared Error: ", mean_squared_error(y_true=y_test, y_pred=y_pred))
        print("Root Mean Squared Error: ", root_mean_squared_error(y_true=y_test, y_pred=y_pred))


        return None



    def forecast_volumes(self):





        return None










#TODO CREATE THE AvgSpeedForecaster CLASS AND THEN CREATE THE OnePointForecaster AND A2BForecaster CLASSES, THEY WILL JUST RETRIEVE AND USE THE PRE-MADE, TESTED AND EXPORTED MODELS


























