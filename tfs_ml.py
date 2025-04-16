from tfs_utils import *
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

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import make_scorer, r2_score, mean_squared_error, mean_absolute_error, root_mean_squared_error


simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)



def sin_transformer(timeframe: int, data: [dd.Series | dd.DataFrame]) -> [dd.Series | dd.DataFrame]:
    """
    The timeframe indicates a number of days
    """
    return np.sin(data * (2. * np.pi / timeframe))

def cos_transformer(timeframe: int, data: [dd.Series | dd.DataFrame]) -> [dd.Series | dd.DataFrame]:
    """
    The timeframe indicates a number of days
    """
    return np.cos((data-1)*(2.*np.pi/timeframe))

def retrieve_n_ml_cpus() -> int:
    n_cpu = os.cpu_count()
    ml_dedicated_cores = int(n_cpu * 0.80)  #To avoid crashing while executing parallel computing in the GridSearchCV algorithm
    return ml_dedicated_cores



class BaseLearner:
    def __init__(self, client: Client):
        self.scorer = {"r2": make_scorer(r2_score),
                       "mean_squared_error": make_scorer(mean_squared_error),
                       "root_mean_squared_error": make_scorer(root_mean_squared_error),
                       "mean_absolute_error": make_scorer(mean_absolute_error)}
        self.client = client


    @staticmethod
    def split_data(volumes_preprocessed: dd.DataFrame, target: str) -> tuple[dd.DataFrame, dd.DataFrame, dd.DataFrame, dd.DataFrame]:
        """
        Splits the Dask DataFrame into training and testing sets based on the target column.

        Parameters:
        - volumes_preprocessed: dd.DataFrame
        - target: str ("volume" or "mean_speed")

        Returns:
        - X_train, X_test, y_train, y_test
        """

        if target not in ["volume", "mean_speed"]:
            raise ValueError("Wrong target variable in the split_data() function. Must be 'volume' or 'mean_speed'.")

        X = volumes_preprocessed.drop(columns=[target])
        y = volumes_preprocessed[[target]]

        #print("X shape: ", f"({len(X)}, {len(X.columns)})", "\n")
        #print("y shape: ", f"({len(y)}, {len(y.columns)})", "\n")

        n_rows = volumes_preprocessed.shape[0].compute()
        p_70 = int(n_rows * 0.70)

        X_train = X.head(p_70)
        X_test = X.tail(len(X) - p_70)

        #print(X_train.head(10))
        #print(X_test.head(10))

        y_train = y.head(p_70)
        y_test = y.tail(len(y) - p_70)

        #print(y_train.head(10))
        #print(y_test.head(10))

        #print(X_train.tail(5), "\n", X_test.tail(5), "\n", y_train.tail(5), "\n", y_test.tail(5), "\n")
        #print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

        return X_train, X_test, y_train, y_test


    def gridsearch_for_model(self, X_train, y_train, target: str, model_name: str, road_category: str) -> None:

        ops_name = get_active_ops()

        if target == "volume":
            parameters_grid = volumes_models_gridsearch_parameters[model_name]
            best_parameters_by_model = volumes_best_parameters_by_model
        elif target == "mean_speed":
            parameters_grid = speeds_models_gridsearch_parameters[model_name]
            best_parameters_by_model = speeds_best_parameters_by_model
        else:
            raise Exception("Wrong target variable in GridSearchCV executor function")

        model = model_names_and_functions[model_name]() #Finding the function which returns the model and executing it

        ml_parameters_folder_path = get_ml_model_parameters_folder_path(target=target, road_category=road_category)
        model_filename = ops_name + "_" + road_category + "_" + model_name + "_" + "parameters"

        t_start = datetime.now()
        print(f"{model_name} GridSearchCV started at {t_start}\n")


        time_cv = TimeSeriesSplit(n_splits=5) #A time series splitter for cross validation (for time series cross validation) is necessary since there's a relationship between the rows, thus we cannot use classic cross validation which shuffles the data because that would lead to a data leakage and incorrect predictions
        gridsearch = GridSearchCV(model, param_grid=parameters_grid, scoring=self.scorer, refit="mean_absolute_error", return_train_score=True, n_jobs=retrieve_n_ml_cpus(), scheduler=self.client, cv=time_cv)  #The models_gridsearch_parameters is obtained from the tfs_models file

        backend_kwargs = {"scatter": [X_train, y_train]}
        with joblib.parallel_backend('dask'): #, **backend_kwargs):
            gridsearch.fit(X=X_train, y=y_train)

        gridsearch_results = pd.DataFrame(gridsearch.cv_results_)[["params", "mean_fit_time", "mean_test_r2", "mean_train_r2",
                                                                   "mean_test_mean_squared_error", "mean_train_mean_squared_error",
                                                                   "mean_test_root_mean_squared_error", "mean_train_root_mean_squared_error",
                                                                   "mean_test_mean_absolute_error", "mean_train_mean_absolute_error",
                                                                   "mean_test_mean_absolute_percentage_error", "mean_train_mean_absolute_percentage_error"]]

        print(f"============== {model_name} grid search results ==============\n")
        print(gridsearch_results, "\n")

        t_end = datetime.now()
        print(f"{model_name} GridSearchCV finished at {t_end}\n")
        print(f"Time passed: {t_end - t_start}")

        gridsearch_results.to_json(f"./ops/{road_category}_{model_name}_grid_params_and_results.json", indent=4) #TODO FOR TESTING PURPOSES

        print("GridSearchCV best estimator: ", gridsearch.best_estimator_)
        print("GridSearchCV best parameters: ", gridsearch.best_params_)
        print("GridSearchCV best score: ", gridsearch.best_score_)

        print("GridSearchCV best combination index (in the results dataframe): ", gridsearch.best_index_, "\n")

        #print(gridsearch.scorer_, "\n")

        #The best_parameters_by_model variable is obtained from the tfs_models file
        true_best_parameters = {model_name: gridsearch_results["params"].loc[best_parameters_by_model[model_name]] if gridsearch_results["params"].loc[best_parameters_by_model[model_name]] is not None else {}}
        #TODO THIS SHOULD PROBABLY BECOME: true_best_parameters = {model_name: gridsearch_results["params"].loc[best_parameters_by_model[road_category][model_name]] if gridsearch_results["params"].loc[best_parameters_by_model[raod_category][model_name]] is not None else {}}
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


    @staticmethod
    def train_model(X_train, y_train, target: str, model_name: str, road_category: str) -> None:

        # -------------- Filenames, etc. --------------

        ops_name = get_active_ops()

        models_parameters_folder_path = get_ml_model_parameters_folder_path(target, road_category)
        models_folder_path = get_ml_models_folder_path(target, road_category)

        model_filename = ops_name + "_" + road_category + "_" + model_name

        model_parameters_filename = ops_name + "_" + road_category + "_" + model_name + "_" + "parameters" + ".json"
        model_parameters_filepath = models_parameters_folder_path + model_parameters_filename


        # -------------- Parameters extraction --------------

        with open(model_parameters_filepath, "r") as parameters_file:
            parameters = json.load(parameters_file)

        parameters = parameters[model_name] #Extracting the model parameters


        # -------------- Training --------------

        model = model_names_and_class_objects[model_name](**parameters) #Unpacking the dictionary to set all parameters to instantiate the model's class object

        with joblib.parallel_backend('dask'):
            model.fit(X_train.compute(), y_train.compute())

        print(f"Successfully trained {model_name} with parameters: {parameters}")


        # -------------- Model exporting --------------

        try:
            joblib.dump(model, models_folder_path + model_filename + ".joblib", protocol=pickle.HIGHEST_PROTOCOL)

            with open(models_folder_path + model_filename + ".pkl", "wb") as ml_pkl_file:
                pickle.dump(model, ml_pkl_file, protocol=pickle.HIGHEST_PROTOCOL)

        except Exception as e:
            print(f"\033[91mCouldn't export trained model. Safely exited the program. Error: {e}\033[0m")
            exit(code=1)


        print("\n\n")


        return None


    @staticmethod
    def test_model(X_test, y_test, target: str, model_name, road_category: str) -> None:

        ops_name = get_active_ops()

        ml_folder_path = get_ml_models_folder_path(target, road_category)
        model_filename = ops_name + "_" + road_category + "_" + model_name


        # -------------- Model loading --------------

        model = joblib.load(ml_folder_path + model_filename + ".joblib")
        #print(model.get_params())

        with joblib.parallel_backend('dask'):
            y_pred = model.predict(X_test)


        print(f"================= {model_name} testing metrics =================")
        print("R^2: ", r2_score(y_true=y_test, y_pred=y_pred))
        print("Mean Absolute Error: ", mean_absolute_error(y_true=y_test, y_pred=y_pred))
        print("Mean Squared Error: ", mean_squared_error(y_true=y_test, y_pred=y_pred))
        print("Root Mean Squared Error: ", root_mean_squared_error(y_true=y_test, y_pred=y_pred))


        return None



class TrafficVolumesLearner(BaseLearner):

    def __init__(self, volumes_data: [dd.DataFrame | pd.DataFrame], client: Client):
        super().__init__(client)
        self.volumes_data = volumes_data


    def get_volumes_data(self) -> dd.DataFrame:
        return self.volumes_data


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

        #print("\n\n")

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



class AverageSpeedLearner(BaseLearner):

    def __init__(self, average_speed_file_path: str, client: Client):
        super().__init__(client)
        self.average_speed_file_path = average_speed_file_path


    def get_average_speed_data(self) -> dd.DataFrame:
        speeds = dd.read_csv(self.average_speed_file_path)
        return speeds


    def avg_speeds_ml_preprocessing_pipeline(self) -> dd.DataFrame:

        speeds = self.get_average_speed_data()

        # ------------------ Cyclical variables encoding ------------------

        speeds["hour_sin"] = sin_transformer(data=speeds["hour_start"], timeframe=24)
        speeds["hour_cos"] = sin_transformer(data=speeds["hour_start"], timeframe=24)

        speeds["week_sin"] = sin_transformer(data=speeds["week"], timeframe=52)
        speeds["week_cos"] = sin_transformer(data=speeds["week"], timeframe=52)

        speeds["day_sin"] = sin_transformer(data=speeds["day"], timeframe=31)
        speeds["day_cos"] = sin_transformer(data=speeds["day"], timeframe=31)

        speeds["month_sin"] = sin_transformer(data=speeds["month"], timeframe=12)
        speeds["month_cos"] = sin_transformer(data=speeds["month"], timeframe=12)

        print("\n\n")

        #------------------ Outliers filtering with Z-Score ------------------

        speeds = ZScore(speeds, "mean_speed")


        speeds = speeds.sort_values(by=["year", "month", "day"], ascending=True)


        # ------------------ Variables normalization ------------------

        scaler = MinMaxScaler()
        speeds[["mean_speed", "percentile_85", "coverage"]] = scaler.fit_transform(speeds[["mean_speed", "percentile_85", "coverage"]])


        #------------------ Creating lag features ------------------

        speeds_lag_column_names = [f"mean_speed_lag{i}" for i in range(1, 62)] #TODO REDUCE THE NUMBER OF LAG FEATURES, OTHERWISE THE GRID SEARCH WILL TAKE FOREVER
        percentile_85_lag_column_names = [f"percentile_85_lag{i}" for i in range(1, 62)]

        for idx, n in enumerate(speeds_lag_column_names): speeds[n] = speeds["mean_speed"].shift(idx + 1)
        for idx, n in enumerate(percentile_85_lag_column_names): speeds[n] = speeds["percentile_85"].shift(idx + 1)

        #print(speeds.head(10))
        #print(speeds.dtypes)

        speeds = speeds.drop(columns=["year", "month", "week", "day", "trp_id", "date"], axis=1).persist()

        #print("Average speeds dataframe head: ")
        #print(speeds.head(5), "\n")

        #print("Average speeds dataframe tail: ")
        #print(speeds.tail(5), "\n")


        return speeds










class OnePointForecaster:
    """
    self.trp_road_category: to find the right model to predict the data
    """
    def __init__(self, trp_id: str, road_category: str):
        self.trp_id = trp_id
        self.road_category = road_category



class OnePointVolumesForecaster(OnePointForecaster):

    def __init__(self, trp_id: str, road_category: str):
        super().__init__(trp_id, road_category) #Calling the father class with its arguments
        self.trp_id = trp_id
        self.road_category = road_category


    def pre_process_data(self, forecasting_target_datetime: datetime, X_test=None, y_test=None): #TODO REMOVE =None AFTER TESTING

        forecasting_target_datetime = forecasting_target_datetime.strftime("%Y-%m-%d %H:%M:%S.%f")

        print(datetime.now().strftime(dt_format))
        print(forecasting_target_datetime)

        forecasting_window = pd.date_range(start=datetime.now(), end=forecasting_target_datetime, freq="1h")

        #TODO CHECK IF DATE ISN'T BEFORE THE ONE OF THE LAST DATA AVAILABLE
        #TODO PARSE DATES OBTAINED IN THIS METHOD TO ONLY PRESERVE DATE AND HOUR
        # THEN DIVIDE THEM INTO YEAR, MONTH, DAY AND HOUR

        print(forecasting_window, "\n\n")

        return None










    def forecast_volumes(self):



        return None




























#TODO CREATE THE AvgSpeedForecaster CLASS AND THEN CREATE THE OnePointForecaster AND A2BForecaster CLASSES, THEY WILL JUST RETRIEVE AND USE THE PRE-MADE, TESTED AND EXPORTED MODELS


























