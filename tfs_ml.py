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
import gc

from dask.distributed import Client
import joblib

from dask_ml.preprocessing import MinMaxScaler, LabelEncoder
from dask_ml.model_selection import GridSearchCV

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import make_scorer, r2_score, mean_squared_error, mean_absolute_error, root_mean_squared_error, PredictionErrorDisplay


simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)



def sin_transformer(data: dd.Series | dd.DataFrame, timeframe: int) -> dd.Series | dd.DataFrame:
    """
    The timeframe indicates a number of days.
    Details:
        - The order of the function parameters has a specific reason. Since this function will be used with Dask's map_partition() (which takes a function and its parameters as input), it's important that the first parameter
          of sin_transformer is actually the data where to execute the transformation itself by map_partition()
    """
    #For more information about Dask's map_partition() function: https://docs.dask.org/en/stable/generated/dask.dataframe.DataFrame.map_partitions.html
    return np.sin(data * (2. * np.pi / timeframe))

def cos_transformer(data: dd.Series | dd.DataFrame, timeframe: int) -> dd.Series | dd.DataFrame:
    """
    The timeframe indicates a number of days
    Details:
        - The order of the function parameters has a specific reason. Since this function will be used with Dask's map_partition() (which takes a function and its parameters as input), it's important that the first parameter
          of sin_transformer is actually the data where to execute the transformation itself by map_partition()
    """
    #For more information about Dask's map_partition() function: https://docs.dask.org/en/stable/generated/dask.dataframe.DataFrame.map_partitions.html
    return np.cos((data-1)*(2.*np.pi/timeframe))


def retrieve_n_ml_cpus() -> int:
    n_cpu = os.cpu_count()
    ml_dedicated_cores = int(n_cpu * 0.75)  #To avoid crashing while executing parallel computing in the GridSearchCV algorithm
    #The value multiplied with the n_cpu values shouldn't be above .80, otherwise processes could crash during execution
    return ml_dedicated_cores



class BaseLearner:
    """
    The base class for other classes which implement machine learning or statistical methods to learn a predict traffic volumes, average speed or other data about traffic.
    Parameters:
        client: a Dask distributed local cluster client to use to distribute processes
    """
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

        if target not in ["volume", "mean_speed"]: raise ValueError("Wrong target variable in the split_data() function. Must be 'volume' or 'mean_speed'.")

        X = volumes_preprocessed.drop(columns=[target])
        y = volumes_preprocessed[[target]]

        #print("X shape: ", f"({len(X)}, {len(X.columns)})", "\n")
        #print("y shape: ", f"({len(y)}, {len(y.columns)})", "\n")

        n_rows = volumes_preprocessed.shape[0].compute()
        p_70 = int(n_rows * 0.70)

        X_train = dd.from_pandas(X.head(p_70)).persist()
        X_test = dd.from_pandas(X.tail(len(X) - p_70)).persist()

        #print(X_train.head(10))
        #print(X_test.head(10))

        y_train = dd.from_pandas(y.head(p_70)).persist()
        y_test = dd.from_pandas(y.tail(len(y) - p_70)).persist()

        #print(y_train.head(10))
        #print(y_test.head(10))

        #print(X_train.tail(5), "\n", X_test.tail(5), "\n", y_train.tail(5), "\n", y_test.tail(5), "\n")
        #print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

        return X_train, X_test, y_train, y_test


    def gridsearch(self, X_train, y_train, target: str, model_name: str, road_category: str) -> None:

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

        with joblib.parallel_backend('dask'):
            gridsearch.fit(X=X_train, y=y_train)

        try:
            gridsearch_results = pd.DataFrame(gridsearch.cv_results_)[["params", "mean_fit_time", "mean_test_r2", "mean_train_r2",
                                                                       "mean_test_mean_squared_error", "mean_train_mean_squared_error",
                                                                       "mean_test_root_mean_squared_error", "mean_train_root_mean_squared_error",
                                                                       "mean_test_mean_absolute_error", "mean_train_mean_absolute_error"]]

        except KeyError as e:
            print(f"\033[91mScoring not found. Error: {e}")


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

        gc.collect()

        return None


    @staticmethod
    def train_model(X_train: dd.DataFrame, y_train: dd.DataFrame, target: str, model_name: str, road_category: str) -> None:

        # -------------- Filenames, etc. --------------

        ops_name = get_active_ops()

        models_parameters_folder_path = get_ml_model_parameters_folder_path(target, road_category)
        models_folder_path = get_ml_models_folder_path(target, road_category)

        model_filename = ops_name + "_" + road_category + "_" + model_name

        model_parameters_filename = ops_name + "_" + road_category + "_" + model_name + "_" + "parameters" + ".json"
        model_parameters_filepath = models_parameters_folder_path + model_parameters_filename


        # -------------- Parameters extraction --------------

        with open(model_parameters_filepath, "r") as parameters_file: parameters = json.load(parameters_file)
        parameters = parameters[model_name] #Extracting the model parameters


        # -------------- Training --------------

        model = model_names_and_class_objects[model_name](**parameters) #Unpacking the dictionary to set all parameters to instantiate the model's class object

        with joblib.parallel_backend('dask'):
            model.fit(X_train, y_train)

        print(f"Successfully trained {model_name} with parameters: {parameters}")


        # -------------- Model exporting --------------

        try:
            joblib.dump(model, models_folder_path + model_filename + ".joblib", protocol=pickle.HIGHEST_PROTOCOL)
            with open(models_folder_path + model_filename + ".pkl", "wb") as ml_pkl_file:
                pickle.dump(model, ml_pkl_file, protocol=pickle.HIGHEST_PROTOCOL)
            return None

        except Exception as e:
            print(f"\033[91mCouldn't export trained model. Safely exited the program. Error: {e}\033[0m")
            exit(code=1)



    @staticmethod
    def test_model(X_test: dd.DataFrame, y_test: dd.DataFrame, target: str, model_name: str, road_category: str) -> None:

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

    def __init__(self, volumes_data: dd.DataFrame | pd.DataFrame, client: Client):
        super().__init__(client)
        self.volumes_data = volumes_data


    def preprocess(self) -> dd.DataFrame:

        volumes = self.volumes_data

        # ------------------ Cyclical variables encoding ------------------

        volumes["hour_sin"] = volumes["hour"].map_partitions(sin_transformer, timeframe=24)
        volumes["hour_cos"] = volumes["hour"].map_partitions(cos_transformer, timeframe=24)

        volumes["week_sin"] = volumes["week"].map_partitions(sin_transformer, timeframe=52)
        volumes["week_cos"] = volumes["week"].map_partitions(cos_transformer, timeframe=52)

        volumes["day_sin"] = volumes["day"].map_partitions(sin_transformer, timeframe=31)
        volumes["day_cos"] = volumes["day"].map_partitions(cos_transformer, timeframe=31)

        volumes["month_sin"] = volumes["month"].map_partitions(sin_transformer, timeframe=12)
        volumes["month_cos"] = volumes["month"].map_partitions(cos_transformer, timeframe=12)

        #print("\n\n")

        # ------------------ Outliers filtering with Z-Score ------------------

        volumes = ZScore(volumes, "volume")

        volumes = volumes.sort_values(by=["date"], ascending=True)

        # ------------------ TRP ID Target-Encoding ------------------

        volumes["trp_id"] = volumes["trp_id"].astype("category")

        encoder = LabelEncoder(use_categorical=True) #Using a label encoder to encode TRP IDs to include the effect of the non-independence of observations from each other inside the forecasting models
        volumes = volumes.assign(trp_id_encoded=encoder.fit_transform(volumes["trp_id"])) #The assign methods returns the dataframe obtained as input with the new column (in this case called "trp_id_encoded") added
        volumes.persist()

        #print("Encoded TRP IDs:", sorted(volumes["trp_id_encoded"].unique().compute()))

        # ------------------ Variables normalization ------------------

        scaler = MinMaxScaler()
        volumes[["volume", "coverage"]] = scaler.fit_transform(volumes[["volume", "coverage"]])

        #------------------ Creating lag features ------------------

        lag6h_column_names = [f"volumes_lag6h_{i}" for i in range(1, 7)]
        lag12h_column_names = [f"volumes_lag12h_{i}" for i in range(1, 7)]
        lag24h_column_names = [f"volumes_lag24h_{i}" for i in range(1, 7)]

        for idx, n in enumerate(lag6h_column_names): volumes[n] = volumes["volume"].shift(idx + 6) #6 hours shift
        for idx, n in enumerate(lag12h_column_names): volumes[n] = volumes["volume"].shift(idx + 12) #12 hours shift
        for idx, n in enumerate(lag24h_column_names): volumes[n] = volumes["volume"].shift(idx + 24) #24 hours shift

        #print(volumes.head(10))
        #print(volumes.dtypes)

        # ------------------ Creating dummy variables to address to the low value for traffic volumes in some years due to covid ------------------

        volumes["is_covid_year"] = (volumes['year'].isin(get_covid_years())).astype("int") #Creating a dummy variable which indicates if the traffic volume for a record has been affected by covid (because the traffic volume was recorded during one of the covid years)

        # ------------------ Dropping columns which won't be fed to the ML models ------------------

        volumes = volumes.drop(columns=["year", "month", "week", "day", "trp_id", "date"], axis=1).persist() #Keeping year and hour data and the encoded_trp_id

        #print("Volumes dataframe head: ")
        #print(volumes.head(5), "\n")

        #print("Volumes dataframe tail: ")
        #print(volumes.tail(5), "\n")

        #print(volumes.compute().head(10))

        return volumes



class AverageSpeedLearner(BaseLearner):

    def __init__(self, speeds_data: dd.DataFrame, client: Client):
        super().__init__(client)
        self.speeds_data = speeds_data


    def preprocess(self) -> dd.DataFrame:

        speeds = self.speeds_data

        # ------------------ Cyclical variables encoding ------------------

        speeds["hour_start_sin"] = speeds["hour_start"].map_partitions(sin_transformer, timeframe=24)
        speeds["hour_start_cos"] = speeds["hour_start"].map_partitions(cos_transformer, timeframe=24)

        speeds["week_sin"] = speeds["week"].map_partitions(sin_transformer, timeframe=52)
        speeds["week_cos"] = speeds["week"].map_partitions(cos_transformer, timeframe=52)

        speeds["day_sin"] = speeds["day"].map_partitions(sin_transformer, timeframe=31)
        speeds["day_cos"] = speeds["day"].map_partitions(cos_transformer, timeframe=31)

        speeds["month_sin"] = speeds["month"].map_partitions(sin_transformer, timeframe=12)
        speeds["month_cos"] = speeds["month"].map_partitions(cos_transformer, timeframe=12)

        print("\n\n")

        #------------------ Outliers filtering with Z-Score ------------------

        speeds = ZScore(speeds, "mean_speed")

        speeds = speeds.sort_values(by=["date"], ascending=True)

        # ------------------ TRP ID Target-Encoding ------------------

        speeds["trp_id"] = speeds["trp_id"].astype("category")

        encoder = LabelEncoder(use_categorical=True)  # Using a label encoder to encode TRP IDs to include the effect of the non-independence of observations from each other inside the forecasting models
        speeds = speeds.assign(trp_id_encoded=encoder.fit_transform(speeds["trp_id"]))  # The assign methods returns the dataframe obtained as input with the new column (in this case called "trp_id_encoded") added
        speeds.persist()

        #print("Encoded TRP IDs:", sorted(volumes["trp_id_encoded"].unique().compute()))

        # ------------------ Variables normalization ------------------

        scaler = MinMaxScaler()
        speeds[["mean_speed", "percentile_85", "coverage"]] = scaler.fit_transform(speeds[["mean_speed", "percentile_85", "coverage"]])


        #------------------ Creating lag features ------------------

        lag6h_column_names = [f"mean_speed_lag6h_{i}" for i in range(1, 7)]
        lag12h_column_names = [f"mean_speed_lag12_{i}" for i in range(1, 7)]
        lag24h_column_names = [f"mean_speed_lag24_{i}" for i in range(1, 7)]
        percentile_85_lag6_column_names = [f"percentile_85_lag{i}" for i in range(1, 7)]
        percentile_85_lag12_column_names = [f"percentile_85_lag{i}" for i in range(1, 7)]
        percentile_85_lag24_column_names = [f"percentile_85_lag{i}" for i in range(1, 7)]

        for idx, n in enumerate(lag6h_column_names): speeds[n] = speeds["mean_speed"].shift(idx + 6) #6 hours shift
        for idx, n in enumerate(lag12h_column_names): speeds[n] = speeds["mean_speed"].shift(idx + 12) #12 hours shift
        for idx, n in enumerate(lag24h_column_names): speeds[n] = speeds["mean_speed"].shift(idx + 24) #24 hours shift

        for idx, n in enumerate(percentile_85_lag6_column_names): speeds[n] = speeds["percentile_85"].shift(idx + 6) #6 hours shift
        for idx, n in enumerate(percentile_85_lag12_column_names): speeds[n] = speeds["percentile_85"].shift(idx + 12) #12 hours shift
        for idx, n in enumerate(percentile_85_lag24_column_names): speeds[n] = speeds["percentile_85"].shift(idx + 24) #24 hours shift

        #print(speeds.head(10))
        #print(speeds.dtypes)

        # ------------------ Creating dummy variables to address to the low value for traffic volumes in some years due to covid ------------------

        speeds["is_covid_year"] = (speeds['year'].isin(get_covid_years())).astype("int") #Creating a dummy variable which indicates if the average speed for a record has been affected by covid (because the traffic volume was recorded during one of the covid years)

        # ------------------ Dropping columns which won't be fed to the ML models ------------------

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


    @staticmethod
    def preprocess_data(target_datetime: datetime, max_days: int = 14, X_test=None, y_test=None): #TODO REMOVE =None AFTER TESTING
        """
        Parameters:
            target_datetime: the target datetime which the user wants to predict data for
            max_days: maximum number of days we want to predict
            X_test: the training dataset
            y_test: the testing dataset
        """
        #Function workflow:
            # 1. The user has to impute a target datetime for which it wants to predict data
            # 1.1 Since the predictions' confidence varies with how much in the future we want to predict, we'll set a limit on the number of days in future that the user may want to forecast
            #     This limit is set by default as 14 days, but can be modified by the specific max_days parameter
            # 2. Given the number of days in the future to predict we'll calculate the number of hours from the last datetime available for the trp which we want to predict the data for and the nth day in the future
            # 3. Once the number of hours to predict has been calculated we'll multiply it by 24, which means that for each hour to predict we'll use 24 hours in the past as reference
            # 4. We'll get exactly n rows from the TRP's individual data (where n = d * 24 and d is the number of days in the future to predict)
            # 5. We'll create n rows (where each row will be one specific hour of the future to predict)
            # 6. Finally, we'll return the new dataset ready to be fed to the model

        first_available_volumes_data_dt = read_metainfo_key(keys_map=["traffic_volumes", "start_date_iso"])
        last_available_volumes_data_dt = read_metainfo_key(keys_map=["traffic_volumes", "end_date_iso"])

        target_datetime = target_datetime.strftime("%Y-%m-%dT%H")

        print(datetime.now().strftime(dt_format))
        print(target_datetime)

        forecasting_window = pd.date_range(start=datetime.now(), end=target_datetime, freq="1h")

        #TODO CHECK IF DATE ISN'T BEFORE THE ONE OF THE LAST DATA AVAILABLE
        #TODO PARSE DATES OBTAINED IN THIS METHOD TO ONLY PRESERVE DATE AND HOUR
        # THEN DIVIDE THEM INTO YEAR, MONTH, DAY AND HOUR

        print(forecasting_window, "\n\n")

        return None










    def forecast_volumes(self):



        return None




























#TODO CREATE THE AvgSpeedForecaster CLASS AND THEN CREATE THE OnePointForecaster AND A2BForecaster CLASSES, THEY WILL JUST RETRIEVE AND USE THE PRE-MADE, TESTED AND EXPORTED MODELS


























