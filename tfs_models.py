from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor, AdaBoostRegressor, HistGradientBoostingRegressor
from xgboost import XGBRegressor
#from xgboost.dask import DaskXGBRegressor

def get_random_forest_regressor():
    random_forest = RandomForestRegressor(n_jobs=-1, random_state=100) #Has the max_depth, criterion (use squared_error, friedman_mse) n_estimators parameters
    return random_forest


def get_adaboost_regressor():
    ada_boost = AdaBoostRegressor(random_state=100) #Has the loss (use linear and square) and n_estimators parameters
    return ada_boost


def get_bagging_regressor():
    bagging = BaggingRegressor(n_jobs=-1, random_state=100) #Has n_estimators parameter
    return bagging


def get_gradient_boosting_regressor():
    gradient_boosting = GradientBoostingRegressor(random_state=100) #Has the loss (use mean_squared_error and absolute_error), n_estimators, validation_fraction, n_iter_no_change, tol and max_depth parameters
    return gradient_boosting


def get_decision_tree_regressor():
    decision_tree = DecisionTreeRegressor(random_state=100) #Has max_depth parameter
    return decision_tree


def get_histgradientboosting_regressor():
    hist_gradient_boosting = HistGradientBoostingRegressor(random_state=100)  # Has the max_iter, max_depth parameters
    return hist_gradient_boosting


def get_xgboost_regressor():
    xgboost = XGBRegressor(random_state=100, max_depth=3) #For early stopping we'll use the validation_fraction, n_iter_no_change, tol parameters
    return xgboost



models_gridsearch_parameters = {
    "RandomForestRegressor": {
        "n_estimators": [25, 40, 50, 70],
        "max_depth": [None, 3, 5, 10],
        "criterion": ["squared_error", "friedman_mse"],
        "ccp_alpha": [0] #ccp_alpha = 1 overfits
    },
    "AdaBoostRegressor": {
        "n_estimators": [50, 70, 100],
        "loss": ["linear", "square"]
    },
    "BaggingRegressor": {
        "n_estimators": [10, 20, 50, 70],
        "bootstrap_features": [False, True]
    },
    "GradientBoostingRegressor": { #TODO CHECK FOR BETTER PARAMETERS TO AVOID OVERFITTING
        "n_estimators": [25, 40, 50, 100],
        "loss": ["squared_error", "absolute_error"],
        "validation_fraction": [0.1, 0.2, 0.3],
        "n_iter_no_change": [5, 10, 20],
        "tol": [1e-4, 1e-3, 1e-2],
        "max_depth": [3, 5, 10]
    },
    "DecisionTreeRegressor": {
        "max_depth": [None, 2, 3, 5]
    },
    "HistGradientBoostingRegressor": { #TODO CHECK FOR BETTER PARAMETERS TO AVOID OVERFITTING
        "max_iter": [50, 100, 200],
        "max_depth": [3, 5, 10],
        "loss": ["squared_error", "absolute_error"],
        "validation_fraction": [0.1, 0.2, 0.3],
        "n_iter_no_change": [5, 10, 20],
        "tol": [1e-4, 1e-3, 1e-2],
        "l2_regularization": [0, 0.5, 1.0]
    },
    "XGBRegressor": {
        "n_estimators": [50, 70, 100],
        "validation_fraction": [0.1, 0.2, 0.3],
        "n_iter_no_change": [5, 10, 20],
        "tol": [1e-4, 1e-3, 1e-2],
        "eta": [0, 0.2, 0.5]
    }
}


model_names_and_functions = {
    "RandomForestRegressor": get_random_forest_regressor,
    "BaggingRegressor": get_bagging_regressor,
    "DecisionTreeRegressor": get_decision_tree_regressor,
    "XGBRegressor": get_xgboost_regressor
}


model_names_and_class_objects = {
    "RandomForestRegressor": RandomForestRegressor,
    "BaggingRegressor": BaggingRegressor,
    "DecisionTreeRegressor": DecisionTreeRegressor,
    "XGBRegressor": XGBRegressor
}

model_auxiliary_parameters = {
    "RandomForestRegressor": {"n_jobs": -1,
                              "random_state": 100},
    "BaggingRegressor": {"n_jobs": -1,
                         "random_state": 100},
    "DecisionTreeRegressor": {"random_state": 100},
    "XGBRegressor": {"random_state": 100}
}













