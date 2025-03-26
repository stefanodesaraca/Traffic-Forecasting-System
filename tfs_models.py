from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor



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


def get_xgboost_regressor():
    xgboost = XGBRegressor(random_state=100) #For early stopping we'll use the validation_fraction, n_iter_no_change, tol parameters
    return xgboost



models_gridsearch_parameters = {
    "RandomForestRegressor": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20, 30],
        "criterion": ["squared_error", "friedman_mse"]
    },
    "AdaBoostRegressor": {
        "n_estimators": [50, 100, 200],
        "loss": ["linear", "square"]
    },
    "BaggingRegressor": {
        "n_estimators": [10, 50, 100]
    },
    "GradientBoostingRegressor": {
        "n_estimators": [50, 100, 200],
        "loss": ["squared_error", "absolute_error"],
        "validation_fraction": [0.1, 0.2, 0.3],
        "n_iter_no_change": [5, 10, 20],
        "tol": [1e-4, 1e-3, 1e-2],
        "max_depth": [3, 5, 10]
    },
    "DecisionTreeRegressor": {
        "max_depth": [None, 10, 20, 30]
    },
    "XGBRegressor": {
        "n_estimators": [50, 100, 200],
        "validation_fraction": [0.1, 0.2, 0.3],
        "n_iter_no_change": [5, 10, 20],
        "tol": [1e-4, 1e-3, 1e-2]
    }
}


model_names_and_functions = {
    "RandomForestRegressor": get_random_forest_regressor,
    "BaggingRegressor": get_bagging_regressor,
    "DecisionTreeRegressor": get_decision_tree_regressor,
    "XGBRegressor": get_xgboost_regressor
    }

#TODO TEMPORARELY REMOVED ADABOOST AND GRADIENTBOOST SINCE IT DOESN'T TAKE NaNs NATIVELY, TRY HistGradientBoosting IN THE FUTURE


#"AdaBoostRegressor": get_adaboost_regressor,
#"GradientBoostingRegressor": get_gradient_boosting_regressor,







