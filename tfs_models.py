from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor, AdaBoostRegressor, HistGradientBoostingRegressor

def get_random_forest_regressor() -> RandomForestRegressor:
    random_forest = RandomForestRegressor(n_jobs=-1, random_state=100) #Has the max_depth, criterion (use squared_error, friedman_mse) n_estimators parameters
    return random_forest


def get_adaboost_regressor() -> AdaBoostRegressor:
    ada_boost = AdaBoostRegressor(random_state=100) #Has the loss (use linear and square) and n_estimators parameters
    return ada_boost


def get_bagging_regressor() -> BaggingRegressor:
    bagging = BaggingRegressor(n_jobs=-1, random_state=100) #Has n_estimators parameter
    return bagging


def get_gradient_boosting_regressor() -> GradientBoostingRegressor:
    gradient_boosting = GradientBoostingRegressor(random_state=100) #Has the loss (use mean_squared_error and absolute_error), n_estimators, validation_fraction, n_iter_no_change, tol and max_depth parameters
    return gradient_boosting


def get_decision_tree_regressor() -> DecisionTreeRegressor:
    decision_tree = DecisionTreeRegressor(random_state=100) #Has max_depth parameter
    return decision_tree


def get_histgradientboosting_regressor() -> HistGradientBoostingRegressor:
    hist_gradient_boosting = HistGradientBoostingRegressor(random_state=100)  # Has the max_iter, max_depth parameters
    return hist_gradient_boosting


#TODO CHECK AGAIN AND/OR POTENTIALLY IMPROVE (CHANGE) THE PARAMETERS ONCE THE GridSearchCV IS EXECUTED ON THE WHOLE DATA AND WITH cv=10
models_gridsearch_parameters = {
    "RandomForestRegressor": {
        "n_estimators": [25, 40, 50, 70],
        "max_depth": [None, 3, 5, 10],
        "criterion": ["squared_error", "friedman_mse"],
        "ccp_alpha": [0] #ccp_alpha = 1 overfits
    },
    "BaggingRegressor": { #BaggingRegressor tends to overfit with whichever parameter it's fed with
        "n_estimators": [10, 20, 50, 70],
        "bootstrap_features": [False, True]
    },
    "DecisionTreeRegressor": {
        "max_depth": [None, 2, 3, 5]
    },
    "HistGradientBoostingRegressor": {
        "max_iter": [50, 70],
        "max_depth": [3, 5],
        "loss": ["squared_error", "absolute_error"],
        "validation_fraction": [0.25],
        "n_iter_no_change": [5, 10],
        "tol": [1e-4, 1e-3],
        "l2_regularization": [0]
    }
}


model_names_and_functions = {
    "RandomForestRegressor": get_random_forest_regressor,
    "BaggingRegressor": get_bagging_regressor,
    "HistGradientBoostingRegressor": get_histgradientboosting_regressor,
    "DecisionTreeRegressor": get_decision_tree_regressor
}


model_names_and_class_objects = {
    "RandomForestRegressor": RandomForestRegressor,
    "BaggingRegressor": BaggingRegressor,
    "HistGradientBoostingRegressor": HistGradientBoostingRegressor,
    "DecisionTreeRegressor": DecisionTreeRegressor
}

model_auxiliary_parameters = {
    "RandomForestRegressor": {"n_jobs": -1,
                              "random_state": 100},
    "BaggingRegressor": {"n_jobs": -1,
                         "random_state": 100},
    "HistGradientBoostingRegressor": {"random_state": 100},
    "DecisionTreeRegressor": {"random_state": 100}
}



best_parameters_by_model = {"RandomForestRegressor": 11,
                            "BaggingRegressor": 4,
                            "HistGradientBoostingRegressor": 26,
                            "DecisionTreeRegressor": 3}









