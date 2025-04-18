import os
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, HistGradientBoostingRegressor


def retrieve_n_ml_cpus() -> int:
    n_cpu = os.cpu_count()
    ml_dedicated_cores = int(n_cpu * 0.70)  # To avoid crashing while executing parallel computing in the GridSearchCV algorithm
    return ml_dedicated_cores


# ------------------- Functions that return the models themselves -------------------

def get_random_forest_regressor() -> RandomForestRegressor:
    random_forest = RandomForestRegressor(n_jobs=retrieve_n_ml_cpus(), random_state=100) #Has the max_depth, criterion (use squared_error, friedman_mse) n_estimators parameters
    return random_forest


def get_adaboost_regressor() -> AdaBoostRegressor:
    ada_boost = AdaBoostRegressor(random_state=100) #Has the loss (use linear and square) and n_estimators parameters
    return ada_boost


def get_gradient_boosting_regressor() -> GradientBoostingRegressor:
    gradient_boosting = GradientBoostingRegressor(random_state=100) #Has the loss (use mean_squared_error and absolute_error), n_estimators, validation_fraction, n_iter_no_change, tol and max_depth parameters
    return gradient_boosting


def get_decision_tree_regressor() -> DecisionTreeRegressor:
    decision_tree = DecisionTreeRegressor(random_state=100) #Has max_depth parameter
    return decision_tree


def get_histgradientboosting_regressor() -> HistGradientBoostingRegressor:
    hist_gradient_boosting = HistGradientBoostingRegressor(random_state=100) #Has the max_iter, max_depth parameters
    return hist_gradient_boosting



# ------------------- GridSearchCV volume parameters -------------------

volumes_models_gridsearch_parameters = {
    "RandomForestRegressor": {
        "n_estimators": [25, 40, 50, 70],
        "max_depth": [None, 3, 5, 10],
        "criterion": ["squared_error", "friedman_mse"],
        "ccp_alpha": [0] #ccp_alpha = 1 overfits
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

volumes_best_parameters_by_model = {"RandomForestRegressor": 12,
                                    "HistGradientBoostingRegressor": 26,
                                    "DecisionTreeRegressor": 3} #TODO CHOOSE SPECIFIC HYPERPARAMETERS FOR EACH MODEL AND FOR EACH ROAD CATEOGRY TO MAXIMISE PREDICTIVE ACCURACY



# ------------------- GridSearchCV auxiliary parameters and functions -------------------

model_names_and_functions = {
    "RandomForestRegressor": get_random_forest_regressor,
    "HistGradientBoostingRegressor": get_histgradientboosting_regressor,
    "DecisionTreeRegressor": get_decision_tree_regressor
}


model_names_and_class_objects = {
    "RandomForestRegressor": RandomForestRegressor,
    "HistGradientBoostingRegressor": HistGradientBoostingRegressor,
    "DecisionTreeRegressor": DecisionTreeRegressor
}

#Auxiliary parameters are common to all road categories' models
model_auxiliary_parameters = {
    "RandomForestRegressor": {"n_jobs": retrieve_n_ml_cpus(),
                              "random_state": 100},
    "HistGradientBoostingRegressor": {"random_state": 100, "categorical_features": None},
    "DecisionTreeRegressor": {"random_state": 100}
}



# ------------------- GridSearchCV average speed parameters -------------------
#Be aware that too low parameters could bring some models to stall while training, so don't go too low with the grid search parameters

speeds_models_gridsearch_parameters = {
    "RandomForestRegressor": {
        "n_estimators": [200, 300, 400, 500], #25, 40, 50, 70, 100, #TODO NEXT TRY WITH 1000
        "max_depth": [None, 3, 5, 10, 20, 30, 50, 100],
        "criterion": ["squared_error", "friedman_mse"],
        "ccp_alpha": [0, 0.001, 0.0001, 0.00001], #ccp_alpha = 1 overfits
        "warm_start": [True, False],
    },
    "DecisionTreeRegressor": {
        "max_depth": [None, 100, 200], #2, 3, 5, 7, 10, 20, 30, 40, 50,
        "criterion": ["squared_error", "absolute_error", "friedman_mse"],
        "max_features": [None, "sqrt", "log2"],
        "ccp_alpha": [0, 0.001, 0.0001, 0.005]
    },
    "HistGradientBoostingRegressor": {
        "max_iter": [20, 30, 40, 50, 60, 70, 100, 200, 300, 400, 500],
        "max_depth": [3, 5, 7, 10, 12, 30, 50, 100, 200],
        "loss": ["squared_error", "absolute_error"],
        "validation_fraction": [0.15, 0.25],
        "n_iter_no_change": [5, 10, 15, 20],
        "tol": [1e-7, 1e-5, 1e-4, 1e-3, 1e-2],
        "l2_regularization": [0, 0.001, 0.0001, 0.0005],
        "early_stopping": [True, "auto"],
        "learning_rate": [1, 0.5, 0.2, 0.1, 0.001, 0.002, 0.005]
    }
}


speeds_best_parameters_by_model = {"RandomForestRegressor": 2,
                                   "HistGradientBoostingRegressor": 44,
                                   "DecisionTreeRegressor": 1}


