
from tfs_utils import retrieve_n_ml_cpus




# ------------------- GridSearchCV auxiliary parameters and functions -------------------

#TODO SIMPLIFY THESE FIRST THREE DICTIONARIES

model_definitions = {
    "class_instance":{
        "RandomForestRegressor": RandomForestRegressor,
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor,
        "DecisionTreeRegressor": DecisionTreeRegressor,
    },
    "auxiliary_parameters": {
        "RandomForestRegressor": {"n_jobs": retrieve_n_ml_cpus(), "random_state": 100},
        "HistGradientBoostingRegressor": {
            "random_state": 100,
            "categorical_features": None,
        },
        "DecisionTreeRegressor": {"random_state": 100},
    }
}

best_params = {
    "traffic_volumes": {
        "RandomForestRegressor": 1,
        "HistGradientBoostingRegressor": 1,
        "DecisionTreeRegressor": 1,
    },
    "average_speed": {
        "RandomForestRegressor": 1,
        "HistGradientBoostingRegressor": 1,
        "DecisionTreeRegressor": 1,
    }
}




