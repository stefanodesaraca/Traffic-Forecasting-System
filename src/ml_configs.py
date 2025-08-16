from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor,
)

from utils import GlobalDefinitions


#Be aware that too low parameters could bring some models to stall while training, so don't go too low with the grid search parameters
grids = {
    GlobalDefinitions.TARGET_DATA["V"]: {
        "RandomForestRegressor": {
            "n_estimators": [200, 400],
            "max_depth": [
                20,
                40,
            ],  # NOTE max_depth ABSOLUTELY SHOULDN'T BE LESS THAN 20 OR 30.. FOR EXAMPLE: 10 CRASHES THE ALGORITHM
            "criterion": ["friedman_mse"],
            "ccp_alpha": [0, 0.00002],  # ccp_alpha = 1 overfits
        },
        "DecisionTreeRegressor": {
            "criterion": ["friedman_mse"],
            "max_depth": [None, 100, 200],
            "ccp_alpha": [0.0002, 0.00002],
        },
        "HistGradientBoostingRegressor": {
            "max_iter": [500, 1500],
            "max_depth": [None, 100],
            "loss": ["absolute_error"],
            "validation_fraction": [0.25],
            "n_iter_no_change": [20],
            "tol": [1e-7, 1e-4, 1e-3],
            "l2_regularization": [0.001, 0.0001],
        },
    },
    GlobalDefinitions.TARGET_DATA["MS"]: {
        "RandomForestRegressor": {
            "n_estimators": [100, 300],
            "max_depth": [
                40,
                70,
            ],  # NOTE max_depth ABSOLUTELY SHOULDN'T BE LESS THAN 20 OR 30.. FOR EXAMPLE: 10 CRASHES THE ALGORITHM
            "criterion": [
                "squared_error",
                "friedman_mse",
            ],
            "ccp_alpha": [0.002, 0.0002],  # ccp_alpha = 1 overfits
            "n_jobs": GlobalDefinitions.ML_CPUS,
            "random_state": 100
        },
        "DecisionTreeRegressor": {
            "random_state": 100,
            "criterion": ["squared_error", "friedman_mse"],
            "max_depth": [None, 30],
            "ccp_alpha": [0, 0.0002, 0.00002],
        },
        "HistGradientBoostingRegressor": {
            "random_state": 100,
            "categorical_features": None,
            "max_iter": [100, 200, 300],
            "max_depth": [None, 20, 50, 100],
            "loss": ["absolute_error"],
            "validation_fraction": [0.25],
            "n_iter_no_change": [20, 50],
            "tol": [1e-7, 1e-4, 1e-3],
            "l2_regularization": [0, 0.001, 0.0001]
        },
    }
}

model_mappings = {
        "RandomForestRegressor": RandomForestRegressor,
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor,
        "DecisionTreeRegressor": DecisionTreeRegressor,
}

