
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor



def get_blockwise_random_forest_regressor():

    random_forest = RandomForestRegressor(n_jobs=-1, random_state=100) #Has the max_depth, criterion (use squared_error, friedman_mse) n_estimators parameters

    return random_forest


def get_blockwise_adaboost_regressor():

    ada_boost = AdaBoostRegressor(random_state=100) #Has the loss (use linear and square) and n_estimators parameters

    return ada_boost


def get_blockwise_bagging_regressor():

    bagging = BaggingRegressor(n_jobs=-1, random_state=100) #Has n_estimators parameter

    return bagging


def get_blockwise_gradient_boosting_regressor():

    gradient_boosting = GradientBoostingRegressor(random_state=100) #Has the loss (use mean_squared_error and absolute_error), n_estimators, validation_fraction, n_iter_no_change, tol and max_depth parameters

    return gradient_boosting


def get_blockwise_decision_tree_regressor():

    decision_tree = DecisionTreeRegressor(random_state=100) #Has max_depth parameter

    return decision_tree


def get_blockwise_xgboost_regressor():

    xgboost = XGBRegressor(random_state=100) #For early stopping we'll use the validation_fraction, n_iter_no_change, tol parameters

    return xgboost














