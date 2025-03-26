from dask_ml.ensemble import BlockwiseVotingRegressor

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor



def get_blockwise_random_forest_regressor():

    sub_estimator = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=100)
    blockwise_random_forest = BlockwiseVotingRegressor(estimator=sub_estimator)

    return blockwise_random_forest


def get_blockwise_adaboost_regressor():

    sub_estimator = AdaBoostRegressor(n_estimators=200, random_state=100)
    blockwise_adaboost = BlockwiseVotingRegressor(estimator=sub_estimator)

    return blockwise_adaboost


def get_blockwise_bagging_regressor():

    sub_estimator = BaggingRegressor(n_estimators=200, n_jobs=-1, random_state=100)
    blockwise_bagging = BlockwiseVotingRegressor(estimator=sub_estimator)

    return blockwise_bagging


def get_blockwise_gradient_boosting_regressor():

    sub_estimator = GradientBoostingRegressor(n_estimators=200, random_state=100)
    blockwise_gradient_boosting = BlockwiseVotingRegressor(estimator=sub_estimator)

    return blockwise_gradient_boosting


def get_blockwise_decision_tree_regressor():

    sub_estimator = DecisionTreeRegressor(random_state=100)
    blockwise_decision_tree = BlockwiseVotingRegressor(estimator=sub_estimator)

    return blockwise_decision_tree


def get_blockwise_xgboost_regressor():

    sub_estimator = XGBRegressor(random_state=100)
    blockwise_xgboost = BlockwiseVotingRegressor(estimator=sub_estimator)

    return blockwise_xgboost














