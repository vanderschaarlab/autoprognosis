# autoprognosis absolute
from autoprognosis.plugins.preprocessors import Preprocessors

default_classifiers_names = [
    "random_forest",
    "xgboost",
    "adaboost",
    "bagging",
    "qda",
    "lda",
    "logistic_regression",
    "lgbm",
    "catboost",
]
default_regressors_names = [
    "random_forest_regressor",
    "xgboost_regressor",
    "linear_regression",
    "catboost_regressor",
]

default_imputers_names = [
    "nop",
    "median",
    "mean",
    "most_frequent",
]
default_feature_scaling_names = Preprocessors(
    category="feature_scaling"
).list_available()
default_feature_selection_names = ["nop", "variance_threshold", "pca", "fast_ica"]
default_risk_estimation_names = [
    "survival_xgboost",
    "loglogistic_aft",
    "deephit",
    "cox_ph",
    "weibull_aft",
    "lognormal_aft",
    "coxnet",
]

percentile_val = 1.96
