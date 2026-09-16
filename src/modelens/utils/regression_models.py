from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    Ridge,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


def regression_models(smodels: list | None = None):

    models = {
        # Linear
        "Linear Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "Ridge": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Ridge()),
            ]
        ),
        "Lasso": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Lasso()),
            ]
        ),
        "ElasticNet": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", ElasticNet()),
            ]
        ),
        "Polynomial Regression": Pipeline(
            [
                ("polynomial", PolynomialFeatures(degree=2, include_bias=False)),
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        # Distance / Kernel
        "KNN": Pipeline([("scaler", StandardScaler()), ("KNN", KNeighborsRegressor())]),
        "SVR": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", SVR()),
            ]
        ),
        # Tree
        "Decision Tree": DecisionTreeRegressor(
            random_state=42,
        ),
        # Bagging
        "Random Forest": RandomForestRegressor(
            random_state=42,
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesRegressor(
            random_state=42,
            n_jobs=-1,
        ),
        # Boosting
        "AdaBoost": AdaBoostRegressor(
            random_state=42,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            random_state=42,
        ),
        "Hist Gradient Boosting": HistGradientBoostingRegressor(
            random_state=42,
        ),
        "XGBoost": XGBRegressor(
            random_state=42,
            n_jobs=-1,
        ),
        "LightGBM": LGBMRegressor(
            random_state=42,
            verbosity=-1,
            n_jobs=-1,
        ),
        "CatBoost": CatBoostRegressor(
            random_state=42,
            verbose=0,
        ),
    }

    if smodels is not None:
        models = {name: models[name] for name in smodels}
    return models
