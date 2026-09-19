from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def classification_models(smodels: list | None = None):

    models = {
        # Linear
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        random_state=42,
                        max_iter=1000,
                    ),
                ),
            ]
        ),

        # Distance
        "KNN": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier()),
            ]
        ),

        # Kernel
        "SVM": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    SVC(
                        probability=True,
                        random_state=42,
                    ),
                ),
            ]
        ),

        # Probabilistic
        "Naive Bayes": GaussianNB(),

        # Tree
        "Decision Tree": DecisionTreeClassifier(
            random_state=42,
        ),

        # Bagging
        "Random Forest": RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesClassifier(
            random_state=42,
            n_jobs=-1,
        ),

        # Boosting
        "AdaBoost": AdaBoostClassifier(
            random_state=42,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=42,
        ),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(
            random_state=42,
        ),
        "XGBoost": XGBClassifier(
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1,
        ),
        "LightGBM": LGBMClassifier(
            random_state=42,
            verbosity=-1,
            n_jobs=-1,
        ),
        "CatBoost": CatBoostClassifier(
            random_state=42,
            verbose=0,
        ),
    }

    if smodels is not None:
        models = {name: models[name] for name in smodels}

    return models
