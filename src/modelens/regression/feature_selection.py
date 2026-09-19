import pandas as pd
from sklearn.base import clone
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold


def feature_selection(
    df: pd.DataFrame,
    target: str,
    model=None,
    features=None,
    scoring="r2",
    cv=5,
    random_state=42,
):
    if features is None:
        features = df.drop(columns=[target]).columns.to_list()
    else:
        features = list(features)

    if model is None:
        model = LinearRegression()

    X = df[features]
    y = df[target]

    cv_strategy = KFold(
        n_splits=cv,
        shuffle=True,
        random_state=random_state,
    )

    selector = RFECV(
        estimator=clone(model),
        step=1,
        cv=cv_strategy,
        scoring=scoring,
        n_jobs=-1,
    )

    selector.fit(X, y)

    result = pd.DataFrame(
        {
            "Feature": features,
            "Selected": selector.support_,
            "Ranking": selector.ranking_,
        }
    )

    result = result.sort_values(
        by=["Selected", "Ranking"],
        ascending=[False, True],
    ).reset_index(drop=True)

    return result
