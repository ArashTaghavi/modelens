import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def vif_with_linear_regression(df: pd.DataFrame, features=None):
    if features is None:
        features = df.select_dtypes(include="number").columns.to_list()

    result = {}

    for feature in features:
        X = df[features].drop(columns=[feature])
        y = df[feature]

        model = LinearRegression()
        model.fit(X, y)

        y_pred = model.predict(X)

        r2 = r2_score(y, y_pred)

        if np.isclose(r2, 1):
            vif = np.inf
        else:
            vif = 1 / (1 - r2)

        result[feature] = {
            "R2 Score": r2,
            "VIF": vif,
        }

    return pd.DataFrame(result).T.sort_values(by="VIF", ascending=False)
