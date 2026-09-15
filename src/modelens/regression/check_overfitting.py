import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_validate


def check_overfitting(
    df: pd.DataFrame,
    target: str,
    random_state: int,
    features,
    model,
):
    if features is None:
        features = df.drop(columns=[target]).columns.to_list()

    if model is None:
        model = LinearRegression()

    X = df[features]
    y = df[target]

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state,
    )

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring={
            "r2": "r2",
            "rmse": "neg_root_mean_squared_error",
        },
        return_train_score=True,
    )

    train_r2_mean = scores["train_r2"].mean()
    test_r2_mean = scores["test_r2"].mean()

    train_rmse_mean = -scores["train_rmse"].mean()
    test_rmse_mean = -scores["test_rmse"].mean()

    result = pd.DataFrame(
        {
            "Metric": ["R2", "RMSE"],
            "Train Mean": [
                train_r2_mean,
                train_rmse_mean,
            ],
            "Test Mean": [
                test_r2_mean,
                test_rmse_mean,
            ],
            "Gap": [
                train_r2_mean - test_r2_mean,
                test_rmse_mean - train_rmse_mean,
            ],
        }
    )

    return result
