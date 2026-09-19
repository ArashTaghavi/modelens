import pandas as pd
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import KFold, cross_validate


def compare_regularization(
    df: pd.DataFrame,
    target: str,
    features=None,
    alphas=None,
    l1_ratio=0.5,
    random_state=42,
):
    if features is None:
        features = df.drop(columns=[target]).columns.to_list()

    if alphas is None:
        alphas = [0.01, 0.1, 1, 10, 100]

    X = df[features]
    y = df[target]

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state,
    )

    results = []

    models = {
        "Linear Regression": LinearRegression(),
    }

    for alpha in alphas:
        models[f"Ridge alpha={alpha}"] = Ridge(alpha=alpha)

        models[f"Lasso alpha={alpha}"] = Lasso(
            alpha=alpha,
            max_iter=10000,
        )

        models[f"ElasticNet alpha={alpha}"] = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            max_iter=10000,
        )

    for name, model in models.items():

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
    results.append(
        {
            "Model": name,
            "Train R2": (
                f"{scores['train_r2'].mean():.4f} " f"± {scores['train_r2'].std():.4f}"
            ),
            "Test R2": (
                f"{scores['test_r2'].mean():.4f} " f"± {scores['test_r2'].std():.4f}"
            ),
            "Train RMSE": (
                f"{-scores['train_rmse'].mean():.4f} "
                f"± {scores['train_rmse'].std():.4f}"
            ),
            "Test RMSE": (
                f"{-scores['test_rmse'].mean():.4f} "
                f"± {scores['test_rmse'].std():.4f}"
            ),
        }
    )
    return (
        pd.DataFrame(results)
        .sort_values(
            by="Test R2 Mean",
            ascending=False,
        )
        .reset_index(drop=True)
    )
