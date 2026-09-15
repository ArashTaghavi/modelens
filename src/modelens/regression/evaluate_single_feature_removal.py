from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_validate

from modelens.utils.html_reporter import export_dataframe_to_html


def evaluate_single_feature_removal(
    df: pd.DataFrame,
    target: str,
    random_state: int,
    export_html,
    html_path,
    model=None,
    features=None,
):
    if features is None:
        features = df.drop(columns=[target]).columns.to_list()
    else:
        features = list(features)

    if model is None:
        model = LinearRegression()

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state,
    )

    result = {}

    for removed_feature in [None] + features:

        if removed_feature is None:
            current_features = features
            name = "All Features"
        else:
            current_features = [
                feature for feature in features if feature != removed_feature
            ]

            name = f"Without {removed_feature}"

        if not current_features:
            continue

        X = df[current_features]
        y = df[target]

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

        # Mean values
        train_r2_mean = scores["train_r2"].mean()
        test_r2_mean = scores["test_r2"].mean()

        train_rmse_mean = -scores["train_rmse"].mean()
        test_rmse_mean = -scores["test_rmse"].mean()

        # Gaps
        r2_gap = train_r2_mean - test_r2_mean
        rmse_gap = test_rmse_mean - train_rmse_mean

        result[name] = {
            "Train R2": (f"{train_r2_mean:.4f} " f"± {scores['train_r2'].std():.4f}"),
            "Test R2": (f"{test_r2_mean:.4f} " f"± {scores['test_r2'].std():.4f}"),
            "R2 Gap": r2_gap,
            "Train RMSE": (
                f"{train_rmse_mean:.4f} " f"± {scores['train_rmse'].std():.4f}"
            ),
            "Test RMSE": (
                f"{test_rmse_mean:.4f} " f"± {scores['test_rmse'].std():.4f}"
            ),
            "RMSE Gap": rmse_gap,
            "_Test R2 Mean": test_r2_mean,
        }

    result_df = pd.DataFrame(result).T

    result = result_df.sort_values(
        by="_Test R2 Mean",
        ascending=False,
    ).drop(
        columns="_Test R2 Mean",
    )

    if export_html:
        _html_report(
            df=result,
            html_path=html_path,
        )

    return result


def _html_report(
    df: pd.DataFrame,
    html_path,
):
    html_path = Path(html_path)

    html_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    export_dataframe_to_html(
        df=df,
        path=html_path,
        title="Evaluate Single Feature Removal",
    )
