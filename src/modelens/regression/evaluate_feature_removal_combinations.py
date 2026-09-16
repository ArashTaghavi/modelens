from itertools import combinations
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_validate

from modelens.utils.html_reporter import export_dataframe_to_html


def evaluate_feature_removal_combinations(
    df: pd.DataFrame,
    features: list,
    candidates: list,
    target: str,
    random_state: int,
    model,
    export_html,
    html_path,
):
    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state,
    )

    results = []

    y = df[target]

    # baseline + removal combinations
    removal_combinations = [()]

    for n_remove in range(1, len(candidates) + 1):
        removal_combinations.extend(combinations(candidates, n_remove))

    for removed_features in removal_combinations:

        current_features = [
            feature for feature in features if feature not in removed_features
        ]

        if not current_features:
            continue

        X = df[current_features]

        if model is None:
            model = LinearRegression()

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

        r2_train = (
            f"{scores['train_r2'].mean():.4f} " f"± {scores['train_r2'].std():.4f}"
        )

        r2_test = f"{scores['test_r2'].mean():.4f} " f"± {scores['test_r2'].std():.4f}"

        rmse_train = (
            f"{-scores['train_rmse'].mean():.4f} " f"± {scores['train_rmse'].std():.4f}"
        )

        rmse_test = (
            f"{-scores['test_rmse'].mean():.4f} " f"± {scores['test_rmse'].std():.4f}"
        )

        results.append(
            {
                "Removed Count": len(removed_features),
                "Removed Features": (
                    ", ".join(removed_features) if removed_features else "None"
                ),
                "Remaining Features": ", ".join(current_features),
                "Train R2": r2_train,
                "Test R2": r2_test,
                "Train RMSE": rmse_train,
                "Test RMSE": rmse_test,
            }
        )

    result = pd.DataFrame(results).sort_values(by="Test R2", ascending=False)

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
        title="Evaluate Feature Removal Combinations",
    )
