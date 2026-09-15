from pathlib import Path

import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold

from modelens.utils.html_reporter import export_dataframe_to_html


def tune_model(
    df: pd.DataFrame,
    target: str,
    model,
    param_grid: dict,
    cv,
    scoring,
    features,
    export_html,
    html_path,
    random_state,
):
    if features is None:
        features = df.drop(columns=[target]).columns.to_list()

    if scoring is None:
        scoring = {
            "r2": "r2",
            "rmse": "neg_root_mean_squared_error",
        }

    X = df[features]
    y = df[target]

    kfold = KFold(
        n_splits=cv,
        shuffle=True,
        random_state=random_state,
    )

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring,
        refit="r2",
        cv=kfold,
        n_jobs=-1,
        return_train_score=True,
    )

    grid_search.fit(X, y)

    cv_results = pd.DataFrame(grid_search.cv_results_)

    results = pd.DataFrame(
        {
            "Params": cv_results["params"],
            "Train R2": (
                cv_results["mean_train_r2"].map("{:.4f}".format)
                + " ± "
                + cv_results["std_train_r2"].map("{:.4f}".format)
            ),
            "Test R2": (
                cv_results["mean_test_r2"].map("{:.4f}".format)
                + " ± "
                + cv_results["std_test_r2"].map("{:.4f}".format)
            ),
            "R2 Gap": (cv_results["mean_train_r2"] - cv_results["mean_test_r2"]),
            "Train RMSE": (
                (-cv_results["mean_train_rmse"]).map("{:.4f}".format)
                + " ± "
                + cv_results["std_train_rmse"].map("{:.4f}".format)
            ),
            "Test RMSE": (
                (-cv_results["mean_test_rmse"]).map("{:.4f}".format)
                + " ± "
                + cv_results["std_test_rmse"].map("{:.4f}".format)
            ),
            "RMSE Gap": (
                (-cv_results["mean_test_rmse"]) - (-cv_results["mean_train_rmse"])
            ),
            "Rank": cv_results["rank_test_r2"],
        }
    )

    results = results.sort_values("Rank").reset_index(drop=True)

    if export_html:
        _html_report(
            df=results,
            html_path=html_path,
        )

    return results


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
        title="Model Tuning",
    )
