import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_validate

from modelens.utils.html_reporter import export_dataframe_to_html


def evaluate_single_feature_removal(
    df: pd.DataFrame,
    target: str,
    random_state: int,
    export_html,
    file_name,
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

    results = {}

    # -----------------------------------
    # Evaluate Feature Removal
    # -----------------------------------

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
            clone(model),
            X,
            y,
            cv=cv,
            scoring={
                "r2": "r2",
                "rmse": "neg_root_mean_squared_error",
            },
            return_train_score=True,
        )

        # -----------------------------------
        # Mean Values
        # -----------------------------------

        train_r2_mean = scores["train_r2"].mean()
        test_r2_mean = scores["test_r2"].mean()

        train_rmse_mean = -scores["train_rmse"].mean()
        test_rmse_mean = -scores["test_rmse"].mean()

        fit_time_mean = scores["fit_time"].mean()
        fit_time_std = scores["fit_time"].std()

        # -----------------------------------
        # Prediction Time
        # -----------------------------------

        predict_times = []

        for train_index, test_index in cv.split(X):

            X_train = X.iloc[train_index]
            y_train = y.iloc[train_index]
            X_test = X.iloc[test_index]

            fold_model = clone(model)

            fold_model.fit(
                X_train,
                y_train,
            )

            # Warm-up
            fold_model.predict(X_test.iloc[:1])

            start = time.perf_counter()

            fold_model.predict(X_test)

            elapsed = time.perf_counter() - start

            predict_times.append(elapsed)

        predict_time_mean = float(np.mean(predict_times))

        predict_time_std = float(np.std(predict_times))

        # -----------------------------------
        # Gaps
        # -----------------------------------

        r2_gap = train_r2_mean - test_r2_mean

        rmse_gap = test_rmse_mean - train_rmse_mean

        # -----------------------------------
        # Result
        # -----------------------------------

        results[name] = {
            "R2": (
                f"Train: {train_r2_mean:.3f} ± "
                f"{scores['train_r2'].std():.3f}\n"
                f"Test: {test_r2_mean:.3f} ± "
                f"{scores['test_r2'].std():.3f}\n"
                f"Gap: {r2_gap:.3f}"
            ),
            "RMSE": (
                f"Train: {train_rmse_mean:.3f} ± "
                f"{scores['train_rmse'].std():.3f}\n"
                f"Test: {test_rmse_mean:.3f} ± "
                f"{scores['test_rmse'].std():.3f}\n"
                f"Gap: {rmse_gap:.3f}"
            ),
            # Internal Values
            "_Test R2 Mean": test_r2_mean,
            "_Test RMSE Mean": test_rmse_mean,
            "_Fit Time Mean": fit_time_mean,
            "_Fit Time Std": fit_time_std,
            "_Predict Time Mean": predict_time_mean,
            "_Predict Time Std": predict_time_std,
        }

    result = pd.DataFrame(results).T

    result.index.name = "Feature Removal"

    # -----------------------------------
    # Sort by Test R2
    # -----------------------------------

    result = result.sort_values(
        by="_Test R2 Mean",
        ascending=False,
    )

    # -----------------------------------
    # Best Row
    # -----------------------------------

    best_test_r2 = float(result.iloc[0]["_Test R2 Mean"])

    best_test_rmse = float(result.iloc[0]["_Test RMSE Mean"])

    best_fit_time = float(result.iloc[0]["_Fit Time Mean"])

    best_predict_time = float(result.iloc[0]["_Predict Time Mean"])

    # -----------------------------------
    # Test R2 Loss
    # -----------------------------------

    r2_values = result["_Test R2 Mean"].astype(float)

    r2_loss = best_test_r2 - r2_values

    r2_loss_percent = r2_loss / best_test_r2 * 100

    result["Test R2 Loss"] = [
        f"{percent:.2f}% ({loss:.3f})"
        for percent, loss in zip(
            r2_loss_percent,
            r2_loss,
        )
    ]

    # -----------------------------------
    # Test RMSE Loss
    # -----------------------------------

    rmse_values = result["_Test RMSE Mean"].astype(float)

    rmse_loss = rmse_values - best_test_rmse

    rmse_loss_percent = rmse_loss / best_test_rmse * 100

    result["Test RMSE Loss"] = [
        f"{percent:.2f}% ({loss:.3f})"
        for percent, loss in zip(
            rmse_loss_percent,
            rmse_loss,
        )
    ]

    # -----------------------------------
    # Fit / Predict Time
    # -----------------------------------

    result["Fit / Predict Time (s)"] = [
        (
            f"Fit: {fit_mean:.3f} ± "
            f"{fit_std:.3f}\n"
            f"Predict: {predict_mean:.3f} ± "
            f"{predict_std:.3f}"
        )
        for (
            fit_mean,
            fit_std,
            predict_mean,
            predict_std,
        ) in zip(
            result["_Fit Time Mean"],
            result["_Fit Time Std"],
            result["_Predict Time Mean"],
            result["_Predict Time Std"],
        )
    ]

    # -----------------------------------
    # Speed vs Best
    # -----------------------------------

    speed_values = []

    for index in result.index:

        fit_time = float(
            result.loc[
                index,
                "_Fit Time Mean",
            ]
        )

        predict_time = float(
            result.loc[
                index,
                "_Predict Time Mean",
            ]
        )

        if fit_time <= best_fit_time:
            fit_speed = best_fit_time / fit_time
        else:
            fit_speed = -(fit_time / best_fit_time)

        if predict_time <= best_predict_time:
            predict_speed = best_predict_time / predict_time
        else:
            predict_speed = -(predict_time / best_predict_time)

        speed_values.append(f"Predict: {predict_speed:.2f}x\n" f"Fit: {fit_speed:.2f}x")

    result["Speed vs Best"] = speed_values

    # -----------------------------------
    # Final Columns
    # -----------------------------------

    result = result[
        [
            "R2",
            "RMSE",
            "Test R2 Loss",
            "Test RMSE Loss",
            "Fit / Predict Time (s)",
            "Speed vs Best",
        ]
    ]

    # -----------------------------------
    # HTML
    # -----------------------------------

    if export_html:

        html_path = (
            "html_reports/" "evaluate_single_feature_removal/" f"{file_name}.html"
        )

        _html_report(
            df=result,
            html_path=html_path,
        )

        return f"✓ Report generated successfully: " f"{html_path}"

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
