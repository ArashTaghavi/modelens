import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, KFold, cross_validate

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
    file_name,
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

    # -----------------------------------
    # Grid Search
    # -----------------------------------

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

    best_params = grid_search.best_params_

    tuned_model = clone(model).set_params(**best_params)

    # -----------------------------------
    # Base / Tuned Models
    # -----------------------------------

    models = {
        "Base": clone(model),
        "Tuned": tuned_model,
    }

    results = {}

    for name, current_model in models.items():

        scores = cross_validate(
            current_model,
            X,
            y,
            cv=kfold,
            scoring=scoring,
            return_train_score=True,
        )

        train_r2_mean = scores["train_r2"].mean()
        test_r2_mean = scores["test_r2"].mean()

        train_rmse_mean = -scores["train_rmse"].mean()
        test_rmse_mean = -scores["test_rmse"].mean()

        fit_time_mean = scores["fit_time"].mean()

        # -----------------------------------
        # Prediction Time
        # -----------------------------------

        predict_times = []

        for train_index, test_index in kfold.split(X):

            X_train = X.iloc[train_index]
            y_train = y.iloc[train_index]
            X_test = X.iloc[test_index]

            fold_model = clone(current_model)

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
            # Internal values
            "_Test R2 Mean": test_r2_mean,
            "_Test RMSE Mean": test_rmse_mean,
            "_Fit Time Mean": fit_time_mean,
            "_Fit Time Std": scores["fit_time"].std(),
            "_Predict Time Mean": predict_time_mean,
            "_Predict Time Std": predict_time_std,
        }

    result = pd.DataFrame(results).T

    result.index.name = "Model"

    # -----------------------------------
    # Sort by Test R2
    # -----------------------------------

    result = result.sort_values(
        by=[
            "_Test R2 Mean",
            "_Fit Time Mean",
        ],
        ascending=[
            False,
            True,
        ],
    )

    # -----------------------------------
    # Best Model
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

    time_values = []

    for index in result.index:

        fit_mean = float(
            result.loc[
                index,
                "_Fit Time Mean",
            ]
        )

        fit_std = float(
            result.loc[
                index,
                "_Fit Time Std",
            ]
        )

        predict_mean = float(
            result.loc[
                index,
                "_Predict Time Mean",
            ]
        )

        predict_std = float(
            result.loc[
                index,
                "_Predict Time Std",
            ]
        )

        time_values.append(
            f"Fit: {fit_mean:.3f} ± "
            f"{fit_std:.3f}\n"
            f"Predict: {predict_mean:.3f} ± "
            f"{predict_std:.3f}"
        )

    result["Fit / Predict Time (s)"] = time_values

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

        speed_values.append(
            f"Predict: {predict_speed:.2f}x\n"
            f"Fit: {fit_speed:.2f}x"
        )

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
            "_Test R2 Mean",
            "_Test RMSE Mean",
            "_Fit Time Mean",
            "_Fit Time Std",
            "_Predict Time Mean",
            "_Predict Time Std",
        ]
    ]

    result = result.drop(
        columns=[
            "_Test R2 Mean",
            "_Test RMSE Mean",
            "_Fit Time Mean",
            "_Fit Time Std",
            "_Predict Time Mean",
            "_Predict Time Std",
        ]
    )

    # -----------------------------------
    # HTML Report
    # -----------------------------------

    if export_html:
        html_path = f"html_reports/tune_models/{file_name}.html"

        _html_report(
            df=result,
            best_params=best_params,
            html_path=html_path,
        )

        return (
            f"✓ Report generated successfully: {html_path}"
        )

    return result, best_params


def _html_report(
    df: pd.DataFrame,
    best_params: dict,
    html_path,
):
    html_path = Path(html_path)

    html_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_params_df = pd.DataFrame(
        {
            "Parameter": best_params.keys(),
            "Best Value": best_params.values(),
        }
    )

    export_dataframe_to_html(
        df=df,
        path=html_path,
        title="Base vs Tuned Model",
        second_df=best_params_df,
        second_title="Best Parameters",
    )