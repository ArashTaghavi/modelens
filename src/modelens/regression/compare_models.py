import base64
import time
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold, cross_validate

from modelens.utils.html_reporter import export_dataframe_to_html


def compare_models(
    df: pd.DataFrame,
    features,
    target: str,
    models: dict,
    export_html,
    file_name,
    random_state,
):
    features = list(features)

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state,
    )

    X = df[features]
    y = df[target]

    results = {}

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

        train_r2_mean = scores["train_r2"].mean()
        test_r2_mean = scores["test_r2"].mean()

        train_rmse_mean = -scores["train_rmse"].mean()
        test_rmse_mean = -scores["test_rmse"].mean()

        fit_time_mean = scores["fit_time"].mean()

        # -----------------------------------
        # Prediction time
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

        results[name] = {
            "R2": (
                f"Train: {train_r2_mean:.3f} ± {scores['train_r2'].std():.3f}\n"
                f"Test: {test_r2_mean:.3f} ± {scores['test_r2'].std():.3f}\n"
                f"Gap: {r2_gap:.3f}"
            ),
            "RMSE": (
                f"Train: {train_rmse_mean:.3f} ± {scores['train_rmse'].std():.3f}\n"
                f"Test: {test_rmse_mean:.3f} ± {scores['test_rmse'].std():.3f}\n"
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
    # Best model = first row
    # -----------------------------------

    best_test_r2 = float(result.iloc[0]["_Test R2 Mean"])
    best_test_rmse = float(result.iloc[0]["_Test RMSE Mean"])
    best_fit_time = float(result.iloc[0]["_Fit Time Mean"])
    best_predict_time = float(result.iloc[0]["_Predict Time Mean"])

    # -----------------------------------
    # Test R2 relative loss
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
    # Test RMSE relative loss
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
    # Fit / Predict time
    # -----------------------------------

    time_values = []

    for index in result.index:

        fit_mean = float(result.loc[index, "_Fit Time Mean"])
        fit_std = float(result.loc[index, "_Fit Time Std"])

        predict_mean = float(result.loc[index, "_Predict Time Mean"])
        predict_std = float(result.loc[index, "_Predict Time Std"])

        time_values.append(
            f"Fit: {fit_mean:.3f} ± {fit_std:.3f}\n"
            f"Predict: {predict_mean:.3f} ± {predict_std:.3f}"
        )

    result["Fit / Predict Time (s)"] = time_values

    # -----------------------------------
    # Fit / Predict speed vs Best
    # -----------------------------------

    speed_values = []

    for index in result.index:

        fit_time = float(result.loc[index, "_Fit Time Mean"])
        predict_time = float(result.loc[index, "_Predict Time Mean"])

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
    # Final column order
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

    chart = _plot(result)

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

    if export_html:
        _html_report(
            df=result,
            file_name=f"html_reports/compare_models/{file_name}.html",
            chart=chart,
        )

    return f"✓ Report generated successfully: html_reports/compare_models/{file_name}.html"

def _plot(df: pd.DataFrame):
    plot_data = (
        df[
            [
                "_Fit Time Mean",
                "_Predict Time Mean",
            ]
        ]
        .astype(float)
        .copy()
    )

    plot_data = plot_data.sort_values(
        by=[
            "_Predict Time Mean",
            "_Fit Time Mean",
        ],
        ascending=True,
    )

    models = plot_data.index.astype(str).to_list()

    fit_values = plot_data["_Fit Time Mean"].to_numpy(dtype=float)
    predict_values = plot_data["_Predict Time Mean"].to_numpy(dtype=float)

    y = np.arange(len(models))
    bar_height = 0.35

    fig, ax = plt.subplots(figsize=(11, 8))

    fit_bars = ax.barh(
        y - bar_height / 2,
        fit_values,
        height=bar_height,
        label="Fit",
    )

    predict_bars = ax.barh(
        y + bar_height / 2,
        predict_values,
        height=bar_height,
        label="Predict",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(models)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Model")
    ax.set_title("Fit / Predict Time Comparison")

    ax.invert_yaxis()

    ax.legend()

    max_value = max(
        fit_values.max(),
        predict_values.max(),
    )

    ax.set_xlim(
        0,
        max_value * 1.18,
    )

    # -----------------------------------
    # Fit values
    # -----------------------------------

    for bar, value in zip(
        fit_bars,
        fit_values,
    ):
        ax.text(
            bar.get_width() + max_value * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.4f}s",
            ha="left",
            va="center",
            fontsize=8,
        )

    # -----------------------------------
    # Predict values
    # -----------------------------------

    for bar, value in zip(
        predict_bars,
        predict_values,
    ):
        ax.text(
            bar.get_width() + max_value * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.4f}s",
            ha="left",
            va="center",
            fontsize=8,
        )

    fig.tight_layout()

    buffer = BytesIO()

    fig.savefig(
        buffer,
        format="png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    buffer.seek(0)

    chart = base64.b64encode(buffer.read()).decode("utf-8")

    buffer.close()

    return chart


def _html_report(
    df: pd.DataFrame,
    file_name,
    chart,
):
    file_name = Path(file_name)

    file_name.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    export_dataframe_to_html(
        df=df,
        path=file_name,
        title="Comparing Models",
        chart=chart,
    )
