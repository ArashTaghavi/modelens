import base64
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold, cross_validate

from modelens.utils.html_reporter import export_dataframe_to_html


def compare_models(
    df: pd.DataFrame,
    features,
    target: str,
    models: dict,
    export_html,
    html_path,
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

        results[name] = {
            "Train R2": (f"{train_r2_mean:.4f} ± " f"{scores['train_r2'].std():.4f}"),
            "Test R2": (f"{test_r2_mean:.4f} ± " f"{scores['test_r2'].std():.4f}"),
            "R2 Gap": train_r2_mean - test_r2_mean,
            "Train RMSE": (
                f"{train_rmse_mean:.4f} ± " f"{scores['train_rmse'].std():.4f}"
            ),
            "Test RMSE": (
                f"{test_rmse_mean:.4f} ± " f"{scores['test_rmse'].std():.4f}"
            ),
            "RMSE Gap": test_rmse_mean - train_rmse_mean,
            "Fit Time (s)": (
                f"{fit_time_mean:.4f} ± " f"{scores['fit_time'].std():.4f}"
            ),
            # Internal values
            "_Test R2 Mean": test_r2_mean,
            "_Fit Time Mean": fit_time_mean,
        }

    result = pd.DataFrame(results).T

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

    chart = _plot(result)

    result = result.drop(
        columns=[
            "_Test R2 Mean",
            "_Fit Time Mean",
        ]
    )

    if export_html:
        _html_report(
            df=result,
            html_path=html_path,
            chart=chart,
        )

    return result


def _plot(df: pd.DataFrame):
    plot_data = df["_Test R2 Mean"].astype(float).sort_values(ascending=False)

    values = plot_data.to_numpy(dtype=float)
    models = plot_data.index.astype(str).to_list()

    fig, ax = plt.subplots(figsize=(10, 7))

    bars = ax.barh(
        models,
        values,
    )

    ax.set_xlabel("Test R²")
    ax.set_ylabel("Model")
    ax.set_title("Test R² Comparison")

    ax.invert_yaxis()

    for bar, value in zip(bars, values):
        value = float(value)

        ax.text(
            value - 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.4f}",
            ha="right",
            va="center",
            color="white",
            fontweight="bold",
            fontsize=9,
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
    html_path,
    chart,
):
    html_path = Path(html_path)

    html_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    export_dataframe_to_html(
        df=df,
        path=html_path,
        title="Comparing Models",
        chart=chart,
    )
