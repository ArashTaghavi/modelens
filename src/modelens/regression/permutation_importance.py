import base64
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from modelens.utils.html_reporter import export_dataframe_to_html


def calculate_permutation_importance(
    df: pd.DataFrame,
    random_state: int,
    target: str,
    model=None,
    export_html=False,
    html_path="html_reports/permutation_importance.html",
):

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
    )

    if model is None:
        model = LinearRegression()

    model.fit(X_train, y_train)

    # -----------------------------
    # Permutation Importance - RMSE
    # -----------------------------

    perm_rmse = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=30,
        scoring="neg_root_mean_squared_error",
        random_state=random_state,
        n_jobs=-1,
    )

    importance_rmse = pd.DataFrame(
        {
            "feature": X_test.columns,
            "RMSE increase": perm_rmse.importances_mean,
            "std": perm_rmse.importances_std,
        }
    ).sort_values(
        by="RMSE increase",
        ascending=False,
    )

    # -----------------------------
    # Permutation Importance - R2
    # -----------------------------

    perm_r2 = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=30,
        scoring="r2",
        random_state=random_state,
        n_jobs=-1,
    )

    importance_r2 = pd.DataFrame(
        {
            "feature": X_test.columns,
            "R2 decrease": perm_r2.importances_mean,
            "std": perm_r2.importances_std,
        }
    ).sort_values(
        by="R2 decrease",
        ascending=False,
    )

    # -----------------------------
    # Chart
    # -----------------------------

    chart = _plot(
        importance_r2=importance_r2,
        importance_rmse=importance_rmse,
    )

    # -----------------------------
    # Format: mean ± std
    # -----------------------------

    importance_r2_output = importance_r2.copy()

    importance_r2_output["R2 decrease"] = (
        importance_r2_output["R2 decrease"].map("{:.4f}".format)
        + " ± "
        + importance_r2_output["std"].map("{:.4f}".format)
    )

    importance_r2_output = importance_r2_output.drop(columns="std")

    importance_rmse_output = importance_rmse.copy()

    importance_rmse_output["RMSE increase"] = (
        importance_rmse_output["RMSE increase"].map("{:.4f}".format)
        + " ± "
        + importance_rmse_output["std"].map("{:.4f}".format)
    )

    importance_rmse_output = importance_rmse_output.drop(columns="std")

    # -----------------------------
    # HTML
    # -----------------------------

    if export_html:

        html_df = importance_r2_output.merge(
            importance_rmse_output,
            on="feature",
        )

        _html_report(
            df=html_df,
            html_path=html_path,
            chart=chart,
        )


def _plot(
    importance_r2: pd.DataFrame,
    importance_rmse: pd.DataFrame,
):
    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(10, 12),
    )

    # -----------------------------
    # R2 Chart
    # -----------------------------

    ax_r2 = sns.barplot(
        data=importance_r2,
        x="R2 decrease",
        y="feature",
        ax=axes[0],
    )

    ax_r2.set_title("Permutation Importance — R²")
    ax_r2.set_xlabel("R² Decrease")
    ax_r2.set_ylabel("Feature")

    for container in ax_r2.containers:
        ax_r2.bar_label(
            container,
            fmt="%.3f",
            padding=3,
        )

    # -----------------------------
    # RMSE Chart
    # -----------------------------

    ax_rmse = sns.barplot(
        data=importance_rmse,
        x="RMSE increase",
        y="feature",
        ax=axes[1],
    )

    ax_rmse.set_title("Permutation Importance — RMSE")
    ax_rmse.set_xlabel("RMSE Increase")
    ax_rmse.set_ylabel("Feature")

    for container in ax_rmse.containers:
        ax_rmse.bar_label(
            container,
            fmt="%.3f",
            padding=3,
        )

    fig.tight_layout()

    # -----------------------------
    # Convert chart to Base64
    # -----------------------------

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
        title="Permutation Importance",
        chart=chart,
    )
