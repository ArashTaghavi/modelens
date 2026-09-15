import base64
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from modelens.utils.html_reporter import export_dataframe_to_html


def residual_analysis(
    df: pd.DataFrame,
    target: str,
    features,
    model,
    test_size,
    random_state,
    residual_threshold,
    export_html,
    html_path,
):
    if features is None:
        features = df.drop(columns=[target]).columns.to_list()

    if model is None:
        model = LinearRegression()

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    residuals = y_test - y_pred

    result = X_test.copy()

    result["Actual"] = y_test
    result["Predicted"] = y_pred
    result["Residual"] = residuals
    result["Absolute Residual"] = residuals.abs()

    high_error_samples = result[
        result["Absolute Residual"] >= residual_threshold
    ].sort_values(
        "Absolute Residual",
        ascending=False,
    )

    chart = _plot(result)

    if export_html:
        _html_report(
            result=result,
            high_error_samples=high_error_samples,
            html_path=html_path,
            chart=chart,
        )

    return result, high_error_samples


def _plot(
    df: pd.DataFrame,
):
    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(10, 12),
    )

    # -----------------------------
    # Residual vs Predicted
    # -----------------------------

    sns.scatterplot(
        x=df["Predicted"],
        y=df["Residual"],
        ax=axes[0],
    )

    axes[0].axhline(
        0,
        linestyle="--",
    )

    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Residual")
    axes[0].set_title("Residuals vs Predicted")

    # -----------------------------
    # Residual Distribution
    # -----------------------------

    sns.histplot(
        df["Residual"],
        kde=True,
        ax=axes[1],
    )

    axes[1].axvline(
        0,
        linestyle="--",
    )

    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Residual Distribution")

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
    result: pd.DataFrame,
    high_error_samples: pd.DataFrame,
    html_path,
    chart,
):
    html_path = Path(html_path)

    html_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_html = result.copy()

    result_html.insert(
        0,
        "Group",
        "All Samples",
    )

    high_error_html = high_error_samples.copy()

    high_error_html.insert(
        0,
        "Group",
        "High Error",
    )

    html_df = pd.concat(
        [
            result_html,
            high_error_html,
        ]
    )

    export_dataframe_to_html(
        df=html_df,
        path=html_path,
        title="Residual Analysis",
        chart=chart,
    )
