import base64
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from modelens.utils.html_reporter import export_dataframe_to_html


def prediction_error_analysis(
    df: pd.DataFrame,
    target: str,
    features,
    model,
    test_size,
    random_state,
    threshold=0.5,
    confidence_threshold=0.8,
    export_html=False,
    html_path="html_reports/prediction_error_analysis.html",
):

    if features is None:
        features = df.drop(columns=[target]).columns.to_list()

    if model is None:
        model = LogisticRegression(
            max_iter=5000,
            random_state=random_state,
        )

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    fitted_model = clone(model)

    fitted_model.fit(
        X_train,
        y_train,
    )

    # -----------------------------------
    # Probability
    # -----------------------------------

    if not hasattr(
        fitted_model,
        "predict_proba",
    ):
        raise ValueError("Model must support predict_proba().")

    y_prob = fitted_model.predict_proba(X_test)[:, 1]

    # -----------------------------------
    # Prediction using threshold
    # -----------------------------------

    y_pred = (y_prob >= threshold).astype(int)

    # -----------------------------------
    # Result
    # -----------------------------------

    result = X_test.copy()

    result["Actual"] = y_test.values

    result["Predicted"] = y_pred

    result["Probability"] = y_prob

    # -----------------------------------
    # Prediction Group
    # -----------------------------------

    result["Group"] = np.select(
        [
            ((result["Actual"] == 1) & (result["Predicted"] == 1)),
            ((result["Actual"] == 0) & (result["Predicted"] == 0)),
            ((result["Actual"] == 0) & (result["Predicted"] == 1)),
            ((result["Actual"] == 1) & (result["Predicted"] == 0)),
        ],
        [
            "TP",
            "TN",
            "FP",
            "FN",
        ],
        default="Unknown",
    )

    # -----------------------------------
    # Confidence
    # -----------------------------------

    result["Confidence"] = np.where(
        result["Predicted"] == 1,
        result["Probability"],
        1 - result["Probability"],
    )

    # -----------------------------------
    # Errors
    # -----------------------------------

    error_samples = result[result["Actual"] != result["Predicted"]].copy()

    error_samples = error_samples.sort_values(
        "Confidence",
        ascending=False,
    )

    # -----------------------------------
    # High Confidence Errors
    # -----------------------------------

    high_confidence_errors = error_samples[
        error_samples["Confidence"] >= confidence_threshold
    ].copy()

    high_confidence_errors = high_confidence_errors.sort_values(
        "Confidence",
        ascending=False,
    )

    # -----------------------------------
    # Chart
    # -----------------------------------

    chart = _plot(
        result,
        threshold,
    )

    # -----------------------------------
    # HTML
    # -----------------------------------

    if export_html:

        _html_report(
            result=result,
            error_samples=error_samples,
            high_confidence_errors=(high_confidence_errors),
            html_path=html_path,
            chart=chart,
        )

    return (
        result,
        error_samples,
        high_confidence_errors,
    )


def _plot(
    df: pd.DataFrame,
    threshold: float,
):

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(10, 12),
    )

    # -----------------------------------
    # Probability vs Actual Class
    # -----------------------------------

    sns.scatterplot(
        data=df,
        x="Probability",
        y="Actual",
        hue="Group",
        ax=axes[0],
    )

    axes[0].axvline(
        threshold,
        linestyle="--",
    )

    axes[0].set_xlim(
        0,
        1,
    )

    axes[0].set_yticks([0, 1])

    axes[0].set_xlabel("Predicted Probability of Class 1")

    axes[0].set_ylabel("Actual Class")

    axes[0].set_title("Prediction Probability by Actual Class")

    # -----------------------------------
    # Probability Distribution
    # -----------------------------------

    sns.histplot(
        data=df,
        x="Probability",
        hue="Actual",
        kde=True,
        ax=axes[1],
    )

    axes[1].axvline(
        threshold,
        linestyle="--",
    )

    axes[1].set_xlim(
        0,
        1,
    )

    axes[1].set_xlabel("Predicted Probability of Class 1")

    axes[1].set_ylabel("Count")

    axes[1].set_title("Probability Distribution")

    fig.tight_layout()

    # -----------------------------------
    # Convert chart to Base64
    # -----------------------------------

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
    error_samples: pd.DataFrame,
    high_confidence_errors: pd.DataFrame,
    html_path,
    chart,
):

    html_path = Path(html_path)

    html_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------
    # All Samples
    # -----------------------------------

    result_html = result.copy()

    result_html.insert(
        0,
        "Analysis",
        "All Samples",
    )

    # -----------------------------------
    # Errors
    # -----------------------------------

    error_html = error_samples.copy()

    error_html.insert(
        0,
        "Analysis",
        "Prediction Error",
    )

    # -----------------------------------
    # High Confidence Errors
    # -----------------------------------

    high_confidence_html = high_confidence_errors.copy()

    high_confidence_html.insert(
        0,
        "Analysis",
        "High Confidence Error",
    )

    # -----------------------------------
    # Combine
    # -----------------------------------

    html_df = pd.concat(
        [
            result_html,
            error_html,
            high_confidence_html,
        ]
    )

    export_dataframe_to_html(
        df=html_df,
        path=html_path,
        title="Classification Prediction Error Analysis",
        chart=chart,
    )
