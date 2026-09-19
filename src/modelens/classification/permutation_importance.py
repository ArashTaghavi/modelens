import base64
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
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
        stratify=y,
        random_state=random_state,
    )

    if model is None:
        model = LogisticRegression(
            max_iter=5000,
            random_state=random_state,
        )

    model.fit(
        X_train,
        y_train,
    )

    # -----------------------------
    # Permutation Importance - F1
    # -----------------------------

    perm_f1 = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=30,
        scoring="f1",
        random_state=random_state,
        n_jobs=-1,
    )

    importance_f1 = pd.DataFrame(
        {
            "feature": X_test.columns,
            "F1 decrease": perm_f1.importances_mean,
            "std": perm_f1.importances_std,
        }
    ).sort_values(
        by="F1 decrease",
        ascending=False,
    )

    # -----------------------------
    # Permutation Importance - ROC AUC
    # -----------------------------

    perm_roc_auc = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=30,
        scoring="roc_auc",
        random_state=random_state,
        n_jobs=-1,
    )

    importance_roc_auc = pd.DataFrame(
        {
            "feature": X_test.columns,
            "ROC AUC decrease": perm_roc_auc.importances_mean,
            "std": perm_roc_auc.importances_std,
        }
    ).sort_values(
        by="ROC AUC decrease",
        ascending=False,
    )

    # -----------------------------
    # Chart
    # -----------------------------

    chart = _plot(
        importance_f1=importance_f1,
        importance_roc_auc=importance_roc_auc,
    )

    # -----------------------------
    # Format: mean ± std
    # -----------------------------

    importance_f1_output = importance_f1.copy()

    importance_f1_output["F1 decrease"] = (
        importance_f1_output["F1 decrease"].map("{:.4f}".format)
        + " ± "
        + importance_f1_output["std"].map("{:.4f}".format)
    )

    importance_f1_output = importance_f1_output.drop(columns="std")

    importance_roc_auc_output = importance_roc_auc.copy()

    importance_roc_auc_output["ROC AUC decrease"] = (
        importance_roc_auc_output["ROC AUC decrease"].map("{:.4f}".format)
        + " ± "
        + importance_roc_auc_output["std"].map("{:.4f}".format)
    )

    importance_roc_auc_output = importance_roc_auc_output.drop(columns="std")

    # -----------------------------
    # HTML
    # -----------------------------

    if export_html:

        html_df = importance_f1_output.merge(
            importance_roc_auc_output,
            on="feature",
        )

        _html_report(
            df=html_df,
            html_path=html_path,
            chart=chart,
        )

    return (
        importance_f1_output,
        importance_roc_auc_output,
    )


def _plot(
    importance_f1: pd.DataFrame,
    importance_roc_auc: pd.DataFrame,
):

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(10, 12),
    )

    # -----------------------------
    # F1 Chart
    # -----------------------------

    ax_f1 = sns.barplot(
        data=importance_f1,
        x="F1 decrease",
        y="feature",
        ax=axes[0],
    )

    ax_f1.set_title("Permutation Importance — F1")

    ax_f1.set_xlabel("F1 Decrease")

    ax_f1.set_ylabel("Feature")

    for container in ax_f1.containers:

        ax_f1.bar_label(
            container,
            fmt="%.3f",
            padding=3,
        )

    # -----------------------------
    # ROC AUC Chart
    # -----------------------------

    ax_roc_auc = sns.barplot(
        data=importance_roc_auc,
        x="ROC AUC decrease",
        y="feature",
        ax=axes[1],
    )

    ax_roc_auc.set_title("Permutation Importance — ROC AUC")

    ax_roc_auc.set_xlabel("ROC AUC Decrease")

    ax_roc_auc.set_ylabel("Feature")

    for container in ax_roc_auc.containers:

        ax_roc_auc.bar_label(
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
        title="Classification Permutation Importance",
        chart=chart,
    )
