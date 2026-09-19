import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate

from modelens.utils.html_reporter import export_dataframe_to_html


def evaluate_feature_removal_combinations(
    df: pd.DataFrame,
    features: list,
    candidates: list,
    target: str,
    random_state: int,
    model,
    export_html,
    file_name,
):
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state,
    )

    y = df[target]

    if model is None:
        model = LogisticRegression(
            max_iter=1000,
            random_state=random_state,
        )

    # -----------------------------------
    # Removal Combinations
    # -----------------------------------

    removal_combinations = [()]

    for n_remove in range(1, len(candidates) + 1):
        removal_combinations.extend(combinations(candidates, n_remove))

    results = []

    # -----------------------------------
    # Evaluate Combinations
    # -----------------------------------

    for removed_features in removal_combinations:

        current_features = [
            feature for feature in features if feature not in removed_features
        ]

        if not current_features:
            continue

        X = df[current_features]

        scores = cross_validate(
            clone(model),
            X,
            y,
            cv=cv,
            scoring={
                "accuracy": "accuracy",
                "precision": "precision",
                "recall": "recall",
                "f1": "f1",
                "roc_auc": "roc_auc",
            },
            return_train_score=True,
        )

        # -----------------------------------
        # Mean Values
        # -----------------------------------

        train_accuracy_mean = scores["train_accuracy"].mean()
        test_accuracy_mean = scores["test_accuracy"].mean()

        train_precision_mean = scores["train_precision"].mean()
        test_precision_mean = scores["test_precision"].mean()

        train_recall_mean = scores["train_recall"].mean()
        test_recall_mean = scores["test_recall"].mean()

        train_f1_mean = scores["train_f1"].mean()
        test_f1_mean = scores["test_f1"].mean()

        train_roc_auc_mean = scores["train_roc_auc"].mean()
        test_roc_auc_mean = scores["test_roc_auc"].mean()

        fit_time_mean = scores["fit_time"].mean()
        fit_time_std = scores["fit_time"].std()

        # -----------------------------------
        # Prediction Time
        # -----------------------------------

        predict_times = []

        for train_index, test_index in cv.split(X, y):

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

        accuracy_gap = train_accuracy_mean - test_accuracy_mean

        precision_gap = train_precision_mean - test_precision_mean

        recall_gap = train_recall_mean - test_recall_mean

        f1_gap = train_f1_mean - test_f1_mean

        roc_auc_gap = train_roc_auc_mean - test_roc_auc_mean

        # -----------------------------------
        # Result
        # -----------------------------------

        results.append(
            {
                "Removed Count": len(removed_features),
                "Removed Features": (
                    ", ".join(removed_features) if removed_features else "None"
                ),
                "Remaining Features": (", ".join(current_features)),
                "Accuracy": (
                    f"Train: {train_accuracy_mean:.3f} ± "
                    f"{scores['train_accuracy'].std():.3f}\n"
                    f"Test: {test_accuracy_mean:.3f} ± "
                    f"{scores['test_accuracy'].std():.3f}\n"
                    f"Gap: {accuracy_gap:.3f}"
                ),
                "Precision": (
                    f"Train: {train_precision_mean:.3f} ± "
                    f"{scores['train_precision'].std():.3f}\n"
                    f"Test: {test_precision_mean:.3f} ± "
                    f"{scores['test_precision'].std():.3f}\n"
                    f"Gap: {precision_gap:.3f}"
                ),
                "Recall": (
                    f"Train: {train_recall_mean:.3f} ± "
                    f"{scores['train_recall'].std():.3f}\n"
                    f"Test: {test_recall_mean:.3f} ± "
                    f"{scores['test_recall'].std():.3f}\n"
                    f"Gap: {recall_gap:.3f}"
                ),
                "F1": (
                    f"Train: {train_f1_mean:.3f} ± "
                    f"{scores['train_f1'].std():.3f}\n"
                    f"Test: {test_f1_mean:.3f} ± "
                    f"{scores['test_f1'].std():.3f}\n"
                    f"Gap: {f1_gap:.3f}"
                ),
                "ROC-AUC": (
                    f"Train: {train_roc_auc_mean:.3f} ± "
                    f"{scores['train_roc_auc'].std():.3f}\n"
                    f"Test: {test_roc_auc_mean:.3f} ± "
                    f"{scores['test_roc_auc'].std():.3f}\n"
                    f"Gap: {roc_auc_gap:.3f}"
                ),
                # Internal Values
                "_Test Accuracy Mean": test_accuracy_mean,
                "_Test Precision Mean": test_precision_mean,
                "_Test Recall Mean": test_recall_mean,
                "_Test F1 Mean": test_f1_mean,
                "_Test ROC-AUC Mean": test_roc_auc_mean,
                "_Fit Time Mean": fit_time_mean,
                "_Fit Time Std": fit_time_std,
                "_Predict Time Mean": predict_time_mean,
                "_Predict Time Std": predict_time_std,
            }
        )

    result = pd.DataFrame(results)

    # -----------------------------------
    # Sort
    # Best F1 first
    # More removed features second
    # -----------------------------------

    result = result.sort_values(
        by=[
            "_Test F1 Mean",
            "Removed Count",
        ],
        ascending=[
            False,
            False,
        ],
    ).reset_index(drop=True)

    # -----------------------------------
    # Best Values
    # -----------------------------------

    best_accuracy = float(result["_Test Accuracy Mean"].max())

    best_precision = float(result["_Test Precision Mean"].max())

    best_recall = float(result["_Test Recall Mean"].max())

    best_f1 = float(result["_Test F1 Mean"].max())

    best_roc_auc = float(result["_Test ROC-AUC Mean"].max())

    best_fit_time = float(result.iloc[0]["_Fit Time Mean"])

    best_predict_time = float(result.iloc[0]["_Predict Time Mean"])

    # -----------------------------------
    # Metric Loss
    # -----------------------------------

    metric_configs = [
        (
            "Accuracy",
            "_Test Accuracy Mean",
            best_accuracy,
        ),
        (
            "Precision",
            "_Test Precision Mean",
            best_precision,
        ),
        (
            "Recall",
            "_Test Recall Mean",
            best_recall,
        ),
        (
            "F1",
            "_Test F1 Mean",
            best_f1,
        ),
        (
            "ROC-AUC",
            "_Test ROC-AUC Mean",
            best_roc_auc,
        ),
    ]

    for metric_name, internal_column, best_value in metric_configs:

        values = result[internal_column].astype(float)

        loss = best_value - values

        if best_value != 0:
            loss_percent = loss / best_value * 100
        else:
            loss_percent = np.zeros(len(loss))

        result[f"Test {metric_name} Loss"] = [
            f"{percent:.2f}% ({value:.3f})"
            for percent, value in zip(
                loss_percent,
                loss,
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

    for _, row in result.iterrows():

        fit_time = float(row["_Fit Time Mean"])

        predict_time = float(row["_Predict Time Mean"])

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
            "Removed Count",
            "Removed Features",
            "Remaining Features",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC-AUC",
            "Test Accuracy Loss",
            "Test Precision Loss",
            "Test Recall Loss",
            "Test F1 Loss",
            "Test ROC-AUC Loss",
            "Fit / Predict Time (s)",
            "Speed vs Best",
        ]
    ]

    # -----------------------------------
    # HTML
    # -----------------------------------

    if export_html:

        html_path = (
            "html_reports/" "evaluate_feature_removal_combinations/" f"{file_name}.html"
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
        title="Evaluate Feature Removal Combinations",
    )
