import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_validate

from modelens.utils.html_reporter import export_dataframe_to_html


def compare_feature_sets(
    df: pd.DataFrame,
    target,
    model,
    feature_sets: dict,
    cv,
    export_html,
    file_name,
    random_state,
):

    kfold = StratifiedKFold(
        n_splits=cv,
        shuffle=True,
        random_state=random_state,
    )

    results = {}

    # -----------------------------------
    # Evaluate Feature Sets
    # -----------------------------------

    for name, features in feature_sets.items():

        X = df[features]
        y = df[target]

        scores = cross_validate(
            clone(model),
            X,
            y,
            cv=kfold,
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

        for train_index, test_index in kfold.split(
            X,
            y,
        ):

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

        results[name] = {
            "Features": len(features),
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
            "ROC AUC": (
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
            "_Test ROC AUC Mean": test_roc_auc_mean,
            "_Fit Time Mean": fit_time_mean,
            "_Fit Time Std": fit_time_std,
            "_Predict Time Mean": predict_time_mean,
            "_Predict Time Std": predict_time_std,
        }

    result = pd.DataFrame(results).T

    result.index.name = "Feature Set"

    # -----------------------------------
    # Sort by Test F1
    # -----------------------------------

    result = result.sort_values(
        by=[
            "_Test F1 Mean",
            "_Test ROC AUC Mean",
        ],
        ascending=[
            False,
            False,
        ],
    )

    # -----------------------------------
    # Best Row
    # -----------------------------------

    best_test_accuracy = float(result.iloc[0]["_Test Accuracy Mean"])

    best_test_precision = float(result.iloc[0]["_Test Precision Mean"])

    best_test_recall = float(result.iloc[0]["_Test Recall Mean"])

    best_test_f1 = float(result.iloc[0]["_Test F1 Mean"])

    best_test_roc_auc = float(result.iloc[0]["_Test ROC AUC Mean"])

    best_fit_time = float(result.iloc[0]["_Fit Time Mean"])

    best_predict_time = float(result.iloc[0]["_Predict Time Mean"])

    # -----------------------------------
    # Test Accuracy Loss
    # -----------------------------------

    accuracy_values = result["_Test Accuracy Mean"].astype(float)

    accuracy_loss = best_test_accuracy - accuracy_values

    accuracy_loss_percent = accuracy_loss / best_test_accuracy * 100

    result["Test Accuracy Loss"] = [
        f"{percent:.2f}% ({loss:.3f})"
        for percent, loss in zip(
            accuracy_loss_percent,
            accuracy_loss,
        )
    ]

    # -----------------------------------
    # Test Precision Loss
    # -----------------------------------

    precision_values = result["_Test Precision Mean"].astype(float)

    precision_loss = best_test_precision - precision_values

    precision_loss_percent = precision_loss / best_test_precision * 100

    result["Test Precision Loss"] = [
        f"{percent:.2f}% ({loss:.3f})"
        for percent, loss in zip(
            precision_loss_percent,
            precision_loss,
        )
    ]

    # -----------------------------------
    # Test Recall Loss
    # -----------------------------------

    recall_values = result["_Test Recall Mean"].astype(float)

    recall_loss = best_test_recall - recall_values

    recall_loss_percent = recall_loss / best_test_recall * 100

    result["Test Recall Loss"] = [
        f"{percent:.2f}% ({loss:.3f})"
        for percent, loss in zip(
            recall_loss_percent,
            recall_loss,
        )
    ]

    # -----------------------------------
    # Test F1 Loss
    # -----------------------------------

    f1_values = result["_Test F1 Mean"].astype(float)

    f1_loss = best_test_f1 - f1_values

    f1_loss_percent = f1_loss / best_test_f1 * 100

    result["Test F1 Loss"] = [
        f"{percent:.2f}% ({loss:.3f})"
        for percent, loss in zip(
            f1_loss_percent,
            f1_loss,
        )
    ]

    # -----------------------------------
    # Test ROC AUC Loss
    # -----------------------------------

    roc_auc_values = result["_Test ROC AUC Mean"].astype(float)

    roc_auc_loss = best_test_roc_auc - roc_auc_values

    roc_auc_loss_percent = roc_auc_loss / best_test_roc_auc * 100

    result["Test ROC AUC Loss"] = [
        f"{percent:.2f}% ({loss:.3f})"
        for percent, loss in zip(
            roc_auc_loss_percent,
            roc_auc_loss,
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

        speed_values.append(
            f"Predict: " f"{predict_speed:.2f}x\n" f"Fit: " f"{fit_speed:.2f}x"
        )

    result["Speed vs Best"] = speed_values

    # -----------------------------------
    # Final Columns
    # -----------------------------------

    result = result[
        [
            "Features",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC AUC",
            "Test Accuracy Loss",
            "Test Precision Loss",
            "Test Recall Loss",
            "Test F1 Loss",
            "Test ROC AUC Loss",
            "Fit / Predict Time (s)",
            "Speed vs Best",
        ]
    ]

    # -----------------------------------
    # HTML
    # -----------------------------------

    if export_html:

        html_path = "html_reports/" "compare_feature_sets/" f"{file_name}.html"

        export_dataframe_to_html(
            df=result,
            path=Path(html_path),
            title=("Classification " "Feature Sets Comparison"),
        )

        return "✓ Report generated successfully: " f"{html_path}"

    return result
