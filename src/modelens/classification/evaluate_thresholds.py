import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


def evaluate_thresholds(
    df: pd.DataFrame,
    model,
    features: list,
    target: str,
    thresholds,
    test_size,
    random_state,
):
    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.1)

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
    fitted_model.fit(X_train, y_train)

    if not hasattr(fitted_model, "predict_proba"):
        raise ValueError("Threshold evaluation requires a model with predict_proba().")

    y_prob = fitted_model.predict_proba(X_test)[:, 1]

    results = []

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_test,
            y_pred,
            labels=[0, 1],
        ).ravel()

        results.append(
            {
                "Threshold": threshold,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(
                    y_test,
                    y_pred,
                    zero_division=0,
                ),
                "Recall": recall_score(
                    y_test,
                    y_pred,
                    zero_division=0,
                ),
                "F1": f1_score(
                    y_test,
                    y_pred,
                    zero_division=0,
                ),
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "TP": tp,
            }
        )

    return pd.DataFrame(results)
