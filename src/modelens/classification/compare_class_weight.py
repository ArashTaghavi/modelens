import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_validate


def compare_class_weight(
    df: pd.DataFrame,
    model,
    features: list,
    target: str,
    cv=5,
    random_state=42,
):
    if "class_weight" not in model.get_params():
        raise ValueError(f"{model.__class__.__name__} does not support class_weight.")

    X = df[features]
    y = df[target]

    kfold = StratifiedKFold(
        n_splits=cv,
        shuffle=True,
        random_state=random_state,
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    results = []

    for class_weight in [None, "balanced"]:
        current_model = clone(model)
        current_model.set_params(class_weight=class_weight)

        scores = cross_validate(
            current_model,
            X,
            y,
            cv=kfold,
            scoring=scoring,
            n_jobs=-1,
        )

        results.append(
            {
                "Class Weight": str(class_weight),
                "CV Accuracy Mean": scores["test_accuracy"].mean(),
                "CV Accuracy Std": scores["test_accuracy"].std(),
                "CV Precision Mean": scores["test_precision"].mean(),
                "CV Precision Std": scores["test_precision"].std(),
                "CV Recall Mean": scores["test_recall"].mean(),
                "CV Recall Std": scores["test_recall"].std(),
                "CV F1 Mean": scores["test_f1"].mean(),
                "CV F1 Std": scores["test_f1"].std(),
                "CV ROC AUC Mean": scores["test_roc_auc"].mean(),
                "CV ROC AUC Std": scores["test_roc_auc"].std(),
            }
        )

    return pd.DataFrame(results)
