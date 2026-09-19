import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split


def analyze_prediction_group(
    df: pd.DataFrame,
    model,
    features: list,
    target: str,
    threshold,
    group,
    test_size,
    random_state,
):
    valid_groups = {"TP", "TN", "FP", "FN"}

    if group not in valid_groups:
        raise ValueError("group must be one of: TP, TN, FP, FN")

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
        raise ValueError(
            "Prediction group analysis requires a model with predict_proba()."
        )

    y_prob = fitted_model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    result = X_test.copy()

    result["Actual"] = y_test
    result["Predicted"] = y_pred
    result["Probability"] = y_prob

    result["Confidence"] = np.where(
        y_pred == 1,
        y_prob,
        1 - y_prob,
    )

    masks = {
        "TP": (result["Actual"] == 1) & (result["Predicted"] == 1),
        "TN": (result["Actual"] == 0) & (result["Predicted"] == 0),
        "FP": (result["Actual"] == 0) & (result["Predicted"] == 1),
        "FN": (result["Actual"] == 1) & (result["Predicted"] == 0),
    }

    group_df = result[masks[group]].copy()

    if group_df.empty:
        print(f"No samples found in group '{group}'.")
        return group_df, pd.DataFrame()

    group_df = group_df.sort_values(
        by="Confidence",
        ascending=False,
    )

    reference_group = {
        "FN": "TP",
        "FP": "TN",
        "TP": "FN",
        "TN": "FP",
    }[group]

    reference_df = result[masks[reference_group]]

    if reference_df.empty:
        comparison = pd.DataFrame(
            {
                f"{group} Mean": group_df[features].mean(),
            }
        )
    else:
        comparison = pd.DataFrame(
            {
                f"{group} Mean": group_df[features].mean(),
                f"{reference_group} Mean": reference_df[features].mean(),
            }
        )

        comparison["Difference"] = (
            comparison[f"{group} Mean"] - comparison[f"{reference_group} Mean"]
        )

    plt.figure(figsize=(12, 4))

    actual_0 = result[result["Actual"] == 0]
    actual_1 = result[result["Actual"] == 1]

    plt.scatter(
        actual_0["Probability"],
        np.zeros(len(actual_0)),
        alpha=0.5,
        label="Actual 0",
    )

    plt.scatter(
        actual_1["Probability"],
        np.ones(len(actual_1)),
        alpha=0.5,
        label="Actual 1",
    )

    plt.scatter(
        group_df["Probability"],
        group_df["Actual"],
        s=120,
        marker="x",
        linewidths=2,
        label=group,
    )

    plt.axvline(
        threshold,
        linestyle="--",
        label=f"Threshold = {threshold}",
    )

    plt.xlabel("Predicted Probability of Class 1")
    plt.ylabel("Actual Class")
    plt.yticks([0, 1])
    plt.xlim(0, 1)
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return group_df, comparison
