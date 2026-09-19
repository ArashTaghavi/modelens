import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold,
    learning_curve,
)


def plot_learning_curve(
    df: pd.DataFrame,
    target: str,
    cv,
    random_state,
    features=None,
    model=None,
    train_sizes=None,
    scoring="f1",
):
    if features is None:
        features = df.drop(columns=[target]).columns.to_list()

    if model is None:
        model = LogisticRegression(
            max_iter=5000,
            random_state=random_state,
        )

    if train_sizes is None:
        train_sizes = [
            0.1,
            0.25,
            0.5,
            0.75,
            1.0,
        ]

    X = df[features]
    y = df[target]

    kfold = StratifiedKFold(
        n_splits=cv,
        shuffle=True,
        random_state=random_state,
    )

    sizes, train_scores, test_scores = learning_curve(
        estimator=model,
        X=X,
        y=y,
        cv=kfold,
        train_sizes=train_sizes,
        scoring=scoring,
        n_jobs=-1,
    )

    train_mean = train_scores.mean(axis=1)

    train_std = train_scores.std(axis=1)

    test_mean = test_scores.mean(axis=1)

    test_std = test_scores.std(axis=1)

    score_gap = train_mean - test_mean

    result = pd.DataFrame(
        {
            "Train Size": sizes,
            "Train Score": [
                f"{mean:.4f} ± {std:.4f}"
                for mean, std in zip(
                    train_mean,
                    train_std,
                )
            ],
            "Validation Score": [
                f"{mean:.4f} ± {std:.4f}"
                for mean, std in zip(
                    test_mean,
                    test_std,
                )
            ],
            "Score Gap": score_gap,
        }
    )

    # -----------------------------------
    # Plot
    # -----------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        sizes,
        train_mean,
        marker="o",
        label="Train",
    )

    plt.plot(
        sizes,
        test_mean,
        marker="o",
        label="Validation",
    )

    plt.fill_between(
        sizes,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.15,
    )

    plt.fill_between(
        sizes,
        test_mean - test_std,
        test_mean + test_std,
        alpha=0.15,
    )

    plt.xlabel("Training Samples")

    plt.ylabel(scoring.upper())

    plt.title(f"Learning Curve — {scoring.upper()}")

    plt.legend()

    plt.tight_layout()

    plt.show()

    return result
