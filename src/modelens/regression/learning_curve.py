import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, learning_curve


def plot_learning_curve(
    df: pd.DataFrame,
    target: str,
    cv,
    random_state,
    features=None,
    model=None,
    train_sizes=None,
):
    if features is None:
        features = df.drop(columns=[target]).columns.to_list()

    if model is None:
        model = LinearRegression()

    if train_sizes is None:
        train_sizes = [0.1, 0.25, 0.5, 0.75, 1.0]

    X = df[features]
    y = df[target]

    kfold = KFold(
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
        scoring="r2",
        n_jobs=-1,
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)

    test_mean = test_scores.mean(axis=1)
    test_std = test_scores.std(axis=1)

    r2_gap = train_mean - test_mean

    result = pd.DataFrame(
        {
            "Train Size": sizes,
            "Train R2": [
                f"{mean:.4f} ± {std:.4f}" for mean, std in zip(train_mean, train_std)
            ],
            "Test R2": [
                f"{mean:.4f} ± {std:.4f}" for mean, std in zip(test_mean, test_std)
            ],
            "R2 Gap": r2_gap,
        }
    )

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
    plt.ylabel("R2")
    plt.title("Learning Curve")

    plt.legend()
    plt.tight_layout()
    plt.show()

    return result
