import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler


def feature_extraction(
    df: pd.DataFrame,
    target: str,
    features=None,
    n_components=None,
):
    if features is None:
        features = df.drop(columns=[target]).columns.to_list()
    else:
        features = list(features)

    X = df[features]
    y = df[target]

    # -----------------------------
    # Scaling
    # -----------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # -----------------------------
    # LDA
    # -----------------------------

    lda = LinearDiscriminantAnalysis(
        n_components=n_components,
    )

    X_lda = lda.fit_transform(
        X_scaled,
        y,
    )

    # -----------------------------
    # Extracted Features
    # -----------------------------

    component_names = [f"LD{i + 1}" for i in range(X_lda.shape[1])]

    extracted_df = pd.DataFrame(
        X_lda,
        columns=component_names,
        index=df.index,
    )

    extracted_df[target] = y

    # -----------------------------
    # LDA Information
    # -----------------------------

    info = pd.DataFrame(
        {
            "Component": component_names,
            "Explained Variance": lda.explained_variance_ratio_,
            "Cumulative Variance": lda.explained_variance_ratio_.cumsum(),
        }
    )

    return extracted_df, info
