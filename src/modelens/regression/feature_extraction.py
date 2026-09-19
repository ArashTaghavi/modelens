import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def feature_extraction(
    df: pd.DataFrame,
    target: str,
    features=None,
    variance_threshold=0.95,
):
    if features is None:
        features = df.drop(columns=[target]).columns.to_list()
    else:
        features = list(features)

    X = df[features]

    # -----------------------------
    # Scaling
    # -----------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # -----------------------------
    # PCA
    # -----------------------------

    pca = PCA(
        n_components=variance_threshold,
    )

    X_pca = pca.fit_transform(X_scaled)

    # -----------------------------
    # Extracted Features
    # -----------------------------

    component_names = [f"PC{i + 1}" for i in range(pca.n_components_)]

    extracted_df = pd.DataFrame(
        X_pca,
        columns=component_names,
        index=df.index,
    )

    extracted_df[target] = df[target]

    # -----------------------------
    # PCA Information
    # -----------------------------

    info = pd.DataFrame(
        {
            "Component": component_names,
            "Explained Variance": pca.explained_variance_ratio_,
            "Cumulative Variance": pca.explained_variance_ratio_.cumsum(),
        }
    )

    return extracted_df, info
