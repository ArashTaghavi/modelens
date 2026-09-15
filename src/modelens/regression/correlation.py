import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import HTML, display


def correlation(
    df: pd.DataFrame,
    threshold,
    figsize,
    features,
):
    if features is None:
        features = df.select_dtypes(include="number").columns.to_list()

    corr = df.corr(numeric_only=True)[features]

    result = []
    suspicious_features = set()

    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            value = corr.iloc[i, j]
            if abs(value) >= threshold:
                suspicious_features.update([corr.columns[i], corr.columns[j]])
                result.append(
                    {
                        "Feature 1": corr.columns[i],
                        "Feature 2": corr.columns[j],
                        "Correlation": value,
                    }
                )
    plt.figure(figsize=figsize)
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.tight_layout()
    plt.show()

    display(HTML(f"""
    <div style="
        padding:8px 12px;
        background:#1e293b;
        border-radius:6px;
        font-family:Arial;
        font-size:12px;
        margin:6px 0;
    ">
        <b style="color:#f97316;">Suspicious Features</b>
        <div style="color:#e2e8f0; margin-top:4px;">
            {suspicious_features}
        </div>
    </div>
    """))

    return pd.DataFrame(result), suspicious_features
