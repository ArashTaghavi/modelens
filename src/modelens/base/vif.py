import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


def vif(df: pd.DataFrame, features):
    if features is None:
        features = df.select_dtypes(include="number").columns.to_list()

    X_vif = df[features]

    # intercept
    X_vif = sm.add_constant(X_vif)

    vif = pd.DataFrame(
        {
            "Feature": features,
            "VIF": [
                variance_inflation_factor(
                    X_vif,
                    X_vif.columns.get_loc(feature),
                )
                for feature in features
            ],
        }
    )

    return vif.sort_values("VIF", ascending=False)
