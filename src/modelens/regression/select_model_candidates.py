import pandas as pd


def select_model_candidates(
    results: pd.DataFrame,
    max_performance_loss: float = 0.05,
    performance_column: str = "Test R2",
    fit_time_column: str = "Fit Time (s)",
    feature_count_column=None,
):
    df = results.copy()

    def extract_mean(value):
        if isinstance(value, str):
            return float(value.split("±")[0].strip())

        return float(value)

    df["_Performance"] = df[performance_column].apply(extract_mean)
    df["_Fit Time"] = df[fit_time_column].apply(extract_mean)

    best_performance = df["_Performance"].max()
    best_index = df["_Performance"].idxmax()

    performance_threshold = best_performance * (1 - max_performance_loss)

    acceptable = df[df["_Performance"] >= performance_threshold].copy()

    balanced_index = acceptable["_Fit Time"].idxmin()

    if feature_count_column is not None and feature_count_column in acceptable.columns:
        minimal_index = acceptable.sort_values(
            by=[
                feature_count_column,
                "_Fit Time",
                "_Performance",
            ],
            ascending=[True, True, False],
        ).index[0]
    else:
        minimal_index = balanced_index

    selections = {
        "Max Performance": best_index,
        "Balanced": balanced_index,
        "Minimal": minimal_index,
    }

    rows = []

    for selection, index in selections.items():
        row = df.loc[index]

        performance_loss = best_performance - row["_Performance"]

        performance_loss_percent = performance_loss / best_performance * 100

        speedup = df.loc[best_index, "_Fit Time"] / row["_Fit Time"]

        rows.append(
            {
                "Selection": selection,
                "Model": index,
                "Test R2": row[performance_column],
                "Performance Loss": performance_loss,
                "Performance Loss %": performance_loss_percent,
                "Fit Time (s)": row[fit_time_column],
                "Speedup vs Best": speedup,
                **(
                    {"Feature Count": row[feature_count_column]}
                    if feature_count_column is not None
                    else {}
                ),
            }
        )

    return pd.DataFrame(rows)
