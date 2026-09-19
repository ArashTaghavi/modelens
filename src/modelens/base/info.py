import pandas as pd
from IPython.display import HTML, display


def info(df: pd.DataFrame):
    shape = df.shape
    null_counts = df.isnull().sum()
    null_counts = null_counts[null_counts > 0]
    duplicated = df.duplicated().sum()
    columns = df.columns.to_list()

    invalid_values = {
        "",
        "null",
        "none",
        "nan",
        "na",
        "n/a",
        "?",
        "-",
        "--",
    }

    invalid_data = {}

    for column in df.columns:
            values = df[column]

            invalid_count = values.isin(invalid_values).sum()

            if invalid_count > 0:
                invalid_data[column] = invalid_count

    display(HTML(f"""
    <div style="
        display:grid;
        grid-template-columns:repeat(2, minmax(250px, 1fr));
        gap:8px;
        font-family:Arial;
        font-size:12px;
    ">
        <div style="padding:8px 12px; background:#1e293b; border-radius:6px;">
            <b style="color:#38bdf8;">Shape</b>
            <div style="color:#e2e8f0; margin-top:4px;">{shape}</div>
        </div>

        <div style="padding:8px 12px; background:#1e293b; border-radius:6px;">
            <b style="color:#a78bfa;">Duplicated</b>
            <div style="color:#e2e8f0; margin-top:4px;">{duplicated}</div>
        </div>

        <div style="padding:8px 12px; background:#1e293b; border-radius:6px;">
            <b style="color:#34d399;">Null Counts</b>
            <pre style="color:#e2e8f0; margin:4px 0 0;">{null_counts}</pre>
        </div>

        <div style="padding:8px 12px; background:#1e293b; border-radius:6px;">
            <b style="color:#fbbf24;">Columns</b>
            <pre style="color:#e2e8f0; margin:4px 0 0;">{columns}</pre>
        </div>

        <div style="
            grid-column:1 / -1;
            padding:8px 12px;
            background:#1e293b;
            border-radius:6px;
        ">
            <b style="color:#fb7185;">Invalid Data</b>
            <pre style="color:#e2e8f0; margin:4px 0 0;">{invalid_data}</pre>
        </div>
    </div>
    """))
