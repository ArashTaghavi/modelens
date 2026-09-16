from pathlib import Path

import pandas as pd


def export_dataframe_to_html(
    df: pd.DataFrame,
    path: Path,
    title: str = "Modelens Report",
    chart=None,
    second_df: pd.DataFrame | None = None,
    second_title: str | None = None,
):
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------
    # Main Table
    # -----------------------------------

    html_df = df.copy()

    for column in html_df.columns:
        html_df[column] = html_df[column].apply(
            lambda value: (
                value.replace("\n", "<br>")
                if isinstance(value, str)
                else value
            )
        )

    table_html = html_df.to_html(
        index=True,
        border=0,
        classes="data-table",
        escape=False,
    )

    # -----------------------------------
    # Second Table
    # -----------------------------------

    second_table_html = ""

    if second_df is not None:
        html_second_df = second_df.copy()

        for column in html_second_df.columns:
            html_second_df[column] = html_second_df[column].apply(
                lambda value: (
                    value.replace("\n", "<br>")
                    if isinstance(value, str)
                    else value
                )
            )

        second_table = html_second_df.to_html(
            index=False,
            border=0,
            classes="data-table",
            escape=False,
        )

        second_table_html = f"""
        <div class="card second-card">

            <h2>
                {second_title or "Additional Information"}
            </h2>

            {second_table}

        </div>
        """

    # -----------------------------------
    # Chart
    # -----------------------------------

    chart_html = ""

    if chart:
        chart_html = f"""
        <div class="card chart-card">

            <h2>
                Fit / Predict Time Comparison
            </h2>

            <img
                src="data:image/png;base64,{chart}"
                alt="Fit / Predict Time Comparison"
            >

        </div>
        """

    # -----------------------------------
    # HTML
    # -----------------------------------

    html = f"""
<!DOCTYPE html>
<html lang="en">

<head>

    <meta charset="UTF-8">

    <meta
        name="viewport"
        content="width=device-width, initial-scale=1.0"
    >

    <title>{title}</title>

    <style>

        * {{
            box-sizing: border-box;
        }}

        body {{
            margin: 0;
            padding: 40px;

            background: #f5f7fa;

            font-family:
                Arial,
                Helvetica,
                sans-serif;

            color: #1f2937;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}

        .header {{
            margin-bottom: 24px;
        }}

        .header h1 {{
            margin: 0;

            font-size: 28px;
            font-weight: 700;
        }}

        .header p {{
            margin-top: 8px;

            color: #6b7280;
            font-size: 14px;
        }}

        .card {{
            background: white;

            border-radius: 12px;

            padding: 20px;

            box-shadow:
                0 2px 8px
                rgba(0, 0, 0, 0.06);

            overflow-x: auto;
        }}

        .second-card {{
            margin-top: 24px;
        }}

        .second-card h2 {{
            margin-top: 0;
            margin-bottom: 20px;

            font-size: 20px;
        }}

        .data-table {{
            width: 100%;

            border-collapse: collapse;

            font-size: 14px;

            white-space: nowrap;
        }}

        .data-table thead th {{
            background: #111827;
            color: white;

            padding: 14px 16px;

            text-align: center;

            font-weight: 600;

            border: none;
        }}

        .data-table thead th:first-child {{
            text-align: left;
        }}

        .data-table tbody th {{
            text-align: left;

            font-weight: 600;

            padding: 13px 16px;

            border-bottom:
                1px solid #e5e7eb;
        }}

        .data-table tbody td {{
            text-align: center;

            padding: 13px 16px;

            border-bottom:
                1px solid #e5e7eb;

            line-height: 1.8;
        }}

        .data-table tbody tr:hover {{
            background: #f3f4f6;
        }}

        .data-table tbody tr:first-child {{
            background: #ecfdf5;
        }}

        .data-table tbody tr:first-child:hover {{
            background: #d1fae5;
        }}

        .chart-card {{
            margin-top: 24px;
            text-align: center;
        }}

        .chart-card h2 {{
            margin-top: 0;
            margin-bottom: 20px;

            font-size: 20px;
        }}

        .chart-card img {{
            display: block;

            max-width: 100%;
            height: auto;

            margin: 0 auto;
        }}

        .footer {{
            margin-top: 16px;

            font-size: 12px;

            color: #9ca3af;

            text-align: right;
        }}

    </style>

</head>

<body>

    <div class="container">

        <div class="header">

            <h1>{title}</h1>

            <p>
                Generated by Modelens
            </p>

        </div>

        <div class="card">
            {table_html}
        </div>

        {second_table_html}

        {chart_html}

        <div class="footer">
            Modelens ML Analysis
        </div>

    </div>

</body>

</html>
"""

    path.write_text(
        html,
        encoding="utf-8",
    )