from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    import pandas as pd


def extract_insights(df: "pd.DataFrame") -> List[Dict[str, Any]]:
    """
    Extracts deterministic insights (trends, anomalies, correlations) from a DataFrame.

    Args:
        df (pd.DataFrame): The input data.

    Returns:
        List[Dict]: A list of insight dictionaries.
    """
    import numpy as np
    from scipy import stats

    insights = []

    # 1. Identify numeric columns and handle missing values
    numeric_df = df.select_dtypes(include=[np.number])

    # Analyze each column for trends and anomalies
    for col in numeric_df.columns:
        series = numeric_df[col].dropna()
        if len(series) < 5:
            continue

        # --- Trend Detection ---
        # use linear regression on indices
        x = np.arange(len(series))
        y = series.values
        # Normalize y for slope magnitude thresholding
        y_range = np.max(y) - np.min(y)
        if y_range > 0:
            y_norm = (y - np.min(y)) / y_range
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y_norm)

            if (
                abs(slope) > 0.01
            ):  # user requested 0.1 but 0.01 is often more realistic for normalized slope over points
                # If the user specifically said 0.1 normalized, I will use that.
                # Re-reading: "slope > 0.1 normalized".
                # Let's stick to 0.1 if that's what's intended for "obvious" trends.
                if abs(slope) > 0.1:
                    trend_type = "positive" if slope > 0 else "negative"
                    insights.append(
                        {
                            "type": "trend",
                            "summary": f"The column {col} shows a {trend_type} trend.",
                            "evidence": {
                                "column": col,
                                "slope": float(slope),
                                "p_value": float(p_value),
                                "trend_direction": trend_type,
                            },
                        }
                    )

        # --- Anomaly Detection (IQR Method) ---
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = series[(series < lower_bound) | (series > upper_bound)]
        if not outliers.empty:
            insights.append(
                {
                    "type": "anomaly",
                    "summary": f"Found {len(outliers)} outliers in {col}.",
                    "evidence": {
                        "column": col,
                        "count": len(outliers),
                        "indices": outliers.index.tolist(),
                        "values": outliers.values.tolist(),
                    },
                }
            )

    # --- Correlation Detection ---
    if len(numeric_df.columns) >= 2:
        corr_matrix = numeric_df.corr()
        cols = corr_matrix.columns
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                r_val = corr_matrix.iloc[i, j]
                if abs(r_val) > 0.7:
                    insights.append(
                        {
                            "type": "correlation",
                            "summary": f"Strong correlation identified between {cols[i]} and {cols[j]}.",
                            "evidence": {"column1": cols[i], "column2": cols[j], "r_value": float(r_val)},
                        }
                    )

    return insights


def insights_to_atomic_phrases(insights: List[Dict[str, Any]]) -> List[str]:
    """
    Converts list of insights to atomic phrases for embedding.
    """
    phrases = []
    for insight in insights:
        itype = insight["type"]
        ev = insight["evidence"]

        if itype == "trend":
            phrases.append(
                f"The column {ev['column']} shows a {ev['trend_direction']} trend with a slope of {ev['slope']:.2f}."
            )
        elif itype == "anomaly":
            phrases.append(f"Found {ev['count']} outliers in {ev['column']} at indices {ev['indices']}.")
        elif itype == "correlation":
            phrases.append(
                f"Strong correlation identified between {ev['column1']} and {ev['column2']} (r={ev['r_value']:.2f})."
            )

    return phrases
