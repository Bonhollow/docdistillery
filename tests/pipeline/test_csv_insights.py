import pandas as pd

from pipeline.csv_insights import extract_insights, insights_to_atomic_phrases


def test_extract_insights_trend():
    # Positive trend
    df = pd.DataFrame(
        {
            "val": [10, 11, 12, 13, 14, 15]  # Pure linear trend
        }
    )
    insights = extract_insights(df)
    trends = [i for i in insights if i["type"] == "trend"]
    assert len(trends) == 1
    assert trends[0]["evidence"]["trend_direction"] == "positive"
    assert trends[0]["evidence"]["column"] == "val"


def test_extract_insights_anomaly():
    # One outlier
    df = pd.DataFrame(
        {
            "val": [10, 10, 11, 10, 10, 100]  # 100 is clearly an outlier
        }
    )
    insights = extract_insights(df)
    anomalies = [i for i in insights if i["type"] == "anomaly"]
    assert len(anomalies) == 1
    assert anomalies[0]["evidence"]["count"] == 1
    assert 5 in anomalies[0]["evidence"]["indices"]


def test_extract_insights_correlation():
    df = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [2, 4, 6, 8, 10],  # Perfectly correlated
            "c": [5, 4, 3, 2, 1],  # Perfectly negatively correlated
        }
    )
    insights = extract_insights(df)
    corrs = [i for i in insights if i["type"] == "correlation"]
    # Should find (a,b), (a,c), (b,c)
    assert len(corrs) >= 2
    ids = [(i["evidence"]["column1"], i["evidence"]["column2"]) for i in corrs]
    assert ("a", "b") in ids or ("b", "a") in ids


def test_insights_to_phrases():
    insights = [{"type": "trend", "evidence": {"column": "X", "trend_direction": "positive", "slope": 0.5}}]
    phrases = insights_to_atomic_phrases(insights)
    assert len(phrases) == 1
    assert "column X" in phrases[0]
    assert "positive trend" in phrases[0]


def test_extract_insights_empty():
    df = pd.DataFrame({"a": []})
    assert extract_insights(df) == []
