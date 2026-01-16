from typing import List


def build_story(insights_chunks: List[str], tone: str = "executive") -> str:
    """
    Transforms atomic insights into a coherent Markdown narrative.

    Args:
        insights_chunks (List[str]): Atomic phrases representing insights.
        tone (str): "executive" | "didactic" | "technical".

    Returns:
        str: A Markdown narrative string.
    """
    if not insights_chunks:
        return "# Data Narrative\n\nNo insights were provided for analysis."

    # 1. Group insights by type to ensure stable structure
    # Patterns based on csv_insights.py templates
    trends = [c for c in insights_chunks if "trend" in c.lower()]
    anomalies = [c for c in insights_chunks if "outlier" in c.lower()]
    correlations = [c for c in insights_chunks if "correlation" in c.lower()]

    all_grouped = trends + anomalies + correlations
    count = len(all_grouped)

    # 2. Tone Configuration
    tone_styles = {
        "executive": {
            "title_prefix": "Executive Brief:",
            "intro": (
                "The following analysis highlights critical performance markers and business impacts "
                "derived from the source data."
            ),
            "section_headers": {
                "trend": "## Development and Trajectory",
                "anomaly": "## Point-in-Time Disruptions",
                "correlation": "## Underlying Relationships",
            },
            "rec_header": "## Key Recommendations",
        },
        "didactic": {
            "title_prefix": "Educational Report:",
            "intro": (
                "Notably, by examining these data patterns, we can learn how different variables "
                "influence our overall system state."
            ),
            "section_headers": {
                "trend": "## Understanding Historical Trajectories",
                "anomaly": "## Learning from Abnormalities",
                "correlation": "## The Logic of Relationships",
            },
            "rec_header": "## Conceptual Takeaways",
        },
        "technical": {
            "title_prefix": "Statistical Summary:",
            "intro": (
                "Quantitative assessment of the dataset reveals specific metric variances and "
                "coefficient distributions as detailed below."
            ),
            "section_headers": {
                "trend": "## Regression and Slope Analysis",
                "anomaly": "## Deviation and IQR Detection",
                "correlation": "## Pearson Coefficient Matrix",
            },
            "rec_header": "## Metric-Based Adjustments",
        },
    }

    style = tone_styles.get(tone, tone_styles["executive"])

    # 3. Fallback for Small Data (< 3 insights)
    if count < 3:
        narrative = f"# {style['title_prefix']} Brief Data Summary\n\n"
        narrative += f"{style['intro']}\n\n"
        combined = " ".join(all_grouped)
        narrative += f"Initial observations indicate: {combined}\n\n"
        narrative += f"{style['rec_header']}\n"
        narrative += "- Monitor current data for further divergence.\n"
        return narrative

    # 4. Standard Flow (>= 3 insights)
    paragraphs = []

    # Title & Intro
    title = f"# {style['title_prefix']} Automated Data Story"
    intro = style["intro"]

    # Body Segments
    if trends:
        p = style["section_headers"]["trend"] + "\n"
        if tone == "executive":
            p += "The data points towards specific momentum. " + " ".join(trends)
        elif tone == "didactic":
            p += "Furthermore, we observe consistent movement over time. " + " ".join(trends)
        else:  # technical
            p += "Slope calculations confirm directional bias. " + " ".join(trends)
        paragraphs.append(p)

    if anomalies:
        p = style["section_headers"]["anomaly"] + "\n"
        if tone == "executive":
            p += "Critical deviations require immediate attention. " + " ".join(anomalies)
        elif tone == "didactic":
            p += "Identifying these anomalies helps us understand system stress. " + " ".join(anomalies)
        else:  # technical
            p += "Outlier detection using IQR bounds flagged significant variances. " + " ".join(anomalies)
        paragraphs.append(p)

    if correlations:
        p = style["section_headers"]["correlation"] + "\n"
        if tone == "executive":
            p += "Synergistic effects are evident between variables. " + " ".join(correlations)
        elif tone == "didactic":
            p += "It is useful to note how certain elements move in tandem. " + " ".join(correlations)
        else:  # technical
            p += "Bivariate correlation analysis yields high confidence scores. " + " ".join(correlations)
        paragraphs.append(p)

    # Recommendations (Clamp 1 to 3)
    rec_count = max(1, min(count, 3))
    rec_list = []
    if trends:
        rec_list.append("Capitalize on identified growth trajectories.")
    if anomalies:
        rec_list.append("Investigate system logs at identified outlier indices.")
    if correlations:
        rec_list.append("Leverage correlations for predictive modeling.")

    # Final recommedatations
    final_recs = rec_list[:rec_count]

    # Construction
    story = [title, intro] + paragraphs + [style["rec_header"]]
    for r in final_recs:
        story.append(f"- {r}")

    return "\n\n".join(story)
