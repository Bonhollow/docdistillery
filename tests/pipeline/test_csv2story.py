from pipeline.csv2story import build_story


def test_build_story_structure():
    insights = [
        "The column sales shows a positive trend with a slope of 0.50.",
        "Found 1 outliers in cost at indices [5].",
        "Strong correlation identified between age and income (r=0.85).",
    ]

    story = build_story(insights, tone="executive")

    # Check Markdown structure
    assert story.startswith("# Executive Brief: Automated Data Story")
    assert "## Development and Trajectory" in story
    assert "## Point-in-Time Disruptions" in story
    assert "## Underlying Relationships" in story
    assert "## Key Recommendations" in story

    # Check paragraph count (Title + Intro + 3 body sections + Rec header + recs)
    # The join uses "\n\n", so we can split by that
    parts = story.split("\n\n")
    assert len(parts) >= 6


def test_build_story_tone_technical():
    insights = [
        "The column X shows a trend with slope 0.10.",
        "Outlier at index [10].",
        "Correlation between A and B (r=0.90).",
    ]
    story = build_story(insights, tone="technical")

    assert "# Statistical Summary:" in story
    assert "## Regression and Slope Analysis" in story
    assert "Slope calculations confirm directional bias" in story


def test_build_story_tone_didactic():
    insights = ["The column X shows a trend.", "Outlier at index [1].", "Correlation between A and B."]
    story = build_story(insights, tone="didactic")

    assert "# Educational Report:" in story
    assert "## Understanding Historical Trajectories" in story
    assert "Furthermore, we observe consistent movement" in story


def test_build_story_fallback():
    # Only 2 insights
    insights = ["Trend in sales.", "Outlier in cost."]
    story = build_story(insights)

    assert "Brief Data Summary" in story
    assert "Initial observations indicate:" in story


def test_build_story_empty():
    story = build_story([])
    assert "No insights were provided" in story
