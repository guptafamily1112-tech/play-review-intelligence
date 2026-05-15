from __future__ import annotations

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()


def _label(score: float) -> str:
    if score >= 0.05:
        return "Positive"
    if score <= -0.05:
        return "Negative"
    return "Neutral"


def score_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add sentiment_score (float, -1 to 1) and sentiment_label columns.
    Returns a new DataFrame; does not mutate the input.
    """
    df = df.copy()
    compound_scores = df["review_text"].apply(
        lambda text: _analyzer.polarity_scores(str(text))["compound"]
    )
    df["sentiment_score"] = compound_scores.round(3)
    df["sentiment_label"] = compound_scores.apply(_label)
    return df
