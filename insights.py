from __future__ import annotations

import json
import os

import pandas as pd
import anthropic
from dotenv import load_dotenv

load_dotenv()

_MODEL = "claude-sonnet-4-6"
_MAX_REVIEWS_PER_BUCKET = 60  # cap per polarity bucket to keep token cost low

_PROMPT = """\
You are a product analyst helping a product manager understand user feedback.

Below are reviews from a mobile app, split by sentiment.

NEGATIVE REVIEWS:
{negative}

POSITIVE REVIEWS:
{positive}

Return a JSON object — no markdown, no extra text — with exactly this shape:
{{
  "top_complaints": ["point 1", "point 2", "point 3", "point 4", "point 5"],
  "loved_features": ["point 1", "point 2", "point 3", "point 4", "point 5"]
}}

Rules:
- 3 to 5 bullet points per key (more if patterns are clear).
- Each point is 1-2 sentences, specific, grounded in the actual review language.
- Quote short phrases from reviews where they are vivid or representative.
- Do NOT mention individual users or app version numbers.
"""


def generate_insights(df: pd.DataFrame) -> dict[str, list[str]]:
    """
    Send a capped sample of reviews to Claude and return structured insights.

    Args:
        df: Reviews DataFrame that already has sentiment_label column.

    Returns:
        Dict with keys "top_complaints" and "loved_features", each a list of strings.

    Raises:
        EnvironmentError: If ANTHROPIC_API_KEY is not set.
        RuntimeError:     On API or parse errors.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set. "
            "Add it to a .env file or set it as an environment variable."
        )

    negative_reviews = _sample_reviews(df, "Negative")
    positive_reviews = _sample_reviews(df, "Positive")

    prompt = _PROMPT.format(
        negative=negative_reviews or "(no negative reviews in this selection)",
        positive=positive_reviews or "(no positive reviews in this selection)",
    )

    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=_MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text.strip()
    except Exception as exc:
        raise RuntimeError(f"Claude API call failed: {exc}") from exc

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        # Attempt to salvage if the model wrapped the JSON in markdown fences
        import re
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            result = json.loads(match.group())
        else:
            raise RuntimeError(
                f"Could not parse LLM response as JSON.\nRaw response:\n{raw}"
            )

    return {
        "top_complaints": result.get("top_complaints", []),
        "loved_features": result.get("loved_features", []),
    }


def _sample_reviews(df: pd.DataFrame, label: str) -> str:
    bucket = df[df["sentiment_label"] == label]
    # Prefer reviews with more thumbs-up; fall back to recency
    if "thumbs_up" in bucket.columns:
        bucket = bucket.nlargest(_MAX_REVIEWS_PER_BUCKET, "thumbs_up")
    else:
        bucket = bucket.head(_MAX_REVIEWS_PER_BUCKET)
    lines = [f"- {text}" for text in bucket["review_text"].tolist()]
    return "\n".join(lines)
