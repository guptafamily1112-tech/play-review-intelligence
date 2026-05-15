from __future__ import annotations

import datetime
from typing import Callable

import pandas as pd
from google_play_scraper import reviews, Sort
from google_play_scraper import app as _gplay_app
from google_play_scraper.exceptions import NotFoundError


_KEEP_COLS = {
    "reviewId": "review_id",
    "at": "date",
    "score": "rating",
    "content": "review_text",
    "thumbsUpCount": "thumbs_up",
    "reviewCreatedVersion": "app_version",
}

_BATCH_SIZE = 200


def fetch_reviews(
    app_id: str,
    target_count: int = 1000,
    days_back: int | None = None,
    lang: str = "en",
    country: str = "us",
    progress_callback: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """
    Fetch reviews for `app_id` from Google Play.

    When `days_back` is set, fetches until reviews older than the cutoff are
    reached (reviews are newest-first), up to `target_count` as a safety cap.
    When `days_back` is None, fetches up to `target_count` reviews.

    Raises:
        ValueError:   App not found, empty app_id, or no reviews in scope.
        RuntimeError: Unexpected scraper error.
    """
    if not app_id:
        raise ValueError("app_id must not be empty.")

    cutoff: datetime.datetime | None = None
    if days_back is not None:
        cutoff = (
            datetime.datetime.now(tz=datetime.timezone.utc)
            - datetime.timedelta(days=days_back)
        )

    raw_reviews: list[dict] = []
    continuation_token = None
    reached_cutoff = False

    try:
        while not reached_cutoff and len(raw_reviews) < target_count:
            batch, continuation_token = reviews(
                app_id,
                lang=lang,
                country=country,
                sort=Sort.NEWEST,
                count=_BATCH_SIZE,
                continuation_token=continuation_token,
            )

            if not batch:
                break

            if cutoff is not None:
                for r in batch:
                    at: datetime.datetime = r["at"]
                    if at.tzinfo is None:
                        at = at.replace(tzinfo=datetime.timezone.utc)
                    if at >= cutoff:
                        raw_reviews.append(r)
                    else:
                        reached_cutoff = True
                        break
            else:
                raw_reviews.extend(batch)

            if progress_callback:
                progress_callback(len(raw_reviews), target_count)

            if continuation_token is None:
                break

    except NotFoundError:
        raise ValueError(
            f"App '{app_id}' was not found on the Play Store. "
            "Check the package ID and try again."
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch reviews: {exc}") from exc

    if not raw_reviews:
        raise ValueError(
            f"No reviews found for '{app_id}'. "
            "The app may be new or have no public reviews in the selected time range."
        )

    return _to_dataframe(raw_reviews)


def fetch_app_info(app_id: str, lang: str = "en", country: str = "us") -> dict:
    """
    Fetch app metadata. Never raises — falls back gracefully on any error.
    Returns: title, store_rating, total_ratings, icon, genre, summary.
    """
    try:
        info = _gplay_app(app_id, lang=lang, country=country)
        return {
            "title": info.get("title", app_id),
            "store_rating": info.get("score"),
            "total_ratings": info.get("ratings"),
            "icon": info.get("icon"),
            "genre": info.get("genre"),
            "summary": info.get("summary") or "",
        }
    except Exception:
        return {
            "title": app_id,
            "store_rating": None,
            "total_ratings": None,
            "icon": None,
            "genre": None,
            "summary": "",
        }


def _to_dataframe(raw: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(raw)
    existing = {k: v for k, v in _KEEP_COLS.items() if k in df.columns}
    df = df[list(existing.keys())].rename(columns=existing)
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
    df["rating"] = df["rating"].astype(int)
    df["review_text"] = df["review_text"].fillna("").str.strip()
    df = df[df["review_text"] != ""].reset_index(drop=True)
    return df
