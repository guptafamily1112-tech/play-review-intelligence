from __future__ import annotations

import pandas as pd

# ── Complaint groups — keyword-based ─────────────────────────────────────────
COMPLAINT_GROUPS: dict[str, list[str]] = {
    "Too many ads": [
        "too many ads", "too much ads", "full of ads", "lots of ads",
        "popup", "pop-up", "pop up", "ads everywhere", "banner ad",
        "advertisements", "showing ads", "irritating ads",
    ],
    "App crashing": [
        "crash", "crashes", "crashed", "crashing",
        "force close", "force stop", "keeps crashing", "suddenly close",
        "closes automatically",
    ],
    "App not opening": [
        "not open", "won't open", "doesn't open", "not opening",
        "can't open", "cannot open", "won't load", "not loading",
        "does not open", "won't start", "stuck on loading",
    ],
    "Login / OTP issues": [
        "login", "log in", "sign in",
        "otp not", "not getting otp", "otp not received", "otp issue",
        "login problem", "login error", "unable to login", "can't login",
        "cannot login", "verification failed", "not receiving otp",
    ],
    "Slow / Loading issues": [
        "very slow", "so slow", "too slow", "loads slow",
        "loading forever", "takes forever", "lag", "lags", "lagging",
        "hangs", "freezes", "frozen", "not responding", "buffering",
    ],
    "Subscription / Payment": [
        "subscription", "payment failed", "refund", "charged",
        "deducted", "billing", "premium", "fee", "money deducted",
        "auto deduct", "unauthorized charge",
    ],
    "Fake / Scam concerns": [
        "fake", "scam", "fraud", "cheating", "misleading",
        "waste of money", "fraud app", "fake app", "not genuine",
    ],
    "Bugs / Errors": [
        "bug", "error", "glitch", "broken", "not working",
        "doesn't work", "error message", "showing error", "technical issue",
    ],
    "Poor quality / Useless": [
        "useless", "worst app", "terrible", "poor quality",
        "not useful", "waste of time", "pathetic", "horrible",
        "garbage", "trash", "awful",
    ],
    "Customer support": [
        "customer support", "customer service", "no support",
        "support team", "no response", "helpline", "not helpful",
        "support not responding",
    ],
    "Data / Privacy concerns": [
        "data breach", "privacy", "data stolen", "personal data",
        "data misuse", "account hacked", "security issue",
    ],
    "Update broke the app": [
        "after update", "new update", "latest update broke",
        "update ruined", "update issue", "since update",
    ],
}

# Specific verbatim phrases to look for in 1–2 star reviews
_NEGATIVE_PHRASES: list[str] = [
    "too many ads", "full of ads", "app crashes", "keeps crashing",
    "not opening", "won't open", "very slow", "so slow",
    "not working", "doesn't work", "can't login", "login problem",
    "otp not received", "waste of time", "waste of money",
    "fake app", "fraud", "scam", "payment failed", "refund",
    "customer support", "no response", "app hangs", "app freezes",
    "force close", "not loading", "subscription issue", "charged",
    "data not saved", "poor quality", "useless", "after update",
    "since update", "account hacked", "data stolen",
]


def key_signals(
    full_df: pd.DataFrame,
    recent_pct: float = 0.2,
    min_change_pp: float = 3.0,
    min_recent_mentions: int = 5,
    critical_rating_threshold: float = 2.5,
) -> list[dict]:
    """
    Return complaint groups that are spiking in the most recent slice of reviews.
    Each entry: complaint, recent_count, avg_rating, days_ago, change (pp), is_critical.
    """
    if full_df.empty or "complaint_categories" not in full_df.columns:
        return []

    recent_n = max(50, int(len(full_df) * recent_pct))
    recent = full_df.nlargest(recent_n, "date")

    overall_s = complaint_summary(full_df)
    recent_s = complaint_summary(recent)

    if overall_s.empty or recent_s.empty:
        return []

    merged = pd.merge(
        overall_s[["Complaint", "% of Reviews"]].rename(columns={"% of Reviews": "Overall %"}),
        recent_s[["Complaint", "% of Reviews", "Mentions"]].rename(
            columns={"% of Reviews": "Recent %", "Mentions": "Recent Mentions"}
        ),
        on="Complaint",
        how="inner",
    ).fillna(0)
    merged["Change"] = (merged["Recent %"] - merged["Overall %"]).round(1)

    signals = []
    for _, row in merged.iterrows():
        if row["Change"] < min_change_pp or row["Recent Mentions"] < min_recent_mentions:
            continue

        name = row["Complaint"]
        in_recent = recent[recent["complaint_categories"].apply(lambda cats: name in cats)]
        avg_rating = round(float(in_recent["rating"].mean()), 1) if not in_recent.empty else 3.0
        oldest = in_recent["date"].min() if not in_recent.empty else None
        days_ago = int((pd.Timestamp.now() - oldest).days) if oldest is not None else None

        signals.append({
            "complaint": name,
            "recent_count": int(row["Recent Mentions"]),
            "change": row["Change"],
            "avg_rating": avg_rating,
            "days_ago": days_ago,
            "is_critical": avg_rating <= critical_rating_threshold,
        })

    signals.sort(key=lambda x: (-x["change"], x["avg_rating"]))
    return signals


def detect_complaints(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add complaint_categories (list[str]) to each row.
    Run once after fetch — all downstream functions read this pre-computed column.
    """
    texts = df["review_text"].str.lower().fillna("").tolist()
    categories = [
        [name for name, kws in COMPLAINT_GROUPS.items() if any(kw in text for kw in kws)]
        for text in texts
    ]
    result = df.copy()
    result["complaint_categories"] = categories
    return result


def complaint_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate complaint_categories into a summary table.
    Returns: Complaint | Mentions | % of Reviews | _samples (list, internal)
    """
    if df.empty or "complaint_categories" not in df.columns:
        return pd.DataFrame(columns=["Complaint", "Mentions", "% of Reviews", "_samples"])

    total = len(df)
    rows = []
    for name in COMPLAINT_GROUPS:
        mask = df["complaint_categories"].apply(lambda cats: name in cats)
        count = int(mask.sum())
        if count == 0:
            continue
        # Prefer low-rated reviews as the most representative samples
        samples = df[mask].nsmallest(3, "rating")["review_text"].tolist()
        rows.append({
            "Complaint": name,
            "Mentions": count,
            "% of Reviews": round(100 * count / total, 1),
            "_samples": samples,
        })

    if not rows:
        return pd.DataFrame(columns=["Complaint", "Mentions", "% of Reviews", "_samples"])
    return pd.DataFrame(rows).sort_values("Mentions", ascending=False).reset_index(drop=True)


def top_negative_phrases(df: pd.DataFrame, top_n: int = 12) -> list[tuple[str, int]]:
    """Count curated negative phrases in 1–2 star reviews, ordered by frequency."""
    low = df[df["rating"].isin([1, 2])]
    if low.empty:
        return []
    texts = low["review_text"].str.lower().tolist()
    results = [
        (phrase, sum(1 for t in texts if phrase in t))
        for phrase in _NEGATIVE_PHRASES
    ]
    results = [(p, c) for p, c in results if c > 0]
    results.sort(key=lambda x: -x[1])
    return results[:top_n]


def rising_complaints(full_df: pd.DataFrame, recent_pct: float = 0.2) -> pd.DataFrame:
    """
    Compare complaint frequency in the most recent slice of reviews vs overall.
    Always receives the full (unfiltered, date-sorted) df to preserve temporal integrity.
    Returns complaints where recent % > overall %, sorted by delta descending.
    """
    recent_n = max(50, int(len(full_df) * recent_pct))
    recent = full_df.nlargest(recent_n, "date")

    overall = complaint_summary(full_df)
    recent_s = complaint_summary(recent)

    if overall.empty or recent_s.empty:
        return pd.DataFrame()

    merged = pd.merge(
        overall[["Complaint", "% of Reviews"]].rename(columns={"% of Reviews": "Overall %"}),
        recent_s[["Complaint", "% of Reviews"]].rename(columns={"% of Reviews": "Recent %"}),
        on="Complaint",
        how="outer",
    ).fillna(0)

    merged["Change"] = (merged["Recent %"] - merged["Overall %"]).round(1)
    return (
        merged[merged["Change"] > 0]
        .sort_values("Change", ascending=False)
        .reset_index(drop=True)
    )


# ── Positive theme keywords ───────────────────────────────────────────────────
PRAISE_GROUPS: dict[str, list[str]] = {
    "Fast & smooth performance": [
        "fast", "quick", "smooth", "snappy", "responsive", "no lag",
        "works great", "works perfectly", "no issues", "stable",
    ],
    "Easy to use": [
        "easy to use", "user friendly", "simple", "intuitive",
        "easy to navigate", "straightforward",
    ],
    "Great value / Deals": [
        "great price", "affordable", "good deal", "value for money",
        "cheap", "discount", "best price", "worth it", "great offers",
    ],
    "Reliable & accurate": [
        "reliable", "accurate", "dependable", "consistent",
        "always works", "never fails",
    ],
    "Good selection / Content": [
        "good selection", "wide range", "variety", "lots of",
        "many options", "good collection", "great content",
    ],
    "Clean design / UI": [
        "beautiful", "nice design", "clean", "love the look",
        "great ui", "looks good", "modern",
    ],
    "Fast delivery": [
        "fast delivery", "quick delivery", "on time",
        "delivered on time", "quick shipping", "same day",
    ],
    "Helpful support": [
        "great support", "helpful support", "quick response",
        "resolved my issue", "customer care",
    ],
}

_REQUEST_SIGNALS: list[str] = [
    "wish", "would love", "should add", "please add", "would like",
    "needs a", "need a", "missing", "would be nice", "if only",
    "add option", "add feature", "could add", "bring back",
    "great but", "good but", "love it but", "nice but", "awesome but",
    "one thing", "only issue", "only problem", "only complaint",
    "but wish", "hope they", "suggestion",
]

_FEATURE_THEMES: dict[str, list[str]] = {
    "Dark mode": ["dark mode", "dark theme", "night mode"],
    "Better search & filters": ["search", "filter", "sort by"],
    "Offline mode": ["offline", "without internet", "no internet"],
    "Performance improvements": ["faster", "speed", "smoother", "battery", "optimize"],
    "More customization": ["customiz", "personaliz", "settings", "preference"],
    "Notifications": ["notification", "notify", "alert", "reminder"],
    "Download / Export": ["download", "export", "save offline"],
    "UI / Design improvements": ["ui", "interface", "design", "layout", "redesign"],
    "More content / Options": ["more content", "more options", "more features", "expand"],
    "Language support": ["language", "translate", "regional", "local"],
    "Better recommendations": ["recommend", "suggest", "personalized", "for me"],
    "Undo / Edit history": ["undo", "history", "go back", "edit history"],
}

# Surfaced in PM investigation hints for critical issue cards
SIGNAL_FRAMING: dict[str, dict] = {
    "App crashing": {
        "headline": "App crashes are breaking core user journeys",
        "context": "Users report sudden closures mid-session, damaging trust and retention.",
    },
    "App not opening": {
        "headline": "Launch failures are blocking users from entering the product",
        "context": "First-open reliability issues risk immediate uninstalls before engagement begins.",
    },
    "Too many ads": {
        "headline": "Ad density is disrupting the core user experience",
        "context": "Volume and placement of ads are creating friction across primary use flows.",
    },
    "Login / OTP issues": {
        "headline": "Authentication friction is preventing users from accessing the product",
        "context": "OTP failures and login errors are creating a broken first experience.",
    },
    "Slow / Loading issues": {
        "headline": "Performance issues are eroding user patience and retention",
        "context": "Slow load times and lag are causing abandonment before users reach value.",
    },
    "Subscription / Payment": {
        "headline": "Payment friction is creating distrust around monetization",
        "context": "Unexpected charges and failed transactions are triggering complaints and refund requests.",
    },
    "Fake / Scam concerns": {
        "headline": "Trust and authenticity concerns are surfacing",
        "context": "Users are questioning the app's legitimacy — a reputational risk worth acting on quickly.",
    },
    "Bugs / Errors": {
        "headline": "Recurring bugs are undermining product reliability",
        "context": "Error messages and broken functionality are reducing user confidence in the product.",
    },
    "Poor quality / Useless": {
        "headline": "Core value proposition is not landing for a segment of users",
        "context": "Users feel the app doesn't deliver on its promise — investigate the expectation gap.",
    },
    "Customer support": {
        "headline": "Support experience is amplifying frustration after issues occur",
        "context": "Unresponsive support turns recoverable situations into negative public reviews.",
    },
    "Data / Privacy concerns": {
        "headline": "Data handling practices are raising user trust concerns",
        "context": "Privacy and security mentions signal reputational risk if left unaddressed.",
    },
    "Update broke the app": {
        "headline": "A recent update introduced regression issues",
        "context": "User experience deteriorated following an update — hotfix or rollback should be considered.",
    },
}

INVESTIGATION_HINTS: dict[str, str] = {
    "App crashing": "Check latest release notes, crash logs, and Play Console ANR reports",
    "App not opening": "Test launch on common devices; review Play Console ANR/crash data",
    "Too many ads": "Review ad frequency caps and placement; evaluate subscription offering",
    "Login / OTP issues": "Audit OTP delivery partner; test login flows on varied networks",
    "Slow / Loading issues": "Profile startup time; check server response times and CDN health",
    "Subscription / Payment": "Review payment gateway logs; check refund policy visibility",
    "Fake / Scam concerns": "Audit app description accuracy; review trust signals and listing",
    "Bugs / Errors": "Triage open bug reports; prioritize user-reported error strings",
    "Poor quality / Useless": "Run user interviews to understand expectation gaps",
    "Customer support": "Review ticket backlog; set up auto-acknowledge for open requests",
    "Data / Privacy concerns": "Audit data handling; improve privacy disclosure in onboarding",
    "Update broke the app": "Prioritize hotfix; consider rollback if adoption window allows",
}


def praised_strengths(df: pd.DataFrame, min_count: int = 3, top_n: int = 5) -> list[dict]:
    """Themes consistently mentioned in 4-5 star reviews."""
    high = df[df["rating"].isin([4, 5])]
    if high.empty:
        return []
    texts_lower = high["review_text"].str.lower().fillna("")
    total = len(high)
    results = []
    for theme, keywords in PRAISE_GROUPS.items():
        mask = texts_lower.apply(lambda t: any(kw in t for kw in keywords))
        count = int(mask.sum())
        if count < min_count:
            continue
        subset = high[mask]
        if "thumbs_up" in subset.columns:
            subset = subset.nlargest(2, "thumbs_up")
        else:
            subset = subset.head(2)
        samples = [
            (s[:110] + "…" if len(s) > 110 else s)
            for s in subset["review_text"].tolist()
        ]
        results.append({
            "theme": theme,
            "count": count,
            "pct_of_positive": round(100 * count / max(total, 1), 1),
            "samples": samples,
        })
    results.sort(key=lambda x: -x["count"])
    return results[:top_n]


def product_opportunities(df: pd.DataFrame, min_count: int = 2, top_n: int = 6) -> list[dict]:
    """
    Feature requests from happy users (4-5 star reviews containing request language).
    Roadmap intelligence: what do users who already like the app still wish for?
    """
    high = df[df["rating"].isin([4, 5])]
    if high.empty:
        return []
    texts_lower = high["review_text"].str.lower().fillna("")
    req_mask = texts_lower.apply(lambda t: any(s in t for s in _REQUEST_SIGNALS))
    request_df = high[req_mask]
    if request_df.empty:
        return []
    req_texts = request_df["review_text"].str.lower().fillna("")
    results = []
    for theme, keywords in _FEATURE_THEMES.items():
        mask = req_texts.apply(lambda t: any(kw in t for kw in keywords))
        count = int(mask.sum())
        if count < min_count:
            continue
        subset = request_df[mask]
        if "thumbs_up" in subset.columns:
            subset = subset.nlargest(2, "thumbs_up")
        else:
            subset = subset.head(2)
        samples = [
            (s[:110] + "…" if len(s) > 110 else s)
            for s in subset["review_text"].tolist()
        ]
        results.append({"request": theme, "count": count, "samples": samples})
    results.sort(key=lambda x: -x["count"])
    return results[:top_n]


def mixed_sentiment_areas(df: pd.DataFrame, min_each: int = 3) -> list[dict]:
    """
    Complaint groups present substantially in both 1-2★ AND 4-5★ reviews.
    Polarising areas where some users struggle, others are fine — worth PM investigation.
    """
    if df.empty or "complaint_categories" not in df.columns:
        return []
    low = df[df["rating"].isin([1, 2])]
    high = df[df["rating"].isin([4, 5])]
    if low.empty or high.empty:
        return []
    results = []
    for name in COMPLAINT_GROUPS:
        low_mask = low["complaint_categories"].apply(lambda c: name in c)
        high_mask = high["complaint_categories"].apply(lambda c: name in c)
        lc = int(low_mask.sum())
        hc = int(high_mask.sum())
        if lc < min_each or hc < min_each:
            continue
        if round(100 * lc / max(len(low), 1), 1) < 2:
            continue
        neg_s = low[low_mask].nsmallest(1, "rating")["review_text"].tolist()
        pos_s = high[high_mask].nlargest(1, "rating")["review_text"].tolist()
        neg_s = [(s[:100] + "…" if len(s) > 100 else s) for s in neg_s]
        pos_s = [(s[:100] + "…" if len(s) > 100 else s) for s in pos_s]
        results.append({
            "area": name,
            "low_count": lc,
            "high_count": hc,
            "neg_sample": neg_s,
            "pos_sample": pos_s,
        })
    results.sort(key=lambda x: -(x["low_count"] + x["high_count"]))
    return results[:4]
