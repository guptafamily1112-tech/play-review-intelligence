from __future__ import annotations

import streamlit as st
import pandas as pd

from utils import extract_package_id
from scraper import fetch_reviews, fetch_app_info
from sentiment import score_sentiment
from complaints import (
    detect_complaints,
    complaint_summary,
    rising_complaints,
    key_signals,
    praised_strengths,
    product_opportunities,
    INVESTIGATION_HINTS,
    SIGNAL_FRAMING,
    PRAISE_GROUPS,
    _FEATURE_THEMES,
    _REQUEST_SIGNALS,
)

st.set_page_config(
    page_title="Play Review Intelligence",
    page_icon="🔍",
    layout="wide",
)

st.markdown("""
<style>
/* ── Base ── */
[data-testid="stAppViewContainer"] > .main { background: #f8fafc; }
[data-testid="block-container"] { max-width: 1060px; padding-top: 2.25rem; padding-bottom: 4rem; }
footer, #MainMenu { display: none !important; }
hr { border-color: #e2e8f0 !important; margin: 1.5rem 0 !important; }

/* ── App header card ── */
.app-card {
    display: flex; gap: 1.125rem; align-items: flex-start;
    background: #fff; border: 1px solid #e2e8f0; border-radius: 14px;
    padding: 1.125rem 1.375rem; margin: 0.625rem 0 0.875rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.app-icon  { width: 52px; height: 52px; border-radius: 10px; object-fit: cover; flex-shrink: 0; background: #e2e8f0; }
.app-body  { flex: 1; min-width: 0; }
.app-title { font-size: 1.15rem; font-weight: 700; color: #0f172a; margin: 0 0 2px; line-height: 1.3; }
.app-genre { font-size: 0.68rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.375rem; }
.app-desc  { font-size: 0.82rem; color: #475569; line-height: 1.5; margin-bottom: 0.5rem; }
.pills { display: flex; gap: 0.3rem; flex-wrap: wrap; align-items: center; }
.pill {
    display: inline-flex; align-items: center;
    padding: 2px 9px; border-radius: 999px;
    font-size: 0.72rem; font-weight: 500; border: 1px solid; white-space: nowrap;
}
.pill-slate { background: #f1f5f9; border-color: #e2e8f0; color: #475569; }
.pill-gold  { background: #fef9c3; border-color: #fde68a; color: #92400e; }
.pill-blue  { background: #eff6ff; border-color: #bfdbfe; color: #1d4ed8; }

/* ── Signal cards ── */
.sig-card {
    background: #fff; border: 1px solid #e2e8f0; border-radius: 11px;
    padding: 0.875rem 1.1rem; margin-bottom: 0.2rem;
    border-left: 4px solid #e2e8f0;
}
.sig-card.sig-critical {
    border-left-color: #ef4444;
    box-shadow: 0 1px 8px rgba(239,68,68,0.07);
}
.sig-card.sig-emerging { border-left-color: #f97316; }
.sig-card.sig-green    { border-left-color: #22c55e; }
.sig-card.sig-blue     { border-left-color: #3b82f6; }

.sig-headline { font-size: 0.94rem; font-weight: 700; color: #0f172a; margin-bottom: 0.15rem; line-height: 1.35; }
.sig-subtext  { font-size: 0.79rem; color: #64748b; margin-bottom: 0.3rem; line-height: 1.45; }
.sig-meta     { display: flex; gap: 0.6rem; flex-wrap: wrap; font-size: 0.72rem; color: #94a3b8; margin-bottom: 0.275rem; }
.sig-action   { font-size: 0.74rem; color: #6366f1; margin-top: 0.3rem; }

/* ── Quote snippets ── */
.sq {
    font-size: 0.78rem; color: #475569; border-left: 2px solid #e2e8f0;
    padding: 0.15rem 0.5rem; margin: 0.225rem 0; line-height: 1.45; font-style: italic;
}
.sq-pos { border-left-color: #86efac; }

/* ── Theme library cards ── */
.comp-card {
    background: #fff; border: 1px solid #e2e8f0; border-radius: 9px;
    padding: 0.7rem 0.95rem; margin-bottom: 0.2rem;
}
.comp-header { display: flex; align-items: center; gap: 0.4rem; margin-bottom: 0.2rem; flex-wrap: wrap; }
.comp-title  { font-size: 0.85rem; font-weight: 600; color: #0f172a; }
.comp-meta   { display: flex; gap: 0.7rem; flex-wrap: wrap; font-size: 0.72rem; color: #64748b; margin-bottom: 0.25rem; }
.badge { font-size: 0.67rem; font-weight: 600; padding: 2px 6px; border-radius: 4px; display: inline-flex; align-items: center; }
.badge-rising      { background: #fef3c7; color: #92400e; border: 1px solid #fde68a; }
.badge-stable      { background: #f1f5f9; color: #64748b;  border: 1px solid #e2e8f0; }
.badge-exploratory { background: #f8fafc; color: #94a3b8;  border: 1px solid #e2e8f0; font-style: italic; }

/* ── Pills (native Streamlit chip controls) ── */
[data-testid="stPillsContainer"] { gap: 6px !important; }
[data-testid="stPillsContainer"] button {
    font-size: 0.79rem !important;
    padding: 3px 12px !important;
    border-radius: 999px !important;
    font-weight: 500 !important;
}

/* ── Primary CTA ── */
[data-testid="baseButton-primary"] { background: #4f46e5 !important; border-color: #4f46e5 !important; color: #fff !important; }
[data-testid="baseButton-primary"]:hover  { background: #4338ca !important; border-color: #4338ca !important; }
[data-testid="baseButton-primary"]:active { background: #3730a3 !important; border-color: #3730a3 !important; }

/* ── Investigation CTA button ── */
[data-testid="baseButton-secondary"].inv-btn {
    font-size: 0.75rem !important; padding: 2px 10px !important;
    color: #6366f1 !important; border-color: #e0e7ff !important;
    background: #f5f3ff !important; border-radius: 6px !important;
}

/* ── Misc polish ── */
[data-testid="stAlert"]     { border-radius: 10px !important; }
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; border: 1px solid #e2e8f0; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
def _confidence_tier(count: int) -> str:
    if count >= 12: return "high"
    if count >= 6:  return "medium"
    return "low"


def _sig_card(
    headline: str,
    subtext: str,
    meta: list[str],
    snippets: list[tuple[str, str]],
    css_color: str,
    action: str = "",
    confidence: str = "high",
) -> str:
    conf_badge = ' <span class="badge badge-exploratory">Exploratory</span>' if confidence == "low" else ""
    meta_html  = "".join(f"<span>{m}</span>" for m in meta if m)
    snip_html  = "".join(
        f'<div class="sq {css}">{text}</div>' for text, css in snippets if text
    )
    act_html = f'<div class="sig-action">💡 {action}</div>' if action else ""
    return (
        f'<div class="sig-card sig-{css_color}">'
        f'<div class="sig-headline">{headline}{conf_badge}</div>'
        f'<div class="sig-subtext">{subtext}</div>'
        f'<div class="sig-meta">{meta_html}</div>'
        f"{snip_html}{act_html}</div>"
    )


def _comp_card(
    title: str, freq: str, rating: str, rising: bool,
    snippets: list[str], confidence: str = "high",
) -> str:
    badge = (
        '<span class="badge badge-rising">↑ Rising</span>'
        if rising else
        '<span class="badge badge-stable">→ Stable</span>'
    )
    conf_badge = ' <span class="badge badge-exploratory">Exploratory</span>' if confidence == "low" else ""
    snip_html  = "".join(
        f'<div class="sq">{(s[:110] + "…") if len(s) > 110 else s}</div>'
        for s in snippets[:2]
    )
    return (
        f'<div class="comp-card">'
        f'<div class="comp-header"><span class="comp-title">{title}</span>{badge}{conf_badge}</div>'
        f'<div class="comp-meta"><span>{freq}</span><span>{rating}</span></div>'
        f"{snip_html}</div>"
    )


@st.dialog("Review Investigation", width="large")
def _investigation_modal(title: str, reviews: pd.DataFrame) -> None:
    if reviews.empty:
        st.info("No related reviews found.")
        return

    st.markdown(f"**{title}**")

    sort_col, _ = st.columns([2, 3])
    with sort_col:
        sort_order = st.radio(
            "Sort",
            ["Latest first", "Oldest first"],
            horizontal=True,
            label_visibility="collapsed",
        )

    reviews = reviews.sort_values("date", ascending=(sort_order == "Oldest first"))

    cap = 100
    shown = reviews.head(cap)
    st.caption(
        f"{len(reviews):,} reviews"
        + (f" · showing first {cap}" if len(reviews) > cap else "")
    )
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    for _, row in shown.iterrows():
        rating  = int(row.get("rating", 3))
        stars   = "★" * rating + "☆" * (5 - rating)
        date_val = row.get("date")
        date_str = date_val.strftime("%d %b %Y") if pd.notna(date_val) else ""
        text     = str(row.get("review_text", ""))
        st.markdown(
            f'<div style="border:1px solid #e2e8f0;border-radius:8px;padding:0.875rem 1rem;'
            f'margin-bottom:0.5rem;background:#fff;">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'margin-bottom:0.375rem;">'
            f'<span style="color:#f59e0b;letter-spacing:1px;font-size:0.9rem;">{stars}</span>'
            f'<span style="font-size:0.7rem;color:#94a3b8;">{date_str}</span>'
            f'</div>'
            f'<div style="font-size:0.86rem;color:#1e293b;line-height:1.6;">{text}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ── Review scope ───────────────────────────────────────────────────────────────
_SCOPE: dict[str, dict] = {
    "Last 7 days":    {"days_back": 7,   "target": 5000},
    "Last 30 days":   {"days_back": 30,  "target": 5000},
    "Last 90 days":   {"days_back": 90,  "target": 5000},
    "Last 12 months": {"days_back": 365, "target": 5000},
    "All recent":     {"days_back": None, "target": 1000},
}

# ── Input ──────────────────────────────────────────────────────────────────────
st.markdown("## Play Review Intelligence")
st.caption("Surface product insights from Google Play reviews — built for PMs.")

url = st.text_input(
    "URL",
    placeholder="https://play.google.com/store/apps/details?id=com.example.app",
    label_visibility="collapsed",
)

scope_col, fetch_col = st.columns([4, 1])
with scope_col:
    _scope_picked = st.pills("Review scope", list(_SCOPE.keys()), default="Last 90 days")
    scope_label = _scope_picked or "Last 90 days"
fetch_col.write("")
fetch_clicked = fetch_col.button("Analyze App →", type="primary", use_container_width=True)

# ── Fetch ──────────────────────────────────────────────────────────────────────
if fetch_clicked:
    if not url.strip():
        st.warning("Please enter a Google Play Store URL.")
    else:
        package_id = extract_package_id(url)
        if not package_id:
            st.error("Could not extract a package ID. Make sure the URL contains `?id=<package.id>`.")
        else:
            scope = _SCOPE[scope_label]
            progress = st.progress(0, text="Fetching app info…")
            try:
                app_info = fetch_app_info(package_id)
                progress.progress(0.05, text=f"Fetching reviews for {app_info['title']}…")

                def on_progress(fetched: int, _total: int) -> None:
                    frac = fetched / max(scope["target"], 1) if scope["days_back"] is None else min(fetched / 300, 0.95)
                    progress.progress(0.05 + min(frac, 1.0) * 0.80, text=f"Fetched {fetched:,} reviews…")

                df = fetch_reviews(
                    package_id,
                    target_count=scope["target"],
                    days_back=scope["days_back"],
                    progress_callback=on_progress,
                )
                progress.progress(0.88, text="Scoring sentiment…")
                df = score_sentiment(df)
                progress.progress(0.94, text="Detecting patterns…")
                df = detect_complaints(df)
                progress.progress(1.0, text=f"Done — {len(df):,} reviews ready.")
                st.session_state.update({
                    "reviews_df": df,
                    "app_id": package_id,
                    "app_info": app_info,
                    "scope_label": scope_label,
                })
            except (ValueError, RuntimeError) as exc:
                progress.empty()
                st.error(str(exc))

# ── Guard ──────────────────────────────────────────────────────────────────────
if "reviews_df" not in st.session_state:
    st.stop()

df: pd.DataFrame = st.session_state["reviews_df"]
app_id: str      = st.session_state["app_id"]
app_info: dict   = st.session_state.get("app_info", {})
_scope_display   = st.session_state.get("scope_label", "")

# ═══════════════════════════════════════════════════════════════════════════════
# APP HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.divider()

_icon    = app_info.get("icon") or ""
_title   = app_info.get("title") or app_id
_genre   = app_info.get("genre") or ""
_summary = (app_info.get("summary") or "")[:200]
_rating  = app_info.get("store_rating")
_ratings = app_info.get("total_ratings")

# Trustworthy date display — day-level when span is short
_span_days = (df["date"].max() - df["date"].min()).days
if _span_days == 0:
    _date_range = df["date"].max().strftime("%d %b %Y")
elif _span_days <= 60:
    _date_range = f"{df['date'].min().strftime('%d %b')}–{df['date'].max().strftime('%d %b %Y')}"
else:
    _date_range = f"{df['date'].min().strftime('%b %Y')}–{df['date'].max().strftime('%b %Y')}"

_icon_tag     = f'<img class="app-icon" src="{_icon}" />' if _icon else '<div class="app-icon"></div>'
_rating_pill  = f'<span class="pill pill-gold">★ {_rating:.1f}</span>' if _rating else ""
_ratings_pill = f'<span class="pill pill-slate">{_ratings:,} ratings</span>' if _ratings else ""
_scope_str    = f"{_scope_display} · " if _scope_display else ""
_sample_pill  = f'<span class="pill pill-blue">{len(df):,} reviews · {_scope_str}{_date_range}</span>'
_desc_block   = f'<p class="app-desc">{_summary}</p>' if _summary else ""

st.markdown(f"""
<div class="app-card">
  {_icon_tag}
  <div class="app-body">
    <div class="app-title">{_title}</div>
    <div class="app-genre">{_genre}</div>
    {_desc_block}
    <div class="pills">{_rating_pill}{_ratings_pill}{_sample_pill}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS CONTEXT — filter chips
# Controls: Theme Library + Review Explorer below
# Key Product Signals always reflect the full dataset
# ═══════════════════════════════════════════════════════════════════════════════
_PRIMARY = ["All Reviews", "Negative", "Mixed", "Positive", "Requests"]
primary_chip = st.pills("Review Scope", _PRIMARY, default="All Reviews") or "All Reviews"

_CAT_CHIP_MAP: dict[str, list[str]] = {
    "Slow":    ["Slow / Loading issues"],
    "Crash":   ["App crashing", "App not opening"],
    "Login":   ["Login / OTP issues"],
    "Payment": ["Subscription / Payment"],
    "Ads":     ["Too many ads"],
    "Support": ["Customer support"],
}
selected_cats = st.pills("Popular Themes", list(_CAT_CHIP_MAP.keys()), selection_mode="multi") or []


def _apply_primary(frame: pd.DataFrame, chip: str) -> pd.DataFrame:
    if chip == "Negative":  return frame[frame["rating"].isin([1, 2])]
    if chip == "Mixed":     return frame[frame["rating"] == 3]
    if chip == "Positive":  return frame[frame["rating"].isin([4, 5])]
    if chip == "Requests":
        high  = frame[frame["rating"].isin([4, 5])]
        texts = high["review_text"].str.lower().fillna("")
        return high[texts.apply(lambda t: any(s in t for s in _REQUEST_SIGNALS))]
    return frame


filtered = _apply_primary(df, primary_chip)
if selected_cats:
    cat_names = [n for c in selected_cats for n in _CAT_CHIP_MAP[c]]
    filtered = filtered[
        filtered["complaint_categories"].apply(lambda cats: any(c in cats for c in cat_names))
    ]

# ═══════════════════════════════════════════════════════════════════════════════
# KEY PRODUCT SIGNALS  — always from full dataset, not affected by filter chips
# ═══════════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown("## Key Product Signals")

# Stricter thresholds — fewer, higher-confidence signals only
_signals       = key_signals(df, min_recent_mentions=8, min_change_pp=5.0)
_critical      = [s for s in _signals if s["is_critical"]]
_emerging      = [s for s in _signals if not s["is_critical"]]
_opportunities = product_opportunities(df, min_count=4, top_n=5)
_strengths     = praised_strengths(df, min_count=5, top_n=4)

_rn        = max(50, int(len(df) * 0.2))
_recent_df = df.nlargest(_rn, "date")


def _get_snippets(src: pd.DataFrame, name: str, n: int = 2) -> list[tuple[str, str]]:
    mask  = src["complaint_categories"].apply(lambda c: name in c)
    texts = src[mask].nsmallest(n, "rating")["review_text"].tolist()
    return [((t[:130] + "…") if len(t) > 130 else t, "") for t in texts]


# ── 🔴 Critical Issues ──
if _critical:
    st.markdown("#### 🔴 Critical Issues")
    for i, s in enumerate(_critical):
        framing  = SIGNAL_FRAMING.get(s["complaint"], {})
        headline = framing.get("headline", s["complaint"])
        subtext  = framing.get("context", "")
        days     = f"Spiked {s['days_ago']}d ago" if s.get("days_ago") else "Recently spiking"
        star     = "Mostly 1-star" if s["avg_rating"] <= 1.5 else f"Avg {s['avg_rating']}★"
        hint     = INVESTIGATION_HINTS.get(s["complaint"], "Review recent logs and user-reported errors")
        st.markdown(
            _sig_card(
                headline, subtext,
                [f"📌 {s['recent_count']} recent reviews", f"⭐ {star}", f"⏱ {days}", f"+{s['change']}pp vs baseline"],
                _get_snippets(_recent_df, s["complaint"]),
                "critical", action=hint,
                confidence=_confidence_tier(s["recent_count"]),
            ),
            unsafe_allow_html=True,
        )
        _related = df[df["complaint_categories"].apply(lambda c: s["complaint"] in c)]
        if not _related.empty and st.button(
            f"View {len(_related):,} related reviews →",
            key=f"inv_crit_{i}",
        ):
            _investigation_modal(headline, _related)

# ── 🟠 Emerging Issues ──
if _emerging:
    st.markdown("#### 🟠 Emerging Issues")
    for i, s in enumerate(_emerging):
        framing  = SIGNAL_FRAMING.get(s["complaint"], {})
        headline = framing.get("headline", s["complaint"])
        subtext  = framing.get("context", "")
        st.markdown(
            _sig_card(
                headline, subtext,
                [f"📌 {s['recent_count']} recent mentions", f"⭐ Avg {s['avg_rating']}★", f"+{s['change']}pp vs baseline"],
                _get_snippets(_recent_df, s["complaint"]),
                "emerging",
                confidence=_confidence_tier(s["recent_count"]),
            ),
            unsafe_allow_html=True,
        )
        _related = df[df["complaint_categories"].apply(lambda c: s["complaint"] in c)]
        if not _related.empty and st.button(
            f"View {len(_related):,} related reviews →",
            key=f"inv_emrg_{i}",
        ):
            _investigation_modal(headline, _related)

# ── 🟢 Opportunities & 🔵 Strengths ──
if _opportunities or _strengths:
    _opp_col, _str_col = st.columns(2)

    if _opportunities:
        with _opp_col:
            st.markdown("#### 🟢 Product Opportunities")
            st.caption("Feature requests from satisfied users:")
            for i, opp in enumerate(_opportunities):
                sq = [(opp["samples"][0], "sq-pos")] if opp["samples"] else []
                st.markdown(
                    _sig_card(
                        opp["request"],
                        "Requested by users who already enjoy the product — high-signal roadmap input.",
                        [f"📝 {opp['count']} users requesting"],
                        sq, "green",
                        confidence=_confidence_tier(opp["count"]),
                    ),
                    unsafe_allow_html=True,
                )
                _kws = _FEATURE_THEMES.get(opp["request"], [])
                if _kws:
                    _feat_texts = df["review_text"].str.lower().fillna("")
                    _related = df[_feat_texts.apply(lambda t: any(kw in t for kw in _kws))]
                    if not _related.empty and st.button(
                        f"View {len(_related):,} related reviews →",
                        key=f"inv_opp_{i}",
                    ):
                        _investigation_modal(opp["request"], _related)

    if _strengths:
        with _str_col:
            st.markdown("#### 🔵 Frequently Praised Strengths")
            st.caption("What users consistently love:")
            for i, s in enumerate(_strengths):
                sq = [(s["samples"][0], "sq-pos")] if s["samples"] else []
                st.markdown(
                    _sig_card(
                        s["theme"],
                        f"Mentioned in {s['pct_of_positive']}% of positive reviews — protect this in every release.",
                        [f"💬 {s['count']} positive reviews"],
                        sq, "blue",
                        confidence=_confidence_tier(s["count"]),
                    ),
                    unsafe_allow_html=True,
                )
                _kws = PRAISE_GROUPS.get(s["theme"], [])
                if _kws:
                    _praise_texts = df["review_text"].str.lower().fillna("")
                    _related = df[_praise_texts.apply(lambda t: any(kw in t for kw in _kws))]
                    if not _related.empty and st.button(
                        f"View {len(_related):,} related reviews →",
                        key=f"inv_str_{i}",
                    ):
                        _investigation_modal(s["theme"], _related)

if not any([_critical, _emerging, _opportunities, _strengths]):
    st.info(
        "Not enough signal in this review sample. Try a wider time scope.",
        icon="ℹ️",
    )

# ═══════════════════════════════════════════════════════════════════════════════
# THEME LIBRARY — structured taxonomy, filtered by chips above
# ═══════════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown("### Theme Library")
st.caption(f"Complaint themes in the current scope · **{len(filtered):,}** of {len(df):,} reviews")

_comp_df    = complaint_summary(filtered)
_rising_df  = rising_complaints(df)
_rising_set = set(_rising_df["Complaint"].tolist()) if not _rising_df.empty else set()

if _comp_df.empty:
    st.info("No complaint themes detected in this selection.")
else:
    _total_f = len(filtered)

    def _freq_label(count: int) -> str:
        pct = 100 * count / max(_total_f, 1)
        if pct >= 20: return f"Very frequently mentioned · {count} reviews"
        if pct >= 10: return f"Frequently mentioned · {count} reviews"
        if pct >= 5:  return f"Commonly mentioned · {count} reviews"
        return f"Mentioned in {count} reviews"

    def _rating_label(avg: float) -> str:
        if avg <= 1.5: return "Almost all 1-star reviews"
        if avg <= 2.0: return "Mostly 1-2 star reviews"
        if avg <= 2.5: return "Mostly low-rated reviews"
        if avg <= 3.5: return "Mixed star ratings"
        return "Appears even in positive reviews"

    for idx, row in _comp_df.iterrows():
        cname = row["Complaint"]
        cmask = filtered["complaint_categories"].apply(lambda c: cname in c)
        cavg  = float(filtered[cmask]["rating"].mean()) if cmask.any() else 3.0
        st.markdown(
            _comp_card(
                cname,
                _freq_label(int(row["Mentions"])),
                _rating_label(cavg),
                cname in _rising_set,
                row["_samples"],
                confidence=_confidence_tier(int(row["Mentions"])),
            ),
            unsafe_allow_html=True,
        )
        _related = filtered[cmask]
        if not _related.empty and st.button(
            f"View {len(_related):,} related reviews →",
            key=f"inv_comp_{idx}",
        ):
            _investigation_modal(cname, _related)

# ═══════════════════════════════════════════════════════════════════════════════
# REVIEW EXPLORER — raw evidence, filtered by chips above
# ═══════════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("Review Explorer")

search_query = st.text_input("Search", placeholder="Search by keyword…", label_visibility="collapsed")

table_df = filtered.copy()
if search_query.strip():
    table_df = table_df[table_df["review_text"].str.contains(search_query.strip(), case=False, na=False)]

st.caption(f"{len(table_df):,} reviews shown.")

display_df = table_df[["date", "rating", "complaint_categories", "review_text"]].copy()
display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")
display_df["complaint_categories"] = display_df["complaint_categories"].apply(lambda c: ", ".join(c) if c else "—")
display_df.columns = ["Date", "Rating", "Theme", "Review"]

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Rating": st.column_config.NumberColumn(format="%d ⭐"),
        "Theme":  st.column_config.TextColumn(width="medium"),
        "Review": st.column_config.TextColumn(width="large"),
    },
)

# ── CSV export ─────────────────────────────────────────────────────────────────
dl1, dl2 = st.columns(2)

_all_export = df[["date", "rating", "sentiment_label", "complaint_categories", "review_text"]].copy()
_all_export["date"] = _all_export["date"].dt.strftime("%Y-%m-%d")
_all_export["complaint_categories"] = _all_export["complaint_categories"].apply(lambda c: ", ".join(c) if c else "")

dl1.download_button(
    label=f"Download all {len(df):,} reviews (CSV)",
    data=_all_export.to_csv(index=False).encode("utf-8"),
    file_name=f"{app_id}_all_reviews.csv",
    mime="text/csv",
)
dl2.download_button(
    label=f"Download filtered {len(table_df):,} reviews (CSV)",
    data=display_df.to_csv(index=False).encode("utf-8"),
    file_name=f"{app_id}_filtered_reviews.csv",
    mime="text/csv",
)
