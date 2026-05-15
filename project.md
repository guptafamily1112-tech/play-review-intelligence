# Google Play Review Insight Explorer

## Project Summary

A single-page web tool where a product manager pastes a public Google Play Store URL and receives structured, AI-assisted insights from at least 500 real user reviews — without any login, payment, or complex infrastructure.

---

## Problem Statement

Product managers need to understand what users love and hate about an app, but reading hundreds of Play Store reviews manually is time-consuming and imprecise. This tool automates that work: fetch reviews, score sentiment, and surface the most actionable patterns using NLP and an LLM.

---

## Target User

**Product Managers** who want fast, self-serve access to voice-of-customer data from Google Play Store reviews — without needing to involve a data analyst or wait for a report.

---

## MVP Flow

1. User pastes a Google Play Store URL (e.g. `https://play.google.com/store/apps/details?id=com.digilocker.android&hl=en_IN`)
2. System extracts the app package ID from the URL
3. System fetches ≥500 public reviews using `google-play-scraper`
4. A progress bar is shown during the fetch (can take 30–60 seconds)
5. System runs NLP-based sentiment scoring on each review
6. System calls an LLM to generate narrative summaries of top complaints and loved features
7. Insights and review data are displayed in the UI

One URL at a time. No concurrent sessions required for MVP.

---

## MVP Features

| Feature | Description |
|---|---|
| URL input | Single text field accepting a Google Play Store app URL |
| Review fetch | Pull ≥500 public reviews via `google-play-scraper` |
| Progress feedback | Progress bar visible during the fetch and processing |
| Total reviews count | Display how many reviews were fetched |
| Date range filter | Filter displayed reviews and charts by date |
| Star rating filter | Filter by 1–5 star rating |
| Rating distribution | Bar chart showing count per star rating |
| Sentiment trend graph | Line/area chart of sentiment score over time |
| Top complaints | LLM-generated summary of recurring negative themes |
| Most loved features | LLM-generated summary of recurring positive themes |
| Review table | Tabular view with: date, rating, sentiment label, review text |

---

## Tech Stack

| Layer | Choice | Reason |
|---|---|---|
| App framework | **Streamlit** | All-Python, fast to build, ideal for PM-facing data tools |
| Review scraping | **google-play-scraper** (Python) | Unofficial but widely used, no auth required |
| Data handling | **pandas** | Filtering, sorting, aggregation |
| Sentiment scoring | **VADER** (`vaderSentiment`) or **TextBlob** | Fast, lexicon-based, no model download needed |
| Insight summaries | **Claude API** (Anthropic) or OpenAI | LLM narrative for top complaints and loved features |
| Charting | **Plotly** (via `st.plotly_chart`) | Interactive charts inside Streamlit |
| Deployment | **Streamlit Community Cloud** | Free public hosting for Streamlit apps, easy deploy from GitHub |

---

## Insight Generation Strategy (Hybrid)

- **Sentiment per review**: Scored locally using VADER — fast, free, no API call per review
- **Top complaints & loved features**: A single LLM call receives the top N negative/positive review texts and returns a structured summary. This keeps API cost low (one call per analysis run, not per review).

### LLM prompt design (sketch)
```
You are a product analyst. Below are user reviews from a mobile app.
Summarize the top 3-5 recurring complaints and top 3-5 most loved features.
Be specific. Use the users' own words where possible.

Reviews:
{review_sample}
```

---

## Project Structure (Planned)

```
google-play-insight-explorer/
├── app.py                  # Streamlit app entry point
├── scraper.py              # google-play-scraper wrapper
├── sentiment.py            # VADER sentiment scoring
├── insights.py             # LLM call for complaint/feature summaries
├── utils.py                # URL parsing, data cleaning helpers
├── requirements.txt
├── .env.example            # API key placeholder
└── README.md
```

---

## Data Model (per review row)

| Field | Type | Source |
|---|---|---|
| `review_id` | string | scraper |
| `date` | datetime | scraper |
| `rating` | int (1–5) | scraper |
| `review_text` | string | scraper |
| `sentiment_score` | float (-1 to 1) | VADER |
| `sentiment_label` | string (Positive/Neutral/Negative) | derived |

---

## Constraints & Non-Goals (MVP)

- No user authentication
- No payment or subscription
- No persistent database — all data lives in Streamlit session state
- No multi-URL comparison
- No scheduled/background jobs
- No user accounts or saved reports

---

## Key Dependencies

```
streamlit
google-play-scraper
pandas
vaderSentiment
anthropic          # or openai
plotly
python-dotenv
```

---

## Open Questions / Future Scope

- **Rate limiting**: `google-play-scraper` may hit rate limits for very high review counts (1000+). Retry logic or batching may be needed.
- **LLM cost control**: Cap the number of reviews sent to the LLM (e.g. top 100 by recency or rating) to keep cost predictable.
- **Caching**: For the public deployment, consider caching results per app ID + date to reduce redundant scrapes.
- **Multi-language reviews**: VADER is English-only. Apps with multilingual reviews will need a translation step or a multilingual model.
- **Future features**: Export to CSV, compare two apps, keyword search within reviews, version-specific filtering.

---

## Success Criteria (MVP)

- A PM can paste a Play Store URL and get a full insight report in under 90 seconds
- At least 500 reviews are fetched and displayed
- Sentiment trend, rating distribution, top complaints, and loved features are all populated
- The tool is publicly accessible via a Streamlit Cloud URL
