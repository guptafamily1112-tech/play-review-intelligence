# Play Review Intelligence

A Streamlit-based Google Play review intelligence tool built to help Product Managers quickly understand user complaints, feature requests, sentiment patterns, and product opportunities from app reviews.

## What it does

- Scrapes Google Play Store reviews
- Analyzes complaints, opportunities, strengths, and sentiment
- Groups reviews into product-relevant themes
- Helps PMs investigate recurring issues from real user feedback
- Supports review recency scopes like last 7 days, 30 days, 90 days, 12 months, and all recent reviews

## Why I built this

Product teams often have thousands of app reviews but limited time to manually read and classify them. This tool is an MVP attempt to convert raw reviews into actionable product intelligence.

## Tech stack

- Python
- Streamlit
- Google Play review scraping
- Gemini / LLM-assisted analysis
- Pandas

## Current status

MVP stage. The tool works, but the investigation flow, review filtering, and insight quality are still being improved.

## Future improvements

- Better review investigation modal
- Sort reviews by latest/oldest
- Show full related reviews for every theme
- Improve filtering logic
- Add stronger AI summarization
