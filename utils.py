from __future__ import annotations

from urllib.parse import urlparse, parse_qs

_HL_TO_COUNTRY: dict[str, str] = {
    "en_in": "in",
    "hi_in": "in",
    "en_us": "us",
    "en_gb": "gb",
    "en_au": "au",
    "en_ca": "ca",
}


def extract_country_hint(url: str) -> str | None:
    """Infer a Play Store country code from the hl= URL parameter, if present."""
    params = parse_qs(urlparse(url.strip()).query)
    hl = params.get("hl", [None])[0]
    if hl:
        return _HL_TO_COUNTRY.get(hl.lower())
    return None


def extract_package_id(url: str) -> str | None:
    """
    Extract the app package ID from a Google Play Store URL.
    Supports both full URLs and bare package IDs.
    e.g. https://play.google.com/store/apps/details?id=com.example.app -> com.example.app
    """
    url = url.strip()
    parsed = urlparse(url)

    # Already a bare package ID (no scheme, no path segments like /store/apps/)
    if not parsed.scheme and not parsed.netloc:
        if "." in url and "/" not in url:
            return url

    params = parse_qs(parsed.query)
    package_id = params.get("id", [None])[0]
    return package_id
