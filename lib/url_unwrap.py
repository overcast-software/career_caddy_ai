"""Unwrap ad-tracker / redirect URLs by pulling the target out of query params.

Pure string/parse logic — no network. Safe to run before handing a URL to the
browser, which is useful when the tracker host is blocked by a local DNS
sinkhole (Pi-hole) and would otherwise fail to resolve.
"""

from __future__ import annotations

import base64
import binascii
from urllib.parse import parse_qs, unquote, urlparse

# Query-string keys that commonly carry the real destination URL.
# Ordered by rough specificity — longer / less ambiguous keys first.
_TARGET_PARAM_KEYS = (
    "redirect_url",
    "redirecturl",
    "destination",
    "target",
    "redirect",
    "url",
    "u",
    "r",
    "q",
)

# Hosts we know are trackers; used to log/flag but not required to unwrap.
KNOWN_TRACKER_HOSTS = frozenset({
    "click.appcast.io",
    "click.linkedin.com",
    "click.mail.linkedin.com",
    "click.indeed.com",
    "www.google.com",  # /url?q=...
    "l.facebook.com",
    "lnkd.in",
})


def _try_b64_decode(value: str) -> str | None:
    """Return decoded URL if value is a base64-encoded http(s) link, else None."""
    s = value.strip()
    # urlsafe base64 pad tolerance
    padded = s + "=" * (-len(s) % 4)
    try:
        decoded = base64.urlsafe_b64decode(padded).decode("utf-8", errors="strict")
    except (binascii.Error, UnicodeDecodeError, ValueError):
        return None
    if decoded.startswith(("http://", "https://")):
        return decoded
    return None


def unwrap_url(url: str, max_depth: int = 5) -> str:
    """Return the innermost http(s) URL found in tracker query params.

    Applies recursively (e.g. tracker → tracker → real URL) up to max_depth.
    Returns the original URL unchanged if nothing to unwrap.
    """
    if not url:
        return url

    current = url
    for _ in range(max_depth):
        parsed = urlparse(current)
        if not parsed.query:
            return current

        params = parse_qs(parsed.query, keep_blank_values=False)
        # Normalize keys to lowercase for lookup
        lower = {k.lower(): v for k, v in params.items()}

        candidate: str | None = None
        for key in _TARGET_PARAM_KEYS:
            values = lower.get(key)
            if not values:
                continue
            raw = unquote(values[0])
            if raw.startswith(("http://", "https://")):
                candidate = raw
                break
            decoded = _try_b64_decode(raw)
            if decoded:
                candidate = decoded
                break

        if not candidate or candidate == current:
            return current
        current = candidate

    return current


def is_known_tracker(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host in KNOWN_TRACKER_HOSTS
