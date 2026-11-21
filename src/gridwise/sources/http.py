"""One HTTP client for all sources: system certificate store, retries, a plain user agent."""

from __future__ import annotations

import time
from typing import Any

import httpx

USER_AGENT = "Mozilla/5.0 (compatible; research-client)"


def _use_system_certs() -> None:
    # Some machines have an antivirus that re-signs HTTPS; the system store knows its certificate.
    try:
        import truststore

        truststore.inject_into_ssl()
    except ImportError:
        pass


def make_client(timeout: float = 60.0) -> httpx.Client:
    _use_system_certs()
    return httpx.Client(timeout=timeout, headers={"User-Agent": USER_AGENT}, follow_redirects=True)


class FetchError(RuntimeError):
    pass


def get_json(client: httpx.Client, url: str, params: dict[str, Any] | None = None, tries: int = 5) -> Any:
    """GET a JSON document, retrying on network errors, 429 and 5xx with exponential back-off."""
    last = ""
    for attempt in range(tries):
        try:
            r = client.get(url, params=params)
            if r.status_code == 200:
                return r.json()
            last = f"HTTP {r.status_code}: {r.text[:200]}"
            if r.status_code not in (429, 500, 502, 503, 504):
                break
        except (httpx.TransportError, ValueError) as e:
            last = repr(e)
        time.sleep(min(60.0, 2.0 * 2**attempt))
    raise FetchError(f"{url} failed after {tries} tries: {last}")
