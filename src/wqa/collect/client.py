"""Wikimedia API client (SPEC §4.3 hard rules).
One request stream (sequential), UA from configs/collection.yaml, maxlag=5,
backoff 2s->60s+jitter on 429/503/maxlag. In: cfg dict. Out: JSON dict per call.
Usage: c = ApiClient(config.load("collection")); c.query(list=..., ...)."""
from __future__ import annotations

import random
import time
from typing import Any
from urllib.parse import quote

import requests
import structlog

from wqa import config

log = structlog.get_logger()

_POLITE_DELAY_S = 0.05
_MAX_ATTEMPTS = 20  # 429 bursts on full runs lasted >8 backoffs (2026-08-25 crash)


class ApiClient:
    """Thin wrapper around one requests.Session enforcing UA + maxlag/429/503 backoff."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.ua = config.user_agent(cfg)
        self.session = requests.Session()
        self.session.headers["User-Agent"] = self.ua
        self.api = cfg["api"]
        self.pageviews_api = cfg["pageviews_api"]
        self.maxlag = cfg["maxlag"]
        self.backoff_start = float(cfg["backoff"]["start_s"])
        self.backoff_max = float(cfg["backoff"]["max_s"])
        self.jitter = bool(cfg["backoff"]["jitter"])

    def query(self, **params: Any) -> dict[str, Any]:
        """GET action=query&format=json&maxlag=<n> with the given params."""
        full = {"action": "query", "format": "json", "maxlag": self.maxlag, **params}
        return self._get(self.api, full)

    def pageviews(self, title_safe: str, start: str, end: str) -> dict[str, Any]:
        """GET per-article pageviews REST endpoint; 404 (no data) returns {}."""
        safe = quote(title_safe, safe="")
        url = f"{self.pageviews_api}/all-access/user/{safe}/daily/{start}/{end}"
        return self._get(url, None, quiet_404=True)

    def _get(self, url: str, params: dict[str, Any] | None, quiet_404: bool = False) -> dict[str, Any]:
        wait = self.backoff_start
        for attempt in range(_MAX_ATTEMPTS):
            try:
                resp = self.session.get(url, params=params, timeout=30)
            except requests.RequestException as e:  # transient network error: back off, do not crash
                log.warning("http_error_retry", err=type(e).__name__, attempt=attempt, wait=wait)
                wait = self._sleep_backoff(wait)
                continue
            if resp.status_code == 404 and quiet_404:
                return {}
            if resp.status_code == 429 or resp.status_code >= 500:  # 504 from pageviews killed a full run (2026-08-29)
                ra = resp.headers.get("Retry-After")
                if ra and ra.isdigit():  # server-stated wait wins over our schedule
                    wait = max(wait, min(float(ra), self.backoff_max))
                log.warning("http_retry", status=resp.status_code, attempt=attempt, wait=wait)
                wait = self._sleep_backoff(wait)
                continue
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()
            if isinstance(data, dict) and data.get("error", {}).get("code") == "maxlag":
                log.warning("maxlag_retry", attempt=attempt, wait=wait)
                wait = self._sleep_backoff(wait)
                continue
            time.sleep(_POLITE_DELAY_S)
            return data
        raise RuntimeError(f"exhausted retries for {url}")

    def _sleep_backoff(self, wait: float) -> float:
        j = random.uniform(0, wait * 0.25) if self.jitter else 0.0
        time.sleep(wait + j)
        return min(wait * 2, self.backoff_max)
