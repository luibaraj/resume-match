"""
Lightweight client for the OpenWeb Ninja JSearch API.

Features:
- API-key auth read from `JSEARCH_API_KEY` env var.
- Simple rate limiting to avoid hitting provider caps.
- Exponential backoff with jitter for 429/5xx responses.
"""

from __future__ import annotations

import logging
import os
import random
import time
from typing import Dict, Iterable, List, Optional

import requests

DEFAULT_BASE_URL = "https://api.openwebninja.com/jsearch"
DEFAULT_RATE_LIMIT_PER_SEC = 2.0
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class JSearchError(Exception):
    """Raised when the JSearch API returns an error response."""


class JSearchClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        rate_limit_per_sec: float = DEFAULT_RATE_LIMIT_PER_SEC,
        max_retries: int = 4,
        backoff_factor: float = 0.75,
        session: Optional[requests.Session] = None,
        request_timeout: float = 30.0,
    ) -> None:
        self.api_key = api_key or os.getenv("JSEARCH_API_KEY")
        if not self.api_key:
            raise ValueError("API key is required. Set JSEARCH_API_KEY.")

        self.base_url = base_url.rstrip("/")
        self.rate_limit_per_sec = max(rate_limit_per_sec, 0)
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.session = session or requests.Session()
        self.request_timeout = request_timeout
        self._min_interval = 1.0 / self.rate_limit_per_sec if self.rate_limit_per_sec else 0
        self._last_request_at = 0.0
        self.logger = logging.getLogger(__name__)

    def search(self, *, query: str, page: int = 1, num_pages: int = 1, **filters) -> List[Dict]:
        """
        Search for jobs. The API supports returning multiple pages in one call via num_pages,
        but this method keeps it defaulted to 1 to allow the caller to handle pagination explicitly.
        """
        params = {"query": query, "page": page, "num_pages": num_pages}
        params.update(self._format_filters(filters))
        payload = self._get("/search", params=params)
        return payload.get("data", [])

    def job_details(
        self,
        *,
        job_ids: Iterable[str],
        country: str = "us",
        language: Optional[str] = None,
        fields: Optional[Iterable[str]] = None,
    ) -> List[Dict]:
        job_id_param = ",".join(job_ids)
        params = {"job_id": job_id_param, "country": country}
        if language:
            params["language"] = language
        if fields:
            params["fields"] = ",".join(fields)
        payload = self._get("/job-details", params=params)
        return payload.get("data", [])

    def estimated_salary(
        self,
        *,
        job_title: str,
        location: str,
        location_type: str = "ANY",
        years_of_experience: str = "ALL",
        fields: Optional[Iterable[str]] = None,
    ) -> List[Dict]:
        params = {
            "job_title": job_title,
            "location": location,
            "location_type": location_type,
            "years_of_experience": years_of_experience,
        }
        if fields:
            params["fields"] = ",".join(fields)
        payload = self._get("/estimated-salary", params=params)
        return payload.get("data", [])

    def _get(self, path: str, *, params: Dict) -> Dict:
        url = f"{self.base_url}{path}"
        headers = {"x-api-key": self.api_key}

        last_error = None
        for attempt in range(self.max_retries + 1):
            self.logger.info(
                "Calling %s with params=%s (attempt %s/%s)",
                url,
                params,
                attempt + 1,
                self.max_retries + 1,
            )
            self._throttle()
            try:
                response = self.session.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=self.request_timeout,
                )
            except requests.RequestException as exc:
                self.logger.error("Request to %s failed: %s", url, exc, exc_info=True)
                last_error = exc
                self._sleep_with_backoff(attempt)
                continue

            if response.status_code < 400:
                self.logger.info(
                    "Received %s from %s on attempt %s",
                    response.status_code,
                    url,
                    attempt + 1,
                )
                try:
                    return response.json()
                except ValueError as exc:
                    raise JSearchError(f"Invalid JSON from API: {exc}") from exc

            if response.status_code not in RETRYABLE_STATUS_CODES:
                self.logger.error(
                    "Non-retryable API error %s from %s: %s",
                    response.status_code,
                    url,
                    response.text,
                )
                raise JSearchError(f"API error {response.status_code}: {response.text}")

            self.logger.warning(
                "Received %s from API. Attempt %s/%s. Retrying.",
                response.status_code,
                attempt + 1,
                self.max_retries,
            )
            last_error = JSearchError(f"Retryable API error {response.status_code}")
            self._sleep_with_backoff(attempt)

        raise JSearchError(f"Request failed after retries: {last_error}")

    def _format_filters(self, filters: Dict) -> Dict:
        formatted = {}
        for key, value in filters.items():
            if value is None:
                continue
            if isinstance(value, bool):
                formatted[key] = str(value).lower()
            elif isinstance(value, (list, tuple, set)):
                formatted[key] = ",".join(str(v) for v in value)
            else:
                formatted[key] = value
        return formatted

    def _throttle(self) -> None:
        if not self._min_interval:
            return
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_at = time.monotonic()

    def _sleep_with_backoff(self, attempt: int) -> None:
        sleep_for = self.backoff_factor * (2**attempt) + random.uniform(0, 0.25)
        time.sleep(sleep_for)
