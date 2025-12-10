"""Minimal helper for pulling structured info from job descriptions with an LLM."""

from __future__ import annotations

import json
import os
import time
from typing import List, Optional, Tuple

import requests


def extract_job_requirements(
    job_description: str,
    *,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    endpoint: str = "https://api.openai.com/v1/chat/completions",
) -> Tuple[Optional[int], List[str], List[str], List[str], int]:
    """Send the description to an LLM and return the requested pieces and token usage."""

    text = job_description.strip()
    if not text:
        raise ValueError("job_description must contain text")

    token = api_key or os.getenv("OPENAI_API_KEY")
    if not token:
        raise RuntimeError("OPENAI_API_KEY is required")

    messages = [
        {
            "role": "system",
            "content": (
                "Extract four things from any job description: minimum_years_experience "
                "(integer or null), required_skills (list), preferred_skills (list), and "
                "responsibilities (list of short sentences). Respond using only valid JSON."
            ),
        },
        {
            "role": "user",
            "content": f"Job description:\n\n{text}",
        },
    ]

    response = None
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                json={"model": model, "messages": messages, "temperature": 0.3},
                timeout=60,
            )
            response.raise_for_status()
            break
        except requests.exceptions.RequestException:
            if attempt == max_attempts:
                raise
            time.sleep(2 ** (attempt - 1))

    if response is None:
        raise RuntimeError("Failed to retrieve a response from the OpenAI API")
    payload = response.json()
    try:
        raw_content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"LLM response missing message content: {payload}") from exc

    if raw_content is None:
        raw_content = ""
    elif not isinstance(raw_content, str):
        raw_content = str(raw_content)

    def _strip_code_fence(text: str) -> str:
        cleaned = text.strip()
        if cleaned.startswith("```") and cleaned.endswith("```"):
            cleaned = cleaned[3:-3].strip()
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].lstrip()
        return cleaned

    content = _strip_code_fence(raw_content)
    try:
        total_tokens = int(payload.get("usage", {}).get("total_tokens", 0))
    except (TypeError, ValueError):
        total_tokens = 0

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        snippet = raw_content.strip()
        if len(snippet) > 200:
            snippet = f"{snippet[:200]}..."
        snippet = snippet or "<empty>"
        raise ValueError(
            f"LLM response could not be parsed as JSON. Received: {snippet}"
        ) from exc

    min_years = parsed.get("minimum_years_experience")
    if isinstance(min_years, str):
        min_years = int(min_years.split()[0]) if min_years.strip() else None
    elif isinstance(min_years, bool):
        min_years = None

    def _ensure_list(key: str) -> List[str]:
        value = parsed.get(key, [])
        if isinstance(value, str):
            value = [value]
        return [str(item).strip() for item in value if str(item).strip()]

    required = _ensure_list("required_skills")
    preferred = _ensure_list("preferred_skills")
    responsibilities = _ensure_list("responsibilities")

    return min_years, required, preferred, responsibilities, total_tokens
