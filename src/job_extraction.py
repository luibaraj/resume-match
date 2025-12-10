"""Minimal helper for pulling structured info from job descriptions with an LLM."""

from __future__ import annotations

import json
import os
from typing import List, Optional, Tuple

import requests


def extract_job_requirements(
    job_description: str,
    *,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    endpoint: str = "https://api.openai.com/v1/chat/completions",
) -> Tuple[Optional[int], List[str], List[str], List[str]]:
    """Send the description to an LLM and return the requested pieces."""

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

    response = requests.post(
        endpoint,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={"model": model, "messages": messages, "temperature": 0.3},
        timeout=60,
    )
    response.raise_for_status()

    content = response.json()["choices"][0]["message"]["content"]
    parsed = json.loads(content)

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

    return min_years, required, preferred, responsibilities
