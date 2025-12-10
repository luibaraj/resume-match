"""Tests for the job extraction helper."""

import json
from typing import Any, Dict, Optional

import pytest

from src.job_extraction import extract_job_requirements


class _FakeResponse:
    def __init__(self, payload: Dict[str, Any]):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Dict[str, Any]:
        return self._payload


def _mock_completion_payload(
    total_tokens: Any,
    *,
    content: Optional[str] = None,
    wrap_in_fence: bool = False,
) -> Dict[str, Any]:
    structured = {
        "minimum_years_experience": 3,
        "required_skills": ["Python", "SQL"],
        "preferred_skills": ["Django"],
        "responsibilities": ["Ship production features"],
    }
    body = content if content is not None else json.dumps(structured)
    if wrap_in_fence:
        body = f"```json\n{body}\n```"
    return {
        "choices": [{"message": {"content": body}}],
        "usage": {"total_tokens": total_tokens},
    }


def test_extract_job_requirements_returns_total_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _mock_completion_payload(total_tokens=321, wrap_in_fence=True)
    monkeypatch.setattr(
        "src.job_extraction.requests.post",
        lambda *args, **kwargs: _FakeResponse(payload),
    )

    min_years, required, preferred, responsibilities, total_tokens = extract_job_requirements(
        "Detailed job description",
        api_key="test-token",
    )

    assert min_years == 3
    assert required == ["Python", "SQL"]
    assert preferred == ["Django"]
    assert responsibilities == ["Ship production features"]
    assert total_tokens == 321


def test_extract_job_requirements_defaults_total_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _mock_completion_payload(total_tokens="unknown")
    monkeypatch.setattr(
        "src.job_extraction.requests.post",
        lambda *args, **kwargs: _FakeResponse(payload),
    )

    *_, total_tokens = extract_job_requirements(
        "Another job description",
        api_key="test-token",
    )

    assert total_tokens == 0


def test_extract_job_requirements_raises_clear_error_for_non_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _mock_completion_payload(total_tokens=15, content="not JSON at all")
    monkeypatch.setattr(
        "src.job_extraction.requests.post",
        lambda *args, **kwargs: _FakeResponse(payload),
    )

    with pytest.raises(ValueError) as excinfo:
        extract_job_requirements(
            "Another job description",
            api_key="test-token",
        )

    assert "could not be parsed as JSON" in str(excinfo.value)
