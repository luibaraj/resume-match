from urllib.parse import urlencode

import pytest

import src.scraper as scraper


class MockResponse:
    def __init__(self, status_code, json_data, url, reason=""):
        self.status_code = status_code
        self._json_data = json_data
        self.url = url
        self.reason = reason

    def json(self):
        return self._json_data


def build_mock_get(responses, captured_calls):
    def fake_get(url, params=None, headers=None, timeout=None, **kwargs):
        index = len(captured_calls)
        response_data = responses[index]
        captured_calls.append(
            {"url": url, "params": params, "headers": headers, "timeout": timeout}
        )
        full_url = url
        if params:
            full_url = f"{url}?{urlencode(params)}"
        return MockResponse(
            response_data["status"],
            response_data.get("json"),
            full_url,
            response_data.get("reason", ""),
        )

    return fake_get


def test_returns_batches_of_successful_responses(monkeypatch):
    monkeypatch.setenv("JSEARCH_API_KEY", "token")
    responses = [
        {"status": 200, "json": {"data": [1]}},
        {"status": 200, "json": {"data": [2, 3]}},
        {"status": 200, "json": []},
    ]
    captured_calls = []
    monkeypatch.setattr(
        scraper.requests, "get", build_mock_get(responses, captured_calls)
    )

    result = scraper.scrape_jobs("python", country="US", start_page=1)

    assert result == {"batches": [{"data": [1]}, {"data": [2, 3]}, []]}
    assert len(captured_calls) == 3


def test_stops_on_empty_list_response(monkeypatch):
    monkeypatch.setenv("JSEARCH_API_KEY", "token")
    responses = [
        {"status": 200, "json": []},
        {"status": 200, "json": {"data": ["should not be reached"]}},
    ]
    captured_calls = []
    monkeypatch.setattr(
        scraper.requests, "get", build_mock_get(responses, captured_calls)
    )

    result = scraper.scrape_jobs("engineer", country="US", start_page=0)

    assert result == {"batches": [[]]}
    assert len(captured_calls) == 1


def test_skips_failed_batch_and_continues(monkeypatch, capsys):
    monkeypatch.setenv("JSEARCH_API_KEY", "token")
    responses = [
        {"status": 500, "json": {"error": "Server Error"}, "reason": "Server Error"},
        {"status": 200, "json": {"data": ["ok"]}},
        {"status": 200, "json": []},
    ]
    captured_calls = []
    monkeypatch.setattr(
        scraper.requests, "get", build_mock_get(responses, captured_calls)
    )

    result = scraper.scrape_jobs("dev", country="US", start_page=5)
    output = capsys.readouterr().out

    assert result == {"batches": [{"data": ["ok"]}, []]}
    assert "Request to" in output
    assert "status code 500" in output
    assert len(captured_calls) == 3
    pages = [call["params"]["page"] for call in captured_calls]
    assert pages == [5, 5, 15]


def test_current_page_increments_by_ten_after_success(monkeypatch):
    monkeypatch.setenv("JSEARCH_API_KEY", "token")
    responses = [
        {"status": 200, "json": {"data": [1]}},
        {"status": 200, "json": []},
    ]
    captured_calls = []
    monkeypatch.setattr(
        scraper.requests, "get", build_mock_get(responses, captured_calls)
    )

    scraper.scrape_jobs("analyst", country="US", start_page=1)

    pages = [call["params"]["page"] for call in captured_calls]
    assert pages == [1, 11]


def test_increments_after_failure(monkeypatch, capsys):
    monkeypatch.setenv("JSEARCH_API_KEY", "token")
    responses = [
        {"status": 400, "json": {"error": "Bad Request"}, "reason": "Bad Request"},
        {"status": 200, "json": []},
    ]
    captured_calls = []
    monkeypatch.setattr(
        scraper.requests, "get", build_mock_get(responses, captured_calls)
    )

    scraper.scrape_jobs("qa", country="US", start_page=2)
    output = capsys.readouterr().out

    pages = [call["params"]["page"] for call in captured_calls]
    assert pages == [2, 2]
    assert "Bad Request" in output


def test_headers_include_api_key_and_accept(monkeypatch):
    api_key = "expected_key"
    monkeypatch.setenv("JSEARCH_API_KEY", api_key)
    responses = [{"status": 200, "json": []}]
    captured_calls = []
    monkeypatch.setattr(
        scraper.requests, "get", build_mock_get(responses, captured_calls)
    )

    scraper.scrape_jobs("java", country="US", start_page=0)
    headers = captured_calls[0]["headers"]

    assert headers["x-api-key"] == api_key
    assert headers["Accept"] == "application/json"


def test_country_and_constant_parameters(monkeypatch):
    monkeypatch.setenv("JSEARCH_API_KEY", "token")
    responses = [{"status": 200, "json": []}]
    captured_calls = []
    monkeypatch.setattr(
        scraper.requests, "get", build_mock_get(responses, captured_calls)
    )

    scraper.scrape_jobs("golang", country="US", start_page=10)
    params = captured_calls[0]["params"]

    assert params["country"] == "US"
    assert params["num_pages"] == 10
    assert params["date_posted"] == "today"


def test_missing_api_key_raises_keyerror(monkeypatch):
    monkeypatch.delenv("JSEARCH_API_KEY", raising=False)

    with pytest.raises(KeyError):
        scraper.scrape_jobs("rust", country="US", start_page=1)


def test_no_print_during_successful_requests(monkeypatch, capsys):
    monkeypatch.setenv("JSEARCH_API_KEY", "token")
    responses = [
        {"status": 200, "json": {"data": [1]}},
        {"status": 200, "json": []},
    ]
    captured_calls = []
    monkeypatch.setattr(
        scraper.requests, "get", build_mock_get(responses, captured_calls)
    )

    scraper.scrape_jobs("c++", country="US", start_page=0)
    output = capsys.readouterr().out

    assert output == ""


def test_message_format_for_failed_request(monkeypatch, capsys):
    monkeypatch.setenv("JSEARCH_API_KEY", "token")
    responses = [
        {"status": 404, "json": {"error": "Not Found"}, "reason": "Not Found"},
        {"status": 200, "json": []},
    ]
    captured_calls = []
    monkeypatch.setattr(
        scraper.requests, "get", build_mock_get(responses, captured_calls)
    )

    scraper.scrape_jobs("cloud", country="US", start_page=3)
    output = capsys.readouterr().out

    assert "Request to" in output
    assert "status code 404" in output
    assert "Not Found" in output


def test_non_list_payload_does_not_stop_loop(monkeypatch):
    monkeypatch.setenv("JSEARCH_API_KEY", "token")
    responses = [
        {"status": 200, "json": {"data": ["first"]}},
        {"status": 200, "json": []},
    ]
    captured_calls = []
    monkeypatch.setattr(
        scraper.requests, "get", build_mock_get(responses, captured_calls)
    )

    result = scraper.scrape_jobs("sql", country="US", start_page=0)

    assert result == {"batches": [{"data": ["first"]}, []]}
    pages = [call["params"]["page"] for call in captured_calls]
    assert pages == [0, 10]


def test_batches_preserve_order(monkeypatch):
    monkeypatch.setenv("JSEARCH_API_KEY", "token")
    responses = [
        {"status": 200, "json": {"data": ["batch1"]}},
        {"status": 200, "json": {"data": ["batch2"]}},
        {"status": 200, "json": []},
    ]
    captured_calls = []
    monkeypatch.setattr(
        scraper.requests, "get", build_mock_get(responses, captured_calls)
    )

    result = scraper.scrape_jobs("frontend", country="US", start_page=1)

    assert result["batches"][0]["data"] == ["batch1"]
    assert result["batches"][1]["data"] == ["batch2"]


def test_retries_before_success(monkeypatch):
    monkeypatch.setenv("JSEARCH_API_KEY", "token")
    responses = [
        {"status": 502, "json": {"error": "Bad Gateway"}, "reason": "Bad Gateway"},
        {"status": 200, "json": []},
    ]
    captured_calls = []
    monkeypatch.setattr(
        scraper.requests, "get", build_mock_get(responses, captured_calls)
    )
    sleeps = []
    monkeypatch.setattr(scraper.time, "sleep", lambda seconds: sleeps.append(seconds))

    scraper.scrape_jobs("retry", country="US", start_page=0)

    assert len(captured_calls) == 2
    assert [call["params"]["page"] for call in captured_calls] == [0, 0]
    assert sleeps == [1]


def test_gives_up_after_max_attempts(monkeypatch):
    monkeypatch.setenv("JSEARCH_API_KEY", "token")
    responses = [
        {"status": 502, "json": {"error": "Bad Gateway"}, "reason": "Bad Gateway"},
        {"status": 502, "json": {"error": "Bad Gateway"}, "reason": "Bad Gateway"},
        {"status": 502, "json": {"error": "Bad Gateway"}, "reason": "Bad Gateway"},
        {"status": 200, "json": []},
    ]
    captured_calls = []
    monkeypatch.setattr(
        scraper.requests, "get", build_mock_get(responses, captured_calls)
    )
    sleeps = []
    monkeypatch.setattr(scraper.time, "sleep", lambda seconds: sleeps.append(seconds))

    scraper.scrape_jobs("give up", country="US", start_page=0)

    pages = [call["params"]["page"] for call in captured_calls]
    assert pages[:3] == [0, 0, 0]
    assert pages[3] == 10
    assert sleeps == [1, 2]
