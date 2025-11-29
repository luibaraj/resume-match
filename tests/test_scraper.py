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
    def fake_get(url, params=None, headers=None):
        index = len(captured_calls)
        response_data = responses[index]
        captured_calls.append({"url": url, "params": params, "headers": headers})
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

    result = scraper.scrape_jobs("python", 1, 2)

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

    result = scraper.scrape_jobs("engineer", 0, 1)

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

    result = scraper.scrape_jobs("dev", 5, 2)
    output = capsys.readouterr().out

    assert result == {"batches": [{"data": ["ok"]}, []]}
    assert "Request to" in output
    assert "status code 500" in output
    assert len(captured_calls) == 3
    assert captured_calls[0]["params"]["page"] == 5
    assert captured_calls[1]["params"]["page"] == 7


def test_uses_increment_by_num_pages_after_success(monkeypatch):
    monkeypatch.setenv("JSEARCH_API_KEY", "token")
    responses = [
        {"status": 200, "json": {"data": [1]}},
        {"status": 200, "json": []},
    ]
    captured_calls = []
    monkeypatch.setattr(
        scraper.requests, "get", build_mock_get(responses, captured_calls)
    )

    scraper.scrape_jobs("analyst", 1, 3)

    pages = [call["params"]["page"] for call in captured_calls]
    assert pages == [1, 4]


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

    scraper.scrape_jobs("qa", 2, 5)
    output = capsys.readouterr().out

    pages = [call["params"]["page"] for call in captured_calls]
    assert pages == [2, 7]
    assert "Bad Request" in output


def test_headers_include_authorization_and_accept(monkeypatch):
    api_key = "expected_key"
    monkeypatch.setenv("JSEARCH_API_KEY", api_key)
    responses = [{"status": 200, "json": []}]
    captured_calls = []
    monkeypatch.setattr(
        scraper.requests, "get", build_mock_get(responses, captured_calls)
    )

    scraper.scrape_jobs("java", 0, 1)
    headers = captured_calls[0]["headers"]

    assert headers["Authorization"] == f"Bearer {api_key}"
    assert headers["Accept"] == "application/json"


def test_location_and_date_posted_parameters_are_constant(monkeypatch):
    monkeypatch.setenv("JSEARCH_API_KEY", "token")
    responses = [{"status": 200, "json": []}]
    captured_calls = []
    monkeypatch.setattr(
        scraper.requests, "get", build_mock_get(responses, captured_calls)
    )

    scraper.scrape_jobs("golang", 10, 4)
    params = captured_calls[0]["params"]

    assert params["location"] == "US"
    assert params["date_posted"] == "today"


def test_missing_api_key_raises_keyerror(monkeypatch):
    monkeypatch.delenv("JSEARCH_API_KEY", raising=False)

    with pytest.raises(KeyError):
        scraper.scrape_jobs("rust", 1, 1)


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

    scraper.scrape_jobs("c++", 0, 2)
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

    scraper.scrape_jobs("cloud", 3, 2)
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

    result = scraper.scrape_jobs("sql", 0, 2)

    assert result == {"batches": [{"data": ["first"]}, []]}
    pages = [call["params"]["page"] for call in captured_calls]
    assert pages == [0, 2]


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

    result = scraper.scrape_jobs("frontend", 1, 1)

    assert result["batches"][0]["data"] == ["batch1"]
    assert result["batches"][1]["data"] == ["batch2"]
