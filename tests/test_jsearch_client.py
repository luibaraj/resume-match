import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from jsearch_client import JSearchClient, JSearchError


class FakeResponse:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append({"url": url, "params": params, "headers": headers, "timeout": timeout})
        try:
            return self.responses.pop(0)
        except IndexError:
            raise AssertionError("No more fake responses queued")


class JSearchClientTests(unittest.TestCase):
    def setUp(self):
        self.session = FakeSession([])

    def _build_client(self):
        return JSearchClient(api_key="test", session=self.session, rate_limit_per_sec=0, max_retries=1)

    def test_search_builds_params_and_returns_data(self):
        payload = {"data": [{"job_id": "job-1"}]}
        self.session.responses.append(FakeResponse(200, payload))
        client = self._build_client()

        result = client.search(query="python developer in chicago", page=2, num_pages=3, country="us", language="en")

        self.assertEqual(result, payload["data"])
        call = self.session.calls[0]
        self.assertIn("/search", call["url"])
        self.assertEqual(call["params"]["query"], "python developer in chicago")
        self.assertEqual(call["params"]["page"], 2)
        self.assertEqual(call["params"]["num_pages"], 3)
        self.assertEqual(call["params"]["country"], "us")
        self.assertEqual(call["params"]["language"], "en")
        self.assertEqual(call["headers"]["x-api-key"], "test")

    def test_format_filters_handles_booleans_and_sequences(self):
        client = self._build_client()
        filters = {
            "work_from_home": True,
            "employment_types": ["FULLTIME", "PARTTIME"],
            "radius": 5,
            "language": None,
        }
        formatted = client._format_filters(filters)
        self.assertEqual(formatted["work_from_home"], "true")
        self.assertEqual(formatted["employment_types"], "FULLTIME,PARTTIME")
        self.assertEqual(formatted["radius"], 5)
        self.assertNotIn("language", formatted)

    def test_retryable_error_then_success(self):
        payload = {"data": [{"job_id": "job-2"}]}
        self.session.responses.extend(
            [
                FakeResponse(429, text="Too Many Requests"),
                FakeResponse(200, payload),
            ]
        )
        client = self._build_client()
        client.logger.disabled = True
        client._sleep_with_backoff = lambda *_args, **_kwargs: None

        result = client.search(query="qa engineer")

        self.assertEqual(result, payload["data"])
        self.assertEqual(len(self.session.calls), 2)

    def test_non_retryable_error_raises(self):
        self.session.responses.append(FakeResponse(404, text="Missing"))
        client = self._build_client()

        with self.assertRaises(JSearchError):
            client.search(query="golang dev")


if __name__ == "__main__":
    unittest.main()
