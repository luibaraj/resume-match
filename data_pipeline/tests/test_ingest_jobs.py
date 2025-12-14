import sys
from pathlib import Path

import pandas as pd
import pytest

# Ensure the project root is on the path for importing utils.*
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import utils.ingest_jobs as ingest_jobs  # noqa: E402


def test_injest_jobs_aggregates_batches(monkeypatch):
    calls = []

    def fake_scrape(term, country, start_page):
        calls.append((term, country, start_page))
        return {
            "batches": [
                {"data": [{"id": f"{term}-1"}]},
                {"data": [{"id": f"{term}-2"}]},
            ]
        }

    monkeypatch.setattr(ingest_jobs, "scrape_jobs", fake_scrape)
    monkeypatch.setattr(ingest_jobs, "SEARCH_TERMS", ["one", "two"])

    df = ingest_jobs.injest_jobs()

    assert calls == [("one", "us", 1), ("two", "us", 1)]
    assert len(df) == 4
    assert set(df["id"]) == {"one-1", "one-2", "two-1", "two-2"}


def test_injest_jobs_handles_empty_batches(monkeypatch):
    def fake_scrape(term, country, start_page):
        return {"batches": [{"data": []}, {"data": []}]}

    monkeypatch.setattr(ingest_jobs, "scrape_jobs", fake_scrape)
    monkeypatch.setattr(ingest_jobs, "SEARCH_TERMS", ["empty"])

    df = ingest_jobs.injest_jobs()

    assert isinstance(df, pd.DataFrame)
    assert df.empty
