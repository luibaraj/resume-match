import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from job_scraper import collect_jobs, collect_and_normalize, normalize_job, save_csv, save_json


class DummyClient:
    def __init__(self):
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        page = kwargs.get("page", 0)
        return [{"job_id": f"job-{page}", "job_title": f"title-{page}"}]


class JobScraperTests(unittest.TestCase):
    def test_collect_jobs_iterates_pages(self):
        client = DummyClient()
        jobs = collect_jobs(
            client,
            query="data scientist in nyc",
            pages=2,
            per_page=2,
            work_from_home=True,
        )
        self.assertEqual(len(jobs), 2)
        self.assertEqual(client.calls[0]["num_pages"], 2)
        self.assertTrue(client.calls[0]["work_from_home"])
        self.assertEqual(client.calls[0]["query"], "data scientist in nyc")

    def test_collect_and_normalize_returns_flattened_jobs(self):
        class ClientWithDetails(DummyClient):
            def search(self, **kwargs):
                return [
                    {
                        "job_id": "123",
                        "job_title": "Engineer",
                        "employer_name": "Acme",
                        "job_publisher": "Indeed",
                        "job_employment_type": "FULLTIME",
                        "job_city": "Chicago",
                        "job_state": "IL",
                        "job_country": "US",
                        "job_description": "Build systems",
                        "job_highlights": {"Qualifications": ["Python"]},
                        "job_benefits": ["health_insurance"],
                        "apply_options": [{"apply_link": "https://apply.here"}],
                    }
                ]

        client = ClientWithDetails()
        jobs = collect_and_normalize(client, query="engineer")
        job = jobs[0]
        self.assertEqual(job["id"], "123")
        self.assertIn("Chicago", job["location"])
        self.assertEqual(job["highlights"]["qualifications"], ["Python"])
        self.assertEqual(job["apply_links"], ["https://apply.here"])

    def test_normalize_job_handles_missing_fields(self):
        record = {
            "job_id": "abc",
            "job_title": "Developer",
            "employer_name": "Globex",
            "job_publisher": "LinkedIn",
            "job_employment_type": "CONTRACTOR",
            "job_location": None,
            "job_city": "Austin",
            "job_state": "TX",
            "job_country": "US",
            "job_highlights": {},
            "apply_options": [{"apply_link": "http://example.com/apply"}, {"publisher": "Other"}],
        }
        normalized = normalize_job(record)
        self.assertEqual(normalized["location"], "Austin, TX, US")
        self.assertEqual(normalized["apply_links"], ["http://example.com/apply"])
        self.assertEqual(normalized["highlights"]["benefits"], [])

    def test_save_json_and_csv_write_files(self):
        jobs = [
            {
                "id": "1",
                "title": "Dev",
                "employer": "Acme",
                "apply_links": ["https://apply"],
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            json_path = tmp_path / "jobs.json"
            csv_path = tmp_path / "jobs.csv"

            save_json(json_path, jobs)
            with json_path.open() as f:
                data = json.load(f)
            self.assertEqual(data, jobs)

            save_csv(csv_path, jobs)
            with csv_path.open() as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            self.assertEqual(rows[0]["id"], "1")
            self.assertEqual(rows[0]["title"], "Dev")


if __name__ == "__main__":
    unittest.main()
