import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from jsearch_client import JSearchError
from multi_job_scraper import fetch_jobs_for_titles, scrape_job_titles


class MultiJobScraperTests(unittest.TestCase):
    def test_fetch_jobs_for_titles_adds_query_and_aggregates(self):
        dummy_client = object()
        job_titles = ["Data Scientist", "ML Engineer"]
        with mock.patch("multi_job_scraper.collect_and_normalize") as mock_collect:
            mock_collect.side_effect = [
                [{"id": "1", "title": "Role 1"}],
                [{"id": "2", "title": "Role 2"}],
            ]
            jobs = fetch_jobs_for_titles(
                job_titles,
                pages=1,
                per_page=10,
                country="us",
                client=dummy_client,
            )

        self.assertEqual(len(jobs), 2)
        self.assertEqual(jobs[0]["search_query"], "Data Scientist")
        self.assertEqual(jobs[1]["search_query"], "ML Engineer")
        self.assertEqual(mock_collect.call_count, 2)
        self.assertIs(mock_collect.call_args_list[0].args[0], dummy_client)

    def test_fetch_jobs_for_titles_handles_empty_results(self):
        with mock.patch("multi_job_scraper.collect_and_normalize", side_effect=[[], []]):
            jobs = fetch_jobs_for_titles(["Data Scientist", "ML Engineer"], client=object())
        self.assertEqual(jobs, [])

    def test_fetch_jobs_for_titles_propagates_errors(self):
        with mock.patch("multi_job_scraper.collect_and_normalize", side_effect=JSearchError("boom")):
            with self.assertRaises(JSearchError):
                fetch_jobs_for_titles(["Data Scientist"], client=object())

    def test_fetch_jobs_for_titles_passes_pagination_params(self):
        with mock.patch("multi_job_scraper.collect_and_normalize", return_value=[]) as mock_collect:
            fetch_jobs_for_titles(
                ["Data Scientist"],
                pages=3,
                per_page=5,
                country="ca",
                date_posted="last_7_days",
                client=object(),
            )
        kwargs = mock_collect.call_args.kwargs
        self.assertEqual(kwargs["pages"], 3)
        self.assertEqual(kwargs["per_page"], 5)
        self.assertEqual(kwargs["country"], "ca")
        self.assertEqual(kwargs["date_posted"], "last_7_days")

    def test_fetch_jobs_for_titles_uses_last_24_hour_default(self):
        with mock.patch("multi_job_scraper.collect_and_normalize", return_value=[]) as mock_collect:
            fetch_jobs_for_titles(["Data Scientist"], client=object())
        self.assertEqual(mock_collect.call_args.kwargs["date_posted"], "today")

    def test_fetch_jobs_for_titles_handles_non_ascii(self):
        with mock.patch("multi_job_scraper.collect_and_normalize") as mock_collect:
            mock_collect.side_effect = [[{"id": "1", "title": "Café", "employer": "Año"}]]
            jobs = fetch_jobs_for_titles(["Data Scientist"], client=object())
        self.assertEqual(jobs[0]["title"], "Café")
        self.assertEqual(jobs[0]["employer"], "Año")

    def test_fetch_jobs_for_titles_keeps_duplicates(self):
        with mock.patch("multi_job_scraper.collect_and_normalize") as mock_collect:
            mock_collect.side_effect = [
                [{"id": "dup", "title": "Role"}],
                [{"id": "dup", "title": "Role"}],
            ]
            jobs = fetch_jobs_for_titles(["Data Scientist", "ML Engineer"], client=object())
        dup_count = sum(1 for job in jobs if job["id"] == "dup")
        self.assertEqual(dup_count, 2)

    def test_scrape_job_titles_propagates_save_errors(self):
        with mock.patch("multi_job_scraper.fetch_jobs_for_titles", return_value=[{"id": "1"}]):
            with mock.patch("multi_job_scraper.save_csv", side_effect=PermissionError("denied")):
                with tempfile.TemporaryDirectory() as tmpdir:
                    output_path = Path(tmpdir) / "jobs.csv"
                    with self.assertRaises(PermissionError):
                        scrape_job_titles(
                            ["Data Scientist"],
                            output_path=output_path,
                            client=object(),
                        )

    def test_scrape_job_titles_saves_to_path(self):
        jobs_payload = [{"id": "1", "title": "Role"}]
        with mock.patch("multi_job_scraper.fetch_jobs_for_titles", return_value=jobs_payload) as mock_fetch:
            with mock.patch("multi_job_scraper.save_csv") as mock_save:
                with tempfile.TemporaryDirectory() as tmpdir:
                    output_path = Path(tmpdir) / "jobs.csv"
                    result = scrape_job_titles(
                        ["Data Scientist"],
                        output_path=output_path,
                        client=object(),
                    )

        self.assertEqual(result, jobs_payload)
        mock_fetch.assert_called_once()
        mock_save.assert_called_once_with(output_path, jobs_payload)


if __name__ == "__main__":
    unittest.main()
