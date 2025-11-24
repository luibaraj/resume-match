"""
Simple script that loops through predefined job titles and saves results to CSV.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

from job_scraper import collect_and_normalize, save_csv
from jsearch_client import JSearchClient

DEFAULT_JOB_TITLES: Sequence[str] = (
    "Data Scientist",
    "ML Engineer",
    "Machine Learning Engineer",
    "AI Engineer",
    "Research Engineer",
    "Data Analyst",
)
DEFAULT_OUTPUT_PATH = Path("/Users/luisbarajas/Desktop/Projects/jobs.csv")
DEFAULT_PAGES = 1
DEFAULT_PER_PAGE = 5
DEFAULT_COUNTRY = "us"


def fetch_jobs_for_titles(
    job_titles: Iterable[str],
    *,
    pages: int = DEFAULT_PAGES,
    per_page: int = DEFAULT_PER_PAGE,
    country: str = DEFAULT_COUNTRY,
    client: JSearchClient | None = None,
) -> List[dict]:
    """
    Fetch and normalize jobs for each title while tagging each row with its search query.
    """
    client = client or JSearchClient()
    aggregated: List[dict] = []
    for title in job_titles:
        jobs = collect_and_normalize(
            client,
            query=title,
            pages=pages,
            per_page=per_page,
            country=country,
        )
        for job in jobs:
            job["search_query"] = title
        aggregated.extend(jobs)
    return aggregated


def scrape_job_titles(
    job_titles: Iterable[str] = DEFAULT_JOB_TITLES,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    *,
    pages: int = DEFAULT_PAGES,
    per_page: int = DEFAULT_PER_PAGE,
    country: str = DEFAULT_COUNTRY,
    client: JSearchClient | None = None,
) -> List[dict]:
    jobs = fetch_jobs_for_titles(
        job_titles,
        pages=pages,
        per_page=per_page,
        country=country,
        client=client,
    )
    save_csv(output_path, jobs)
    return jobs


def main() -> None:
    jobs = scrape_job_titles()
    print(f"Saved {len(jobs)} jobs to {DEFAULT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
