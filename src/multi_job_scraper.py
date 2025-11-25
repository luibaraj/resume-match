"""Simple script that loops through predefined job titles and persists results."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, List, Sequence

import boto3
from botocore.exceptions import BotoCoreError, ClientError

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
DEFAULT_PAGES = 4
DEFAULT_PER_PAGE = 5
DEFAULT_COUNTRY = "us"
DEFAULT_DATE_POSTED_FILTER = "today"
DEFAULT_S3_BUCKET = os.getenv("JOB_SCRAPER_S3_BUCKET")
DEFAULT_S3_PREFIX = os.getenv("JOB_SCRAPER_S3_PREFIX", "scraped-jobs")
VERBOSE = True

def _default_object_key(prefix: str | None = None) -> str:
    """Generate an S3 object key with a timestamp suffix."""
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    if not prefix:
        return f"jobs_{timestamp}.json"
    sanitized_prefix = prefix.strip("/")
    return f"{sanitized_prefix}/jobs_{timestamp}.json"


def upload_jobs_to_s3(
    jobs: Sequence[dict],
    bucket: str,
    *,
    key: str | None = None,
    s3_client: Any | None = None,
) -> str:
    """Serialize the scraped jobs as JSON and upload them to S3."""
    if not bucket:
        raise ValueError("An S3 bucket name must be provided to upload jobs.")
    object_key = key or _default_object_key(DEFAULT_S3_PREFIX)
    client = s3_client or boto3.client("s3")
    payload = json.dumps(jobs, ensure_ascii=False, default=str).encode("utf-8")
    try:
        client.put_object(
            Bucket=bucket,
            Key=object_key,
            Body=payload,
            ContentType="application/json",
        )
    except (BotoCoreError, ClientError) as exc:
        raise RuntimeError(f"Failed to upload jobs to s3://{bucket}/{object_key}") from exc
    return f"s3://{bucket}/{object_key}"


def fetch_jobs_for_titles(
    job_titles: Iterable[str],
    *,
    pages: int = DEFAULT_PAGES,
    per_page: int = DEFAULT_PER_PAGE,
    country: str = DEFAULT_COUNTRY,
    date_posted: str = DEFAULT_DATE_POSTED_FILTER,
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
            date_posted=date_posted,
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
    date_posted: str = DEFAULT_DATE_POSTED_FILTER,
    client: JSearchClient | None = None,
    s3_bucket: str | None = DEFAULT_S3_BUCKET,
    s3_key: str | None = None,
    s3_client: Any | None = None,
) -> List[dict]:
    jobs = fetch_jobs_for_titles(
        job_titles,
        pages=pages,
        per_page=per_page,
        country=country,
        date_posted=date_posted,
        client=client,
    )
    save_csv(output_path, jobs)
    if s3_bucket:
        s3_uri = upload_jobs_to_s3(jobs, s3_bucket, key=s3_key, s3_client=s3_client)
        print(f"Uploaded jobs to {s3_uri}")
    return jobs


def main() -> None:
    jobs = scrape_job_titles()
    print(f"Saved {len(jobs)} jobs to {DEFAULT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
