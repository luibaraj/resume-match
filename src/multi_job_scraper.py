"""Simple script that loops through predefined job titles and persists results."""

from __future__ import annotations

import json
import logging
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
_LOGGING_CONFIGURED = False
logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    """Turn on verbose logging for API/AWS calls when enabled."""
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED or not VERBOSE:
        return
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not root.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        )
    logging.getLogger("jsearch_client").setLevel(logging.INFO)
    _LOGGING_CONFIGURED = True


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
    _configure_logging()
    if not bucket:
        raise ValueError("An S3 bucket name must be provided to upload jobs.")
    object_key = key or _default_object_key(DEFAULT_S3_PREFIX)
    client = s3_client or boto3.client("s3")
    payload = json.dumps(jobs, ensure_ascii=False, default=str).encode("utf-8")
    logger.info("Uploading %s jobs to s3://%s/%s", len(jobs), bucket, object_key)
    try:
        client.put_object(
            Bucket=bucket,
            Key=object_key,
            Body=payload,
            ContentType="application/json",
        )
    except (BotoCoreError, ClientError) as exc:
        logger.error(
            "Failed to upload jobs to s3://%s/%s: %s",
            bucket,
            object_key,
            exc,
            exc_info=True,
        )
        raise RuntimeError(f"Failed to upload jobs to s3://{bucket}/{object_key}") from exc
    logger.info(
        "Uploaded %s bytes to s3://%s/%s",
        len(payload),
        bucket,
        object_key,
    )
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
    _configure_logging()
    titles = list(job_titles)
    logger.info(
        "Fetching jobs for %s titles (pages=%s, per_page=%s, country=%s, date_posted=%s)",
        len(titles),
        pages,
        per_page,
        country,
        date_posted,
    )
    client = client or JSearchClient()
    aggregated: List[dict] = []
    for title in titles:
        logger.info("Requesting jobs for search query '%s'", title)
        try:
            jobs = collect_and_normalize(
                client,
                query=title,
                pages=pages,
                per_page=per_page,
                country=country,
                date_posted=date_posted,
            )
        except Exception as exc:
            logger.error("Error fetching jobs for '%s': %s", title, exc, exc_info=True)
            raise
        logger.info("Fetched %s jobs for '%s'", len(jobs), title)
        for job in jobs:
            job["search_query"] = title
        aggregated.extend(jobs)
    logger.info("Aggregated %s jobs across %s titles", len(aggregated), len(titles))
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
    _configure_logging()
    titles = list(job_titles)
    logger.info("Starting scrape for %s job titles", len(titles))
    jobs = fetch_jobs_for_titles(
        titles,
        pages=pages,
        per_page=per_page,
        country=country,
        date_posted=date_posted,
        client=client,
    )
    logger.info("Persisting %s jobs to %s", len(jobs), output_path)
    save_csv(output_path, jobs)
    if s3_bucket:
        s3_uri = upload_jobs_to_s3(jobs, s3_bucket, key=s3_key, s3_client=s3_client)
        print(f"Uploaded jobs to {s3_uri}")
    else:
        logger.info("Skipping S3 upload because no bucket was provided.")
    return jobs


def main() -> None:
    jobs = scrape_job_titles()
    print(f"Saved {len(jobs)} jobs to {DEFAULT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
