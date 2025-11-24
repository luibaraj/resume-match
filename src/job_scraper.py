"""
High-level scraping helpers built on top of the JSearch API client.

The helpers keep the logic modular so you can import them into other scripts
without coupling to a CLI or persistence layer.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from jsearch_client import JSearchClient


def normalize_job(record: Dict) -> Dict:
    """Flatten the raw API job record into a simpler shape."""
    highlights = record.get("job_highlights") or {}
    return {
        "id": record.get("job_id"),
        "title": record.get("job_title"),
        "employer": record.get("employer_name"),
        "publisher": record.get("job_publisher"),
        "employment_type": record.get("job_employment_type"),
        "location": record.get("job_location")
        or ", ".join(
            filter(
                None,
                [record.get("job_city"), record.get("job_state"), record.get("job_country")],
            )
        ),
        "is_remote": record.get("job_is_remote"),
        "posted_at": record.get("job_posted_at_datetime_utc") or record.get("job_posted_at"),
        "min_salary": record.get("job_min_salary"),
        "max_salary": record.get("job_max_salary"),
        "salary_period": record.get("job_salary_period"),
        "benefits": record.get("job_benefits") or [],
        "apply_links": [opt["apply_link"] for opt in record.get("apply_options", []) if opt.get("apply_link")],
        "description": record.get("job_description"),
        "highlights": {
            "qualifications": highlights.get("Qualifications") or [],
            "responsibilities": highlights.get("Responsibilities") or [],
            "benefits": highlights.get("Benefits") or [],
        },
    }


def collect_jobs(
    client: JSearchClient,
    *,
    query: str,
    pages: int = 1,
    per_page: int = 1,
    country: str = "us",
    language: Optional[str] = None,
    **filters,
) -> List[Dict]:
    """
    Collect jobs across pages. `per_page` uses the API's num_pages parameter to
    request multiple pages at once when desired.
    """
    all_jobs: List[Dict] = []
    for page in range(1, pages + 1):
        batch = client.search(
            query=query,
            page=page,
            num_pages=per_page,
            country=country,
            language=language,
            **filters,
        )
        all_jobs.extend(batch)
    return all_jobs


def collect_and_normalize(client: JSearchClient, **kwargs) -> List[Dict]:
    """Convenience wrapper to fetch jobs and normalize them."""
    raw_jobs = collect_jobs(client, **kwargs)
    return [normalize_job(job) for job in raw_jobs]


def save_json(path: Path, jobs: Iterable[Dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(list(jobs), f, indent=2)


def save_csv(path: Path, jobs: Iterable[Dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(jobs)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
