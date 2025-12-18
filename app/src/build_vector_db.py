# Enables postponed evaluation of type annotations (PEP 563 / PEP 649 behavior)
from __future__ import annotations

# Standard library imports
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Iterable


# ------------------------------------------------------------------------------
# Project root resolution and sys.path setup
# ------------------------------------------------------------------------------

def resolve_project_root(marker: str = "src") -> Path:
    """
    Walk upward from the current file's location to find the repository root.

    The repository root is identified as the first directory containing
    the specified `marker` directory (default: "src").
    """
    start = Path(__file__).resolve()

    # Check the current directory and each parent directory
    for candidate in (start.parent, *start.parents):
        if (candidate / marker).exists():
            return candidate

    # If no suitable directory is found, fail fast
    raise RuntimeError("Unable to locate the project root. Ensure `src/` exists.")


# Determine project root and src directory
PROJECT_ROOT = resolve_project_root()
SRC_DIR = PROJECT_ROOT / "src"


def _report_execution_time(start_time: float) -> None:
    """Emit the total runtime for observability."""
    elapsed = time.perf_counter() - start_time
    print(f"build_vector_db.py finished in {elapsed:.2f} seconds")


SCRIPT_START_TIME = time.perf_counter()

# Ensure src/ is importable without requiring PYTHONPATH configuration
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ------------------------------------------------------------------------------
# Environment file parsing and loading
# ------------------------------------------------------------------------------

def _parse_env_assignment(line: str) -> tuple[str, str] | None:
    """
    Parse a single line from an env file into a (key, value) pair.

    Supports:
      - Blank lines
      - Commented lines (# ...)
      - `export KEY=value` syntax
      - Quoted or unquoted values

    Returns None if the line does not represent a valid assignment.
    """
    text = line.strip()

    # Ignore empty lines and comments
    if not text or text.startswith("#"):
        return None

    # Strip optional `export ` prefix
    if text.startswith("export "):
        text = text[len("export ") :]

    # Must contain an assignment operator
    if "=" not in text:
        return None

    key, value = text.split("=", 1)
    key = key.strip()

    # Ignore malformed keys
    if not key:
        return None

    # Remove surrounding quotes from the value
    value = value.strip().strip('"').strip("'")
    return key, value


def load_env_file(paths: Iterable[Path], *, overwrite: bool = False) -> Path | None:
    """
    Load environment variables from the first existing file in `paths`.

    - Variables are added to os.environ
    - Existing variables are preserved unless overwrite=True
    - Returns the path of the loaded env file, or None if none were found
    """
    for path in paths:
        if not path.exists():
            continue

        loaded = 0
        with path.open(encoding="utf-8") as handle:
            for raw_line in handle:
                assignment = _parse_env_assignment(raw_line)
                if not assignment:
                    continue

                key, value = assignment

                # Respect existing environment unless overwrite is enabled
                if overwrite or key not in os.environ:
                    os.environ[key] = value
                    loaded += 1

        print(f"Loaded {loaded} environment variables from {path}")
        return path

    # No env file found; rely entirely on existing environment
    print("No env file found; relying on existing environment variables.")
    return None


# Candidate env file locations (checked in order)
ENV_FILES = [
    PROJECT_ROOT / ".env.local",
    PROJECT_ROOT / ".env",
    PROJECT_ROOT / "config" / ".env",
    PROJECT_ROOT / "env_vars.txt",
]

# Attempt to load environment variables
load_env_file(ENV_FILES)


# ------------------------------------------------------------------------------
# Vector database path resolution
# ------------------------------------------------------------------------------

def _resolve_vector_db_path() -> Path:
    """
    Resolve and prepare VECTOR_DB_PATH.

    Requirements:
      - VECTOR_DB_PATH must be defined in the environment
      - It must point to a directory inside the project repository
      - Any existing directory is deleted and recreated
    """
    value = os.environ.get("VECTOR_DB_PATH")
    if not value:
        raise RuntimeError(
            "VECTOR_DB_PATH must be set to a directory inside this repository."
        )

    path = Path(value).expanduser().resolve()

    # Enforce repository containment for safety
    try:
        path.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise RuntimeError(
            "VECTOR_DB_PATH must point inside the project directory."
        ) from exc

    # Reset the directory to ensure a clean build
    if path.exists():
        shutil.rmtree(path)

    path.mkdir(parents=True, exist_ok=True)
    return path


# Final, validated vector DB path
VECTOR_DB_PATH = _resolve_vector_db_path()


# ------------------------------------------------------------------------------
# Environment validation helper
# ------------------------------------------------------------------------------

def require_env(keys: list[str]) -> dict[str, str]:
    """
    Ensure all required environment variables are present and non-empty.

    Raises RuntimeError if any are missing.
    """
    missing = [key for key in keys if not os.environ.get(key)]
    if missing:
        raise RuntimeError(
            f"Missing environment variables: {', '.join(missing)}"
        )

    return {key: os.environ[key] for key in keys}


# ------------------------------------------------------------------------------
# Job scraping and ingestion pipeline
# ------------------------------------------------------------------------------

from concurrent.futures import ThreadPoolExecutor, as_completed

from scraper import scrape_jobs
from s3_uploader import upload_job_records_to_s3


# Job search terms to query
SEARCH_TERMS = [
    "data scientist",
    "research engineer",
    "machine learning engineer",
    "ai engineer",
    "data analyst",
    "data science",
]

# AWS-related environment variables required for S3 upload
S3_ENV_VARS = [
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_DEFAULT_REGION",
    "JOB_SCRAPER_S3_BUCKET",
    "JOB_SCRAPER_S3_PREFIX",
]


def _fetch_jobs_for_term(
    term: str,
    *,
    country: str,
    start_page: int,
    num_pages: int,
) -> list[dict]:
    """
    Fetch and normalize job listings for a single search term.

    This function is intentionally isolated so it can be executed
    concurrently across multiple threads.
    """
    # Invoke the external job scraper for the given term
    response = scrape_jobs(
        term,
        country=country,
        start_page=start_page,
        num_pages=num_pages,
    )

    # Safely extract batch-level results from the response payload
    batches = response.get("batches", []) if isinstance(response, dict) else []

    # Accumulator for normalized job records for this search term
    term_records: list[dict] = []

    for batch in batches:
        # Handle both dictionary-based and raw list batch formats
        data_block = batch.get("data") if isinstance(batch, dict) else batch
        if not isinstance(data_block, list):
            continue

        for job in data_block:
            # Ensure each job entry is a dictionary before processing
            if not isinstance(job, dict):
                continue

            # Copy the job payload to avoid mutating source data
            payload = dict(job)

            # Attach the originating search term for traceability
            payload.setdefault("search_term", term)
            term_records.append(payload)

    # Emit per-term record counts for observability
    print(f"{term}: {len(term_records)} rows")

    return term_records


def fetch_jobs(
    search_terms: list[str],
    *,
    country: str = "us",
    start_page: int = 1,
    num_pages: int = 10,
    max_workers: int | None = None,
) -> list[dict]:
    """
    Fetch job listings for each search term concurrently and normalize
    results into a flat list of job dictionaries.

    Concurrency is applied at the search-term level, which is appropriate
    for I/O-bound scraping workloads.
    """
    # Ensure required API credentials are available before execution
    require_env(["JSEARCH_API_KEY"])

    # Shared accumulator for all normalized job records
    all_records: list[dict] = []

    # Initialize a thread pool to execute per-term fetches in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit one concurrent task per search term
        futures = {
            executor.submit(
                _fetch_jobs_for_term,
                term,
                country=country,
                start_page=start_page,
                num_pages=num_pages,
            ): term
            for term in search_terms
        }

        # Collect results as each task completes
        for future in as_completed(futures):
            records = future.result()

            # Extend the shared accumulator defensively
            if records:
                all_records.extend(records)

    return all_records


# Execute job fetch across all configured search terms
job_records = fetch_jobs(SEARCH_TERMS, max_workers=8)

# Fail fast if no data was returned
if not job_records:
    raise RuntimeError("No job listings found for the requested search terms.")


# ------------------------------------------------------------------------------
# Upload to S3
# ------------------------------------------------------------------------------

# Validate AWS configuration
require_env(S3_ENV_VARS)

upload_success, s3_uri = upload_job_records_to_s3(job_records)
if not upload_success:
    raise RuntimeError(
        "Failed to upload job listings to S3; aborting vector DB build."
    )

if s3_uri:
    print(f"Job listings archived in {s3_uri}")


# ------------------------------------------------------------------------------
# Offline vector database build (Chroma)
# ------------------------------------------------------------------------------

from offline_helpers import run_offline_chroma_pipeline

# Ensure OpenAI credentials are present
require_env(["OPENAI_API_KEY"])

# Build the vector database from raw job records
run_offline_chroma_pipeline(
    raw_job_records=job_records,
    db_path=str(VECTOR_DB_PATH),
    collection_name="jobs",
    batch_size=200,
    max_workers=8,
    max_professional_years=3,
    max_project_internship_years=3,
    no_internships=True,
)

print(f"Vector database ready in {VECTOR_DB_PATH}")

_report_execution_time(SCRIPT_START_TIME)
