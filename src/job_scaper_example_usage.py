"""
Minimal usage example for the JSearch job scraper.

Set JSEARCH_API_KEY in your environment before running:
  export JSEARCH_API_KEY="YOUR_KEY"

python src/example_usage.py \
  --query "data scientist" \
  --pages 1 \
  --per-page 1 \
  --country us \
  --language en \
  --remote-only \
  --output-json /path/to/jobs.json \
  --output-csv /path/to/jobs.csv \
  --verbose

python src/example_usage.py \
  --query "data scientist in united states" \
  --pages 1 \
  --per-page 1 \
  --country us \
  --output-csv /Users/luisbarajas/Desktop/Projects/jobs.csv



"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from jsearch_client import JSearchClient
from job_scraper import collect_and_normalize, save_csv, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch job listings via JSearch API.")
    parser.add_argument("--query", required=True, help="Search query, e.g. 'python developer in chicago'.")
    parser.add_argument("--pages", type=int, default=1, help="Number of pages to fetch (default: 1).")
    parser.add_argument(
        "--per-page",
        type=int,
        default=1,
        help="How many pages to request per API call using num_pages (default: 1).",
    )
    parser.add_argument("--country", default="us", help="Country code (default: us).")
    parser.add_argument("--language", default=None, help="Language code, optional.")
    parser.add_argument("--remote-only", action="store_true", help="Return only remote jobs.")
    parser.add_argument("--output-json", type=Path, help="Path to save JSON results.")
    parser.add_argument("--output-csv", type=Path, help="Path to save CSV results.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    client = JSearchClient()
    jobs = collect_and_normalize(
        client,
        query=args.query,
        pages=args.pages,
        per_page=args.per_page,
        country=args.country,
        language=args.language,
        work_from_home=args.remote_only,
    )
    logging.info("Fetched %s jobs", len(jobs))

    if args.output_json:
        save_json(args.output_json, jobs)
        logging.info("Saved JSON to %s", args.output_json)
    if args.output_csv:
        save_csv(args.output_csv, jobs)
        logging.info("Saved CSV to %s", args.output_csv)

    if not args.output_json and not args.output_csv:
        for job in jobs:
            print(f"{job['title']} at {job['employer']} — {job['location']}")


if __name__ == "__main__":
    main()
