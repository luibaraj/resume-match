"""
Upload raw job records to AWS S3 using environment-supplied credentials.
"""

import datetime
import json
import os
import uuid
from typing import Optional, Tuple

import boto3
from botocore.exceptions import BotoCoreError, ClientError


def _load_env_config() -> Optional[dict]:
    """
    Load and validate required environment variables for S3 uploads.

    Returns:
        A dictionary of configuration values if all are present; otherwise None.
    """
    required_vars = (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
        "JOB_SCRAPER_S3_BUCKET",
        "JOB_SCRAPER_S3_PREFIX",
    )
    config = {}
    missing = []
    for var in required_vars:
        value = os.environ.get(var)
        if not value:
            missing.append(var)
        else:
            config[var] = value

    if missing:
        print(f"Missing required environment variables: {', '.join(missing)}")
        return None

    return config


def upload_job_records_to_s3(job_records: list) -> Tuple[bool, Optional[str]]:
    """
    Upload a list of raw job records to S3 as a JSON file.

    Args:
        job_records: Raw job entries to upload. No validation or transformation
            is performed.

    Returns:
        Tuple of (success flag, S3 URI or None on failure).
    """
    config = _load_env_config()
    if not config:
        return False, None

    try:
        payload = json.dumps(job_records)
    except (TypeError, ValueError) as exc:
        print(f"Failed to serialize job records: {exc}")
        return False, None

    date_str = datetime.datetime.now().strftime("%Y%m%d")
    suffix = uuid.uuid4().hex[:8]
    prefix = config["JOB_SCRAPER_S3_PREFIX"].strip("/")
    key = f"{date_str}-{suffix}.json"
    if prefix:
        key = f"{prefix}/{key}"

    client = boto3.client(
        "s3",
        aws_access_key_id=config["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=config["AWS_SECRET_ACCESS_KEY"],
        region_name=config["AWS_DEFAULT_REGION"],
    )

    attempts = 5
    for attempt in range(1, attempts + 1):
        try:
            client.put_object(
                Bucket=config["JOB_SCRAPER_S3_BUCKET"],
                Key=key,
                Body=payload.encode("utf-8"),
                ContentType="application/json",
            )
            uri = f"s3://{config['JOB_SCRAPER_S3_BUCKET']}/{key}"
            print(f"Uploaded {len(job_records)} job records to {uri}")
            return True, uri
        except (BotoCoreError, ClientError, Exception) as exc:
            print(f"Attempt {attempt} to upload to S3 failed: {exc}")

    return False, None
