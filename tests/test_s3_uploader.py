import types

import pytest
from botocore.exceptions import BotoCoreError

import src.s3_uploader as s3_uploader


class StubS3Client:
    def __init__(self, fail_times=0):
        self.fail_times = fail_times
        self.put_calls = []

    def put_object(self, Bucket=None, Key=None, Body=None, ContentType=None):
        self.put_calls.append(
            {
                "Bucket": Bucket,
                "Key": Key,
                "Body": Body,
                "ContentType": ContentType,
            }
        )
        if self.fail_times > 0:
            self.fail_times -= 1
            raise BotoCoreError("planned failure")


def _patch_datetime_and_uuid(monkeypatch):
    class FixedDateTime(s3_uploader.datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2024, 3, 20)

    monkeypatch.setattr(s3_uploader.datetime, "datetime", FixedDateTime)
    fake_uuid = types.SimpleNamespace(hex="abc12345deadbeef")
    monkeypatch.setattr(s3_uploader.uuid, "uuid4", lambda: fake_uuid)


def _set_env(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-2")
    monkeypatch.setenv("JOB_SCRAPER_S3_BUCKET", "resume-job-match-proj")
    monkeypatch.setenv("JOB_SCRAPER_S3_PREFIX", "datalake")


def test_upload_success(monkeypatch, capsys):
    _set_env(monkeypatch)
    _patch_datetime_and_uuid(monkeypatch)
    stub_client = StubS3Client()
    monkeypatch.setattr(s3_uploader.boto3, "client", lambda *_, **__: stub_client)

    success, uri = s3_uploader.upload_job_records_to_s3([{"id": 1}])
    output = capsys.readouterr().out

    assert success is True
    assert uri == "s3://resume-job-match-proj/datalake/20240320-abc12345.json"
    assert len(stub_client.put_calls) == 1
    assert "Uploaded 1 job records" in output


def test_missing_environment_variables(monkeypatch, capsys):
    _set_env(monkeypatch)
    monkeypatch.delenv("JOB_SCRAPER_S3_BUCKET")
    monkeypatch.setattr(
        s3_uploader.boto3, "client", lambda *_, **__: pytest.fail("Should not call S3")
    )

    success, uri = s3_uploader.upload_job_records_to_s3([])
    output = capsys.readouterr().out

    assert success is False
    assert uri is None
    assert "Missing required environment variables" in output


def test_serialization_failure(monkeypatch, capsys):
    _set_env(monkeypatch)
    monkeypatch.setattr(
        s3_uploader.boto3, "client", lambda *_, **__: pytest.fail("Should not call S3")
    )

    success, uri = s3_uploader.upload_job_records_to_s3([set([1, 2])])
    output = capsys.readouterr().out

    assert success is False
    assert uri is None
    assert "Failed to serialize job records" in output


def test_retries_then_succeeds(monkeypatch, capsys):
    _set_env(monkeypatch)
    _patch_datetime_and_uuid(monkeypatch)
    stub_client = StubS3Client(fail_times=2)
    monkeypatch.setattr(s3_uploader.boto3, "client", lambda *_, **__: stub_client)

    success, uri = s3_uploader.upload_job_records_to_s3([{"id": 1}, {"id": 2}])
    output = capsys.readouterr().out

    assert success is True
    assert uri == "s3://resume-job-match-proj/datalake/20240320-abc12345.json"
    assert len(stub_client.put_calls) == 3
    assert output.count("Attempt") == 2
    assert "Uploaded 2 job records" in output


def test_retries_exhausted(monkeypatch, capsys):
    _set_env(monkeypatch)
    stub_client = StubS3Client(fail_times=10)
    monkeypatch.setattr(s3_uploader.boto3, "client", lambda *_, **__: stub_client)

    success, uri = s3_uploader.upload_job_records_to_s3([{"id": 1}])
    output = capsys.readouterr().out

    assert success is False
    assert uri is None
    assert output.count("Attempt") == 5
