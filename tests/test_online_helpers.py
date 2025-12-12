"""Unit tests for src.online_helpers helpers."""

from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace

import pytest


if "openai" not in sys.modules:  # pragma: no cover - exercised only when SDK missing
    class _StubOpenAI:
        def __init__(self, *args, **kwargs):
            pass

    sys.modules["openai"] = SimpleNamespace(OpenAI=_StubOpenAI)

if "chromadb" not in sys.modules:  # pragma: no cover - exercised only when SDK missing
    chromadb_module = ModuleType("chromadb")

    class _StubPersistentClient:
        def __init__(self, *args, **kwargs):
            pass

    chromadb_module.PersistentClient = _StubPersistentClient
    sys.modules["chromadb"] = chromadb_module

    chromadb_api = ModuleType("chromadb.api")
    chromadb_models = ModuleType("chromadb.api.models")
    chromadb_module.api = chromadb_api
    chromadb_api.models = chromadb_models
    sys.modules["chromadb.api"] = chromadb_api
    sys.modules["chromadb.api.models"] = chromadb_models

    chromadb_collection_module = ModuleType("chromadb.api.models.Collection")

    class _StubCollection:
        pass

    chromadb_collection_module.Collection = _StubCollection
    sys.modules["chromadb.api.models.Collection"] = chromadb_collection_module


from src import online_helpers


def test_normalize_profile_returns_expected_payload(monkeypatch):
    """Ensure the helper cleans text, parses JSON, and shapes the payload."""

    captured: dict[str, object] = {}

    def fake_clean(text: str) -> str:
        captured["raw_text"] = text
        return "CLEAN::" + text

    monkeypatch.setattr(online_helpers, "clean_text_blob", fake_clean)

    class FakeChatCompletions:
        def create(self, **kwargs):
            captured["prompt"] = kwargs["messages"][0]["content"]
            payload = {
                "project_internship_years": 1,
                "professional_years": 3,
                "core_skills": ["python"],
                "secondary_skills": ["sql"],
                "responsibilities": ["Built APIs"],
                "summary": "Seasoned engineer.",
            }
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))]
            )

    class FakeChat:
        def __init__(self):
            self.completions = FakeChatCompletions()

    class FakeClient:
        def __init__(self, **kwargs):
            captured["api_key"] = kwargs.get("api_key")
            self.chat = FakeChat()

    monkeypatch.setattr(online_helpers, "OpenAI", FakeClient)

    result = online_helpers.normalize_profile("raw >> text", profile_id="abc", api_key="KEY", retries=1)

    assert result["profile_id"] == "abc"
    assert result["profile_text"] == "CLEAN::raw >> text"
    assert result["project_internship_experience_years"] == 1
    assert result["professional_experience_years"] == 3
    assert result["required_skills"] == ["python"]
    assert result["preferred_skills"] == ["sql"]
    assert result["responsibilities"] == ["Built APIs"]
    assert result["profile_summary"] == "Seasoned engineer."
    assert captured["api_key"] == "KEY"
    assert "Profile Text" in captured["prompt"]


def test_build_profile_document_includes_summary(monkeypatch):
    """The profile document should reuse job formatting and prepend the summary."""

    job_payloads: list[dict[str, object]] = []

    def fake_build_job_document(payload: dict[str, object]) -> str:
        job_payloads.append(payload)
        return "JobDoc"

    monkeypatch.setattr(online_helpers, "build_job_document", fake_build_job_document)

    profile = {
        "project_internship_experience_years": 2,
        "professional_experience_years": 5,
        "required_skills": ["python"],
        "preferred_skills": ["sql"],
        "responsibilities": ["Owns systems"],
        "profile_summary": "Experienced developer.",
    }

    document = online_helpers.build_profile_document(profile)

    assert document.startswith("Candidate summary: Experienced developer.")
    assert "JobDoc" in document
    assert job_payloads[0]["project_internship_experience_years"] == 2
    assert job_payloads[0]["professional_experience_years"] == 5


def test_generate_profile_embedding_runs_full_pipeline(monkeypatch):
    """High-level orchestration should normalize, build docs, and embed."""

    normalized_profile = {"profile_id": "user-1"}

    def fake_normalize(*args, **kwargs):
        assert kwargs["profile_id"] == "user-1"
        return dict(normalized_profile)

    monkeypatch.setattr(online_helpers, "normalize_profile", fake_normalize)

    def fake_build_profile_document(profile):
        profile["called"] = True
        return "PROFILE DOC"

    monkeypatch.setattr(online_helpers, "build_profile_document", fake_build_profile_document)

    def fake_select(model_name: str, **kwargs):
        assert model_name == "text-embedding-3-small"
        assert kwargs == {"api_key": "API", "timeout": 30}
        return "CLIENT"

    monkeypatch.setattr(online_helpers, "select_embedding_client", fake_select)

    def fake_embed(document: str, client: object):
        assert document == "PROFILE DOC"
        assert client == "CLIENT"
        return [0.1, 0.2]

    monkeypatch.setattr(online_helpers, "embed_job_postings", fake_embed)

    result = online_helpers.generate_profile_embedding(
        "raw text",
        profile_id="user-1",
        api_key="API",
        embedding_client_kwargs={"timeout": 30},
        normalization_retries=2,
    )

    assert result["document"] == "PROFILE DOC"
    assert result["embedding"] == [0.1, 0.2]
    assert result["normalized_profile"]["profile_document"] == "PROFILE DOC"
    assert result["normalized_profile"]["called"] is True


def test_load_jobs_collection_returns_client_and_collection(monkeypatch):
    """Ensure the helper instantiates the client and fetches the jobs collection."""

    captured: dict[str, object] = {}

    class FakeCollection:
        pass

    class FakeClient:
        def __init__(self, path, **kwargs):
            captured["path"] = path
            captured["kwargs"] = kwargs

        def get_collection(self, name):
            captured["collection_name"] = name
            return FakeCollection()

    monkeypatch.setattr(online_helpers.chromadb, "PersistentClient", FakeClient)

    client, collection = online_helpers.load_jobs_collection(
        db_path="/tmp/jobs",
        collection_name="jobs",
        chroma_client_kwargs={"tenant": "alpha"},
    )

    assert isinstance(client, FakeClient)
    assert isinstance(collection, FakeCollection)
    assert captured["path"] == "/tmp/jobs"
    assert captured["collection_name"] == "jobs"
    assert captured["kwargs"] == {"tenant": "alpha"}


def test_load_jobs_collection_requires_path():
    """db_path is mandatory for loading the existing collection."""
    with pytest.raises(ValueError):
        online_helpers.load_jobs_collection(db_path="")


def test_query_jobs_collection_formats_matches(monkeypatch):
    """Collection queries should return flattened dictionaries."""

    captured: dict[str, object] = {}

    class FakeCollection:
        def query(self, **kwargs):
            captured.update(kwargs)
            return {
                "ids": [["1", "2"]],
                "documents": [["Doc1", "Doc2"]],
                "metadatas": [[{"job_title": "Engineer"}, {"job_title": "Analyst"}]],
                "distances": [[0.1, 0.3]],
            }

    results = online_helpers.query_jobs_collection(
        [0.1, 0.2],
        collection=FakeCollection(),
        top_k=2,
        include=["documents", "metadatas"],
    )

    assert captured["query_embeddings"] == [[0.1, 0.2]]
    assert captured["n_results"] == 2
    assert captured["include"] == ["documents", "metadatas"]
    assert results == [
        {
            "id": "1",
            "document": "Doc1",
            "metadata": {"job_title": "Engineer"},
            "distance": 0.1,
        },
        {
            "id": "2",
            "document": "Doc2",
            "metadata": {"job_title": "Analyst"},
            "distance": 0.3,
        },
    ]


def test_query_jobs_collection_loads_collection_when_missing(monkeypatch):
    """If a collection is not passed, the helper should load it from disk."""

    captured: dict[str, object] = {}

    class FakeCollection:
        def __init__(self):
            self.called = False

        def query(self, **kwargs):
            self.called = True
            return {"ids": [["x"]]}

    fake_collection = FakeCollection()

    def fake_load(**kwargs):
        captured.update(kwargs)
        return "CLIENT", fake_collection

    monkeypatch.setattr(online_helpers, "load_jobs_collection", fake_load)

    results = online_helpers.query_jobs_collection([0.5], db_path="/tmp/jobs", top_k=1)

    assert fake_collection.called is True
    assert captured["db_path"] == "/tmp/jobs"
    assert captured["collection_name"] == "jobs"
    assert results == [{"id": "x"}]


def test_query_jobs_collection_validates_inputs():
    """Basic validation for query vectors and top_k."""

    with pytest.raises(ValueError):
        online_helpers.query_jobs_collection([], collection=SimpleNamespace())

    fake_collection = SimpleNamespace(
        query=lambda **kwargs: {"ids": [["x"]]}
    )

    with pytest.raises(ValueError):
        online_helpers.query_jobs_collection([0.1], collection=fake_collection, top_k=0)
