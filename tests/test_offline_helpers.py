"""Unit tests for src.offline_helpers helper functions."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

# Provide lightweight stubs for optional dependencies so the module imports cleanly.
if "openai" not in sys.modules:  # pragma: no cover - only exercised in tests
    class _StubOpenAI:
        def __init__(self, *args, **kwargs):
            pass

    sys.modules["openai"] = SimpleNamespace(OpenAI=_StubOpenAI)

if "chromadb" not in sys.modules:  # pragma: no cover - only exercised in tests
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

from src import offline_helpers


def test_create_chroma_client_creates_directory(tmp_path, monkeypatch):
    """Ensure a persistent client is initialized with the requested path."""

    captured: dict[str, object] = {}

    def fake_persistent_client(*, path: str, **kwargs):
        captured["path"] = path
        captured["kwargs"] = kwargs
        return SimpleNamespace(path=path, kwargs=kwargs)

    monkeypatch.setattr(offline_helpers.chromadb, "PersistentClient", fake_persistent_client)

    db_dir = tmp_path / "vector_store"
    client = offline_helpers.create_chroma_client(str(db_dir), tenant="acme")

    assert db_dir.exists(), "helper should create the vector store directory"
    assert client.path == str(db_dir)
    assert captured["kwargs"] == {"tenant": "acme"}


def test_get_or_create_jobs_collection_uses_client(monkeypatch):
    """Verify the helper uses the client's get_or_create_collection entry point."""

    calls: dict[str, object] = {}

    class FakeClient:
        def get_or_create_collection(self, *, name: str, metadata: dict[str, str]):
            calls["name"] = name
            calls["metadata"] = metadata
            return "collection"

    client = FakeClient()
    result = offline_helpers.get_or_create_jobs_collection(client, collection_name="custom")

    assert result == "collection"
    assert calls["name"] == "custom"
    assert calls["metadata"] == {"source": "offline_pipeline"}


def test_prepare_chroma_job_data_returns_aligned_lists():
    """Confirm IDs, embeddings, docs, and metadata stay aligned and complete."""

    jobs = [
        {
            "job_id": 1,
            "embedding": [0.1, 0.2],
            "document": "doc-one",
            "search_term": "python",
            "job_description": "desc",
            "job_apply_link": "apply",
            "job_publisher": "board",
            "employer_name": "Acme",
            "job_title": "Engineer",
            "extra": "ignored",
        },
        {
            "job_id": "2",
            "embedding": (0.3, 0.4),
            "job_document": "doc-two",
            "search_term": "ml",
            "job_description": "desc2",
            "job_apply_link": "apply2",
            "job_publisher": "board2",
            "employer_name": "Beta",
            "job_title": "Scientist",
        },
    ]

    ids, embeddings, documents, metadatas = offline_helpers.prepare_chroma_job_data(jobs)

    assert ids == ["1", "2"]
    assert embeddings == [[0.1, 0.2], [0.3, 0.4]]
    assert documents == ["doc-one", "doc-two"]
    assert metadatas == [
        {
            "search_term": "python",
            "job_description": "desc",
            "job_apply_link": "apply",
            "job_publisher": "board",
            "employer_name": "Acme",
            "job_title": "Engineer",
        },
        {
            "search_term": "ml",
            "job_description": "desc2",
            "job_apply_link": "apply2",
            "job_publisher": "board2",
            "employer_name": "Beta",
            "job_title": "Scientist",
        },
    ]


def test_prepare_chroma_job_data_skips_incomplete_records():
    """Records missing IDs, embeddings, or documents should be ignored."""

    jobs = [
        {"job_id": "valid", "embedding": [0.5], "document": "ready"},
        {"job_id": "missing-doc", "embedding": [0.1]},
        {"embedding": [0.2], "document": "no-id"},
        {"job_id": "missing-embedding", "document": "doc"},
    ]

    ids, embeddings, documents, metadatas = offline_helpers.prepare_chroma_job_data(jobs)

    assert ids == ["valid"]
    assert embeddings == [[0.5]]
    assert documents == ["ready"]
    assert metadatas == [{"search_term": None, "job_description": None, "job_apply_link": None, "job_publisher": None, "employer_name": None, "job_title": None}]


class _FakeCollection:
    """Simple stand-in that records collection.add calls."""

    def __init__(self):
        self.calls: list[dict[str, object]] = []

    def add(self, **kwargs):
        self.calls.append(kwargs)


def test_insert_jobs_into_collection_batches_requests():
    """Validate batching logic splits payloads correctly before insertion."""

    collection = _FakeCollection()
    ids = ["1", "2", "3"]
    embeddings = [[0.1], [0.2], [0.3]]
    documents = ["doc1", "doc2", "doc3"]
    metadatas = [{"m": 1}, {"m": 2}, {"m": 3}]

    offline_helpers.insert_jobs_into_collection(
        collection,
        ids,
        embeddings,
        documents,
        metadatas,
        batch_size=2,
    )

    assert len(collection.calls) == 2
    assert collection.calls[0]["ids"] == ["1", "2"]
    assert collection.calls[1]["ids"] == ["3"]


def test_insert_jobs_into_collection_validates_inputs():
    """Ensure mismatched inputs or invalid batch sizes raise helpful errors."""

    collection = _FakeCollection()

    with pytest.raises(ValueError):
        offline_helpers.insert_jobs_into_collection(
            collection,
            ["1"],
            [],
            [],
            [],
        )

    with pytest.raises(ValueError):
        offline_helpers.insert_jobs_into_collection(
            collection,
            [],
            [],
            [],
            [],
            batch_size=0,
        )


def test_insert_jobs_into_collection_noop_on_empty_payload():
    """When the payload is empty, no collection.add calls should be issued."""

    collection = _FakeCollection()

    offline_helpers.insert_jobs_into_collection(collection, [], [], [], [])

    assert collection.calls == []


def test_run_offline_chroma_pipeline_happy_path(monkeypatch):
    """Smoke test to ensure the pipeline wires helpers together as expected."""

    raw_jobs = [{"raw_id": "a"}, {"raw_id": "b"}]

    normalized_calls = []
    def fake_normalize(job):
        normalized_calls.append(job)
        return {"job_id": job["raw_id"], "search_term": "python"}

    documents_built = []
    def fake_build_document(job):
        documents_built.append(job)
        return f"doc-{job['job_id']}"

    embedding_client = object()
    selected_client = {}
    def fake_select_client(model_name, **kwargs):
        selected_client["model_name"] = model_name
        selected_client["kwargs"] = kwargs
        return embedding_client

    embedding_calls = []
    def fake_embed(document, client):
        embedding_calls.append((document, client))
        return [len(document)]

    prepared_payload = {
        "jobs": None,
        "ids": ["id-1", "id-2"],
        "embeddings": [[1.0], [2.0]],
        "documents": ["doc-a", "doc-b"],
        "metadatas": [{"meta": 1}, {"meta": 2}],
    }
    def fake_prepare(jobs):
        prepared_payload["jobs"] = list(jobs)
        return (
            prepared_payload["ids"],
            prepared_payload["embeddings"],
            prepared_payload["documents"],
            prepared_payload["metadatas"],
        )

    created_client = {}
    def fake_create_client(db_path, **kwargs):
        created_client["db_path"] = db_path
        created_client["kwargs"] = kwargs
        return "client"

    def fake_get_collection(client, collection_name):
        assert client == "client"
        created_client["collection_name"] = collection_name
        return "collection"

    insert_calls = {}
    def fake_insert(collection, ids, embeddings, documents, metadatas, batch_size):
        insert_calls["collection"] = collection
        insert_calls["ids"] = ids
        insert_calls["embeddings"] = embeddings
        insert_calls["documents"] = documents
        insert_calls["metadatas"] = metadatas
        insert_calls["batch_size"] = batch_size

    monkeypatch.setattr(offline_helpers, "normalize_job", fake_normalize)
    monkeypatch.setattr(offline_helpers, "build_job_document", fake_build_document)
    monkeypatch.setattr(offline_helpers, "select_embedding_client", fake_select_client)
    monkeypatch.setattr(offline_helpers, "embed_job_postings", fake_embed)
    monkeypatch.setattr(offline_helpers, "prepare_chroma_job_data", fake_prepare)
    monkeypatch.setattr(offline_helpers, "create_chroma_client", fake_create_client)
    monkeypatch.setattr(offline_helpers, "get_or_create_jobs_collection", fake_get_collection)
    monkeypatch.setattr(offline_helpers, "insert_jobs_into_collection", fake_insert)

    offline_helpers.run_offline_chroma_pipeline(
        raw_jobs,
        db_path="/tmp/db",
        collection_name="jobs",
        model_name="text-embedding-3-small",
        embedding_client_kwargs={"api_key": "test"},
        chroma_client_kwargs={"tenant": "acme"},
        batch_size=25,
    )

    assert normalized_calls == raw_jobs
    assert documents_built == [
        {"job_id": "a", "search_term": "python"},
        {"job_id": "b", "search_term": "python"},
    ]
    assert embedding_calls == [("doc-a", embedding_client), ("doc-b", embedding_client)]
    assert isinstance(prepared_payload["jobs"], list) and len(prepared_payload["jobs"]) == 2
    assert created_client == {
        "db_path": "/tmp/db",
        "kwargs": {"tenant": "acme"},
        "collection_name": "jobs",
    }
    assert insert_calls["collection"] == "collection"
    assert insert_calls["ids"] == prepared_payload["ids"]
    assert insert_calls["batch_size"] == 25


def test_run_offline_chroma_pipeline_exits_when_no_jobs(monkeypatch):
    """The pipeline should short-circuit before touching Chroma when nothing is prepared."""

    monkeypatch.setattr(offline_helpers, "normalize_job", lambda job: {"job_id": job["id"]})
    monkeypatch.setattr(offline_helpers, "build_job_document", lambda job: job["job_id"])
    monkeypatch.setattr(offline_helpers, "select_embedding_client", lambda **_: object())
    monkeypatch.setattr(offline_helpers, "embed_job_postings", lambda *args, **kwargs: [0.0])

    prepare_called = {"count": 0}
    def fake_prepare(jobs):
        prepare_called["count"] += 1
        return ([], [], [], [])
    monkeypatch.setattr(offline_helpers, "prepare_chroma_job_data", fake_prepare)

    def _fail(*args, **kwargs):
        raise AssertionError("should not reach Chroma steps when nothing to insert")

    monkeypatch.setattr(offline_helpers, "create_chroma_client", _fail)
    monkeypatch.setattr(offline_helpers, "get_or_create_jobs_collection", _fail)
    monkeypatch.setattr(offline_helpers, "insert_jobs_into_collection", _fail)

    offline_helpers.run_offline_chroma_pipeline(
        [{"id": "only"}],
        db_path="/tmp/db",
    )

    assert prepare_called["count"] == 1


def test_run_offline_chroma_pipeline_sequential_path(monkeypatch):
    """It should process every job in order when no workers are requested."""

    raw_jobs = [{"raw_id": "seq-1"}, {"raw_id": "seq-2"}]

    processed_jobs: list[str] = []

    # Pretend _process_job_record returns normalized payloads and track call order.
    def fake_process(raw_job, model_name, _provider):
        processed_jobs.append(raw_job["raw_id"])
        return {
            "job_id": raw_job["raw_id"],
            "document": f"doc-{raw_job['raw_id']}",
            "embedding": [1.0],
            "search_term": raw_job["raw_id"],
        }

    prepared_payload = {}

    # Capture the jobs handed to prepare_chroma_job_data to prove all records flow through.
    def fake_prepare(jobs):
        job_list = list(jobs)
        prepared_payload["jobs"] = job_list
        ids = [job["job_id"] for job in job_list]
        embeddings = [job["embedding"] for job in job_list]
        documents = [job["document"] for job in job_list]
        metadatas = [{"search_term": job["search_term"]} for job in job_list]
        return ids, embeddings, documents, metadatas

    created_client = {}
    def fake_create_client(db_path, **kwargs):
        created_client["db_path"] = db_path
        created_client["kwargs"] = kwargs
        return "client"

    def fake_get_collection(client, collection_name):
        created_client["collection_name"] = collection_name
        return "collection"

    insert_calls = []
    def fake_insert(collection, ids, embeddings, documents, metadatas, batch_size):
        insert_calls.append((collection, list(ids), batch_size))

    # Stubs for dependencies so this test isolates the control flow.
    monkeypatch.setattr(offline_helpers, "select_embedding_client", lambda **kwargs: object())
    monkeypatch.setattr(offline_helpers, "_process_job_record", fake_process)
    monkeypatch.setattr(offline_helpers, "prepare_chroma_job_data", fake_prepare)
    monkeypatch.setattr(offline_helpers, "create_chroma_client", fake_create_client)
    monkeypatch.setattr(offline_helpers, "get_or_create_jobs_collection", fake_get_collection)
    monkeypatch.setattr(offline_helpers, "insert_jobs_into_collection", fake_insert)

    offline_helpers.run_offline_chroma_pipeline(
        raw_jobs,
        db_path="/tmp/db",
    )

    assert processed_jobs == ["seq-1", "seq-2"]
    assert [job["job_id"] for job in prepared_payload["jobs"]] == processed_jobs
    assert insert_calls == [("collection", ["seq-1", "seq-2"], 500)]


def test_run_offline_chroma_pipeline_threaded_path(monkeypatch):
    """It should go through the executor branch when workers are requested."""

    raw_jobs = [{"raw_id": str(i)} for i in range(3)]
    processed_jobs: list[str] = []

    def fake_process(raw_job, model_name, _provider):
        processed_jobs.append(raw_job["raw_id"])
        return {
            "job_id": raw_job["raw_id"],
            "document": f"doc-{raw_job['raw_id']}",
            "embedding": [float(raw_job["raw_id"])],
            "search_term": raw_job["raw_id"],
        }

    prepared_payload = {}

    def fake_prepare(jobs):
        job_list = list(jobs)
        prepared_payload["jobs"] = job_list
        ids = [job["job_id"] for job in job_list]
        embeddings = [job["embedding"] for job in job_list]
        documents = [job["document"] for job in job_list]
        metadatas = [{"search_term": job["search_term"]} for job in job_list]
        return ids, embeddings, documents, metadatas

    insert_calls = []
    def fake_insert(collection, ids, embeddings, documents, metadatas, batch_size):
        insert_calls.append((collection, list(ids), batch_size))

    # Minimal collection plumbing to keep the focus on threading behavior.
    monkeypatch.setattr(offline_helpers, "select_embedding_client", lambda **kwargs: object())
    monkeypatch.setattr(offline_helpers, "prepare_chroma_job_data", fake_prepare)
    monkeypatch.setattr(offline_helpers, "create_chroma_client", lambda **kwargs: "client")
    monkeypatch.setattr(offline_helpers, "get_or_create_jobs_collection", lambda *args, **kwargs: "collection")
    monkeypatch.setattr(offline_helpers, "insert_jobs_into_collection", fake_insert)
    monkeypatch.setattr(offline_helpers, "_process_job_record", fake_process)

    executor_stats = {"submitted": 0, "max_workers": None}

    class DummyFuture:
        def __init__(self, fn, args):
            self._fn = fn
            self._args = args

        def result(self):
            return self._fn(*self._args)

    class RecordingExecutor:
        """Synchronous executor that records submissions for easy assertions."""

        def __init__(self, max_workers):
            executor_stats["max_workers"] = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def submit(self, fn, *args, **kwargs):
            executor_stats["submitted"] += 1
            return DummyFuture(fn, args)

    monkeypatch.setattr(offline_helpers, "ThreadPoolExecutor", RecordingExecutor)

    offline_helpers.run_offline_chroma_pipeline(
        raw_jobs,
        db_path="/tmp/db",
        max_workers=4,
    )

    assert executor_stats == {"submitted": len(raw_jobs), "max_workers": 4}
    assert sorted(processed_jobs) == sorted(job["raw_id"] for job in raw_jobs)
    assert [job["job_id"] for job in prepared_payload["jobs"]] == [job["raw_id"] for job in raw_jobs]
    assert insert_calls == [("collection", [job["raw_id"] for job in raw_jobs], 500)]
