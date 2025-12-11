"""Offline pipeline helpers for building the job vector index."""

from __future__ import annotations

from typing import Any, Iterable

JobRecord = dict[str, Any]
EmbeddingPayload = dict[str, Any]


def collect_and_normalize_jobs(job_records: Iterable[JobRecord]) -> list[JobRecord]:
    """
    Load raw job descriptions and standardize the fields needed for embeddings.

    Returns a list of dictionaries (one per job) that contain years of project and internship experience,
    years of professional work experience, required skills, 
    preferred skills, responsibilities, and metadata the index should store.
    """
    raise NotImplementedError


def select_embedding_client(
    model_name: str = "text-embedding-3-small", **client_kwargs: Any
) -> Any:
    """
    Create the embedding client that will be reused for every job posting.

    Keeps the model choice (`text-embedding-3-small`) and client configuration in one place
    so the rest of the offline pipeline can remain agnostic.
    """
    raise NotImplementedError


def build_job_document(job: JobRecord) -> str:
    """
    Concatenate the normalized job details into a single text block.

    This text combines years of experience, required/preferred skills, and responsibilities
    to match the format expected by the embedding model.
    """
    raise NotImplementedError


def embed_job_postings(
    jobs: Iterable[JobRecord], client: Any
) -> list[EmbeddingPayload]:
    """
    Iterate through normalized jobs, build their text documents, and embed them.

    The returned payloads should include the embedding vector plus any metadata needed for
    storage in the vector database/index.
    """
    raise NotImplementedError


def init_vector_index(index_name: str, dimension: int, **client_kwargs: Any) -> Any:
    """
    Connect to (or create) the vector database/index that will store job embeddings.

    Handles all Pinecone-like setup so the upsert step can focus only on sending vectors.
    """
    raise NotImplementedError


def upsert_job_embeddings(
    index: Any, embeddings: Iterable[EmbeddingPayload], batch_size: int = 100
) -> None:
    """
    Send the embedding payloads to the vector index, chunking the uploads when helpful.

    Batching keeps the offline job efficient when processing large corpora of positions.
    """
    raise NotImplementedError


def run_offline_embedding_pipeline(
    raw_job_records: Iterable[JobRecord],
    *,
    model_name: str = "text-embedding-3-small",
    index_name: str,
    index_dimension: int,
    embedding_client_kwargs: dict[str, Any] | None = None,
    index_client_kwargs: dict[str, Any] | None = None,
) -> None:
    """
    High-level entry point that glues the offline steps together.

    Collect raw jobs → normalize → embed → upsert into the vector index so the online
    retrieval stack can operate on a fully prepared job corpus.
    """
    raise NotImplementedError
