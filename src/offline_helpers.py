"""Offline pipeline helpers for building the job vector index."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterable

import json
import threading
import time
import os
import re

from openai import OpenAI
import chromadb
from chromadb.api.models.Collection import Collection

from .normalize_helpers import clean_text_blob



def normalize_job(job_record: dict[str, Any], api_key: str = os.getenv("OPENAI_API_KEY"), retries: int = 5) -> dict[str, Any]:
    _client = OpenAI(api_key=api_key)
    # normalize text
    description = clean_text_blob(job_record.get("job_description", ""))

    # LLM-extraction prompt
    prompt = (
        "Extract the following from the job description:\n"
        "1. Years of project/internship experience required.\n"
        "2. Years of professional experience required.\n"
        "3. Required skills (list).\n"
        "4. Preferred skills (list).\n"
        "5. Responsibilities (list).\n"
        "Respond as JSON with keys project_internship_years, professional_years, "
        "required_skills, preferred_skills, responsibilities.\n\n"
        "Rules:\n"
        "- Do not infer or guess any requirement that is not explicitly stated.\n"
        "- If project/internship years are not explicitly stated, assign 0.\n"
        "- If professional years are not explicitly stated, assign 0.\n"
        "- If professional years are not explicitly stated but the job title indicates high seniority (for example: principal, senior, founding), assign 10.\n"
        "- If required skills are not explicitly stated, assign an empty list.\n"
        "- If preferred skills are not explicitly stated, assign an empty list.\n"
        "- If responsibilities are not explicitly stated, assign an empty list.\n\n"
        f"Job Title: {job_record.get('job_title', 'Unknown')}\n"
        f"Description:\n{description}"
    )

    # required fields to be extracted from each job posting
    required_keys = [
        "project_internship_years",
        "professional_years",
        "required_skills",
        "preferred_skills",
        "responsibilities",
    ]

    # LLM-extraction + error handling
    for attempt in range(retries):
        try:
            response = _client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "job_requirements",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "project_internship_years": {"type": "number"},
                                "professional_years": {"type": "number"},
                                "required_skills": {"type": "array", "items": {"type": "string"}},
                                "preferred_skills": {"type": "array", "items": {"type": "string"}},
                                "responsibilities": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": required_keys,
                        },
                    },
                },
            )
        except Exception as e: # ERROR: API failed
            if attempt == retries - 1:
                raise e
            time.sleep(2 ** attempt)
            continue

        # Parse JSON
        try:
            raw = response.choices[0].message.content
            parsed = json.loads(raw)
        except Exception: # ERROR: json is malformed
            if attempt == retries - 1:
                raise ValueError("Failed to parse JSON output from model.")
            time.sleep(2 ** attempt)
            continue

        # Validate
        if not all(k in parsed for k in required_keys):
            if attempt == retries - 1: # ERROR: invalid schema
                raise ValueError("Model returned incomplete structured output.")
            time.sleep(2 ** attempt)
            continue

        # SUCCESS: return immediately
        return {
            "job_id": job_record.get("job_id"),
            "job_title": job_record.get("job_title"),
            "employer_name": job_record.get("employer_name"),
            "job_publisher": job_record.get("job_publisher"),
            "job_apply_link": job_record.get("job_apply_link"),
            "job_description": description,
            "search_term": job_record.get("search_term"),
            "project_internship_experience_years": parsed["project_internship_years"],
            "professional_experience_years": parsed["professional_years"],
            "required_skills": parsed["required_skills"],
            "preferred_skills": parsed["preferred_skills"],
            "responsibilities": parsed["responsibilities"],
        }

    # Should never reach here due to returns and raises
    raise RuntimeError("Unexpected normalize_job failure.")


def select_embedding_client(
    model_name: str = "text-embedding-3-small", **client_kwargs: Any
) -> Any:
    """
    Create the embedding client that will be reused for every job posting.

    Keeps the model choice (`text-embedding-3-small`) and client configuration in one place
    so the rest of the offline pipeline can remain agnostic.
    """

    # Create and return a lightweight client object that callers will reuse.
    return OpenAI(**client_kwargs)


def build_job_document(job: dict) -> str:
    """
    Concatenate the normalized job details into a single text block.

    This text combines years of experience, required/preferred skills, and responsibilities
    to match the format expected by the embedding model.
    """
    internship_years = job.get("project_internship_experience_years", 0)
    professional_years = job.get("professional_experience_years", 0)

    required_skills = job.get("required_skills") or []
    preferred_skills = job.get("preferred_skills") or []
    responsibilities = job.get("responsibilities") or []

    def _stringify(values: Any, separator: str) -> str:
        if not values:
            return "None specified"
        if isinstance(values, str):
            return values.strip()
        try:
            return separator.join(str(item).strip() for item in values if str(item).strip())
        except TypeError:
            return str(values).strip()

    required_skills_str = _stringify(required_skills, ", ")
    preferred_skills_str = _stringify(preferred_skills, ", ")
    responsibilities_str = _stringify(responsibilities, " ")

    return (
        f"Internship and project experience required: {internship_years} years.\n"
        f"Professional work experience required: {professional_years} years.\n"
        f"Required skills: {required_skills_str}.\n"
        f"Preferred skills: {preferred_skills_str}.\n"
        f"Responsibilities: {responsibilities_str}."
    )


def embed_job_postings(job_document: str, client: Any):
    """
    Embed the job_document

    The returned result should include the embedding vector
    """
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=job_document or "",
    )
    return response.data[0].embedding





def create_chroma_client(
    db_path: str="/Users/luisbarajas/Desktop/Projects/resume-match/job_vector_db",
    **client_kwargs: Any,
) -> chromadb.PersistentClient:
    """
    Create a Chroma client bound to a local directory on disk.

    The returned client controls a persistent vector store rooted at `db_path`.
    Deleting this directory is enough to fully dispose of the underlying index.
    """
    os.makedirs(db_path, exist_ok=True)
    return chromadb.PersistentClient(path=db_path, **client_kwargs)


def get_or_create_jobs_collection(
    client: chromadb.PersistentClient,
    collection_name: str = "jobs",
) -> Collection:
    """
    Retrieve the collection used to store job postings, creating it if needed.

    Keeping the collection lookup in one place makes it easy to change naming
    or configuration later without touching the rest of the offline pipeline.
    """
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"source": "offline_pipeline"},
    )


def prepare_chroma_job_data(
    jobs: Iterable[dict[str, Any]],
) -> tuple[list[str], list[list[float]], list[str], list[dict[str, Any]]]:
    """
    Transform normalized job records into aligned lists for Chroma insertion.

    Extracts job IDs (job_id), embedding vectors, concatenated job documents, and metadata (search_term, job_description, job_apply_link, job_publisher, employer_name, job_title)
    so they can be passed directly to `collection.add(...)` in a single call.
    """
    ids: list[str] = []
    embeddings: list[list[float]] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    metadata_fields = [
        "search_term",
        "job_description",
        "job_apply_link",
        "job_publisher",
        "employer_name",
        "job_title",
    ]

    for job in jobs:
        job_id = job.get("job_id")
        embedding = job.get("embedding")
        document = job.get("document") or job.get("job_document")

        if job_id is None or embedding is None or document is None:
            continue

        str_job_id = str(job_id)
        if str_job_id in seen_ids:
            continue

        seen_ids.add(str_job_id)
        ids.append(str_job_id)
        embeddings.append(list(embedding))
        documents.append(str(document))

        metadata = {field: job.get(field) for field in metadata_fields}
        metadatas.append(metadata)

    return ids, embeddings, documents, metadatas


def insert_jobs_into_collection(
    collection: Collection,
    ids: list[str],
    embeddings: list[list[float]],
    documents: list[str],
    metadatas: list[dict[str, Any]],
    *,
    batch_size: int = 500,
) -> None:
    """
    Bulk-insert job postings into the Chroma collection in fixed-size batches.

    Splits the aligned id/embedding/document/metadata lists into chunks and calls
    `collection.add(...)` repeatedly so large corpora can be indexed safely.
    """
    total = len(ids)

    if not (len(embeddings) == total == len(documents) == len(metadatas)):
        raise ValueError("IDs, embeddings, documents, and metadatas must have the same length.")
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if total == 0:
        return

    for start in range(0, total, batch_size):
        end = start + batch_size
        collection.add(
            ids=ids[start:end],
            embeddings=embeddings[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )


def _create_embedding_client_provider(
    client_factory: Callable[[], Any],
    *,
    use_thread_local: bool,
) -> Callable[[], Any]:
    """
    Definition:
    Returns a callable that supplies embedding clients.
    If use_thread_local is False, a single shared client instance is reused for all calls.
    If True, each thread receives its own client instance stored in thread-local state.

    Why:
    Some clients are not thread-safe or become bottlenecks when shared across threads.
    Using a separate client per thread avoids state conflicts and improves concurrent performance.
    """
    if not use_thread_local:
        client = client_factory()
        return lambda: client

    local_state = threading.local()

    def _provider() -> Any:
        client = getattr(local_state, "embedding_client", None)
        if client is None:
            client = client_factory()
            local_state.embedding_client = client
        return client

    return _provider


def _process_job_record(
    raw_job: dict[str, Any],
    model_name: str,
    embedding_client_provider: Callable[[], Any],
    max_project_internship_years: float | None = None,
    max_professional_years: float | None = None,
    no_internships: bool = True
) -> dict[str, Any] | None:
    """Normalize a job, apply experience filters, build its document, and attach its embedding."""
    normalized_job = normalize_job(raw_job)

    internship_years = normalized_job.get("project_internship_experience_years") or 0
    professional_years = normalized_job.get("professional_experience_years") or 0

    if max_project_internship_years is not None and internship_years >= max_project_internship_years:
        return None
    if max_professional_years is not None and professional_years >= max_professional_years:
        return None

    INTERN_TITLE_PATTERN = re.compile(r"\b(intern(ship)?|co-?op)\b", re.IGNORECASE)
    title = (normalized_job.get("job_title") or "")
    if no_internships and INTERN_TITLE_PATTERN.search(title):
        return None

    document = build_job_document(normalized_job)
    embedding_client = embedding_client_provider()

    if model_name == "text-embedding-3-small":
        embedding = embed_job_postings(document, embedding_client)
    else:
        response = embedding_client.embeddings.create(model=model_name, input=document or "")
        embedding = response.data[0].embedding

    job_payload = dict(normalized_job)
    job_payload["document"] = document
    job_payload["embedding"] = embedding
    return job_payload




def run_offline_chroma_pipeline(
    raw_job_records: Iterable[dict[str, Any]],
    *,
    db_path: str,
    collection_name: str = "jobs",
    model_name: str = "text-embedding-3-small",
    max_project_internship_years: float | None = None,
    max_professional_years: float | None = None,
    no_internships: bool=True,
    embedding_client_kwargs: dict[str, Any] | None = None,
    chroma_client_kwargs: dict[str, Any] | None = None,
    batch_size: int = 500,
    max_workers: int | None = None,
) -> None:
    """
    High-level entry point that glues the offline Chroma steps together.

    Collect raw jobs → normalize → build documents → embed → prepare Chroma payloads
    → create the local Chroma collection → insert all job vectors so the online
    retrieval stack can query a fully prepared job corpus. Set `max_workers` > 1 to
    process job records concurrently.

    Use `max_project_internship_years` and `max_professional_years` to skip jobs whose
    required experience is at or above the provided threshold(s).
    """
    # collect embedding and chroma clients
    embedding_client_kwargs = embedding_client_kwargs or {}
    chroma_client_kwargs = chroma_client_kwargs or {}

    # create client
    client_factory = lambda: select_embedding_client(model_name=model_name, **embedding_client_kwargs)

    # determine if running jobs in parallel and create the corresponding client provider
    run_in_parallel = isinstance(max_workers, int) and max_workers > 1
    worker_count = max_workers if run_in_parallel else 1
    embedding_client_provider = _create_embedding_client_provider(
        client_factory,
        use_thread_local=run_in_parallel,
    )
    jobs_with_embeddings: list[dict[str, Any]] = []

    if run_in_parallel:
        # Run jobs through the pipeline in parallel
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    _process_job_record,
                    raw_job,
                    model_name,
                    embedding_client_provider,
                    max_project_internship_years,
                    max_professional_years,
                    no_internships,
                )
                for raw_job in raw_job_records
            ]
            for future in futures:
                result = future.result()
                if result is not None:
                    jobs_with_embeddings.append(result)
    else:
        # Run jobs through the pipeline sequentially
        for raw_job in raw_job_records:
            result = _process_job_record(
                raw_job,
                model_name,
                embedding_client_provider,
                max_project_internship_years,
                max_professional_years,
                no_internships,
            )
            if result is not None:
                jobs_with_embeddings.append(result)

    # prepare data to store in the vector database
    ids, embeddings, documents, metadatas = prepare_chroma_job_data(jobs_with_embeddings)
    if not ids:
        return

    # create chroma client and jobs collection
    client = create_chroma_client(db_path=db_path, **chroma_client_kwargs)
    collection = get_or_create_jobs_collection(client, collection_name=collection_name)

    # add the jobs into the collection
    insert_jobs_into_collection(
        collection,
        ids,
        embeddings,
        documents,
        metadatas,
        batch_size=batch_size,
    )
