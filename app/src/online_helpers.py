"""Online helpers for Vector Retrieval + Re-ranking."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

import json
import os
import time

import chromadb
from chromadb.api.models.Collection import Collection
from openai import OpenAI

# Prefer local package imports; fall back to absolute when executed as a script.
try:
    from .normalize_helpers import clean_text_blob  # type: ignore
    from .offline_helpers import (  # type: ignore
        build_job_document,
        embed_job_postings,
        select_embedding_client,
    )
except ImportError:  # pragma: no cover - script execution fallback
    from src.normalize_helpers import clean_text_blob  # type: ignore
    from src.offline_helpers import (  # type: ignore
        build_job_document,
        embed_job_postings,
        select_embedding_client,
    )


def normalize_profile(
    profile_text: str,
    *,
    profile_id: str | None = None,
    api_key: str | None = None,
    retries: int = 5,
    model: str = "gpt-4o-mini",
) -> dict[str, Any]:
    """
    Normalize a raw profile/resume string using the same strategy as job postings.

    The function sanitizes the text, prompts the LLM for structured data, and returns
    a dict compatible with the downstream job helpers (skills, years, responsibilities).
    """

    normalized_text = clean_text_blob(profile_text or "")
    client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    prompt = (
        "You are extracting structured resume details.\n"
        "From the profile text provided, extract:\n"
        "1. Explicitly stated internship or project experience years.\n"
        "2. Explicitly stated professional work experience years.\n"
        "3. Core skills that the candidate is strongest at (list).\n"
        "4. Supporting/secondary skills that still appear (list).\n"
        "5. Responsibilities or achievements mentioned (list of short bullet sentences).\n"
        "Follow these rules:\n"
        "- Only use numbers or skills that are explicitly present in the text.\n"
        "- If internship/project years are missing, respond with 0.\n"
        "- If professional years are missing, respond with 0.\n"
        "- Always return lists for skills and responsibilities (they may be empty).\n"
        "- Optionally summarize the candidate in one concise sentence.\n\n"
        "Respond as JSON with keys project_internship_years, professional_years, "
        "core_skills, secondary_skills, responsibilities, summary.\n\n"
        f"Profile Text:\n{normalized_text}"
    )

    required_keys = [
        "project_internship_years",
        "professional_years",
        "core_skills",
        "secondary_skills",
        "responsibilities",
    ]

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "profile_requirements",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "project_internship_years": {"type": "number"},
                                "professional_years": {"type": "number"},
                                "core_skills": {"type": "array", "items": {"type": "string"}},
                                "secondary_skills": {"type": "array", "items": {"type": "string"}},
                                "responsibilities": {"type": "array", "items": {"type": "string"}},
                                "summary": {"type": "string"},
                            },
                            "required": required_keys,
                        },
                    },
                },
            )
        except Exception as exc:
            if attempt == retries - 1:
                raise exc
            time.sleep(2**attempt)
            continue

        try:
            raw_content = response.choices[0].message.content
            parsed = json.loads(raw_content)
        except Exception:
            if attempt == retries - 1:
                raise ValueError("Failed to parse JSON output from profile normalization model.")
            time.sleep(2**attempt)
            continue

        if not all(key in parsed for key in required_keys):
            if attempt == retries - 1:
                raise ValueError("Profile normalization model returned incomplete data.")
            time.sleep(2**attempt)
            continue

        return {
            "profile_id": profile_id,
            "profile_text": normalized_text,
            "project_internship_experience_years": parsed["project_internship_years"],
            "professional_experience_years": parsed["professional_years"],
            "required_skills": parsed["core_skills"],
            "preferred_skills": parsed["secondary_skills"],
            "responsibilities": parsed["responsibilities"],
            "profile_summary": parsed.get("summary"),
        }

    raise RuntimeError("Unexpected normalize_profile failure.")  # pragma: no cover - defensive


def build_profile_document(profile: dict[str, Any]) -> str:
    """
    Construct a document for the profile using the same layout as job postings.

    The resulting text mirrors job documents so embeddings live in the same space.
    """

    job_like_payload = {
        "project_internship_experience_years": profile.get(
            "project_internship_experience_years", 0
        ),
        "professional_experience_years": profile.get("professional_experience_years", 0),
        "required_skills": profile.get("required_skills") or [],
        "preferred_skills": profile.get("preferred_skills") or [],
        "responsibilities": profile.get("responsibilities") or [],
    }

    document = build_job_document(job_like_payload)
    summary = profile.get("profile_summary")

    if isinstance(summary, str) and summary.strip():
        document = f"Candidate summary: {summary.strip()}\n{document}"

    return document


def generate_profile_embedding(
    profile_text: str,
    *,
    profile_id: str | None = None,
    model_name: str = "text-embedding-3-small",
    api_key: str | None = None,
    embedding_client_kwargs: dict[str, Any] | None = None,
    normalization_retries: int = 5,
) -> dict[str, Any]:
    """
    Orchestrate the full profile → embedding pipeline for a single user document.

    Steps:
    1. Normalize the raw profile (clean text + structured extraction).
    2. Build a job-like document so embeddings match the job corpora format.
    3. Obtain the embedding client and embed the document.

    Returns a dict containing the embedding, profile document, and normalized payload.
    """

    normalized_profile = normalize_profile(
        profile_text,
        profile_id=profile_id,
        api_key=api_key,
        retries=normalization_retries,
    )

    document = build_profile_document(normalized_profile)
    normalized_profile["profile_document"] = document

    kwargs = dict(embedding_client_kwargs or {})
    if api_key and "api_key" not in kwargs:
        kwargs["api_key"] = api_key

    embedding_client = select_embedding_client(model_name=model_name, **kwargs)
    embedding = embed_job_postings(document, embedding_client)

    return {
        "profile_id": normalized_profile.get("profile_id"),
        "document": document,
        "embedding": embedding,
        "normalized_profile": normalized_profile,
    }


def load_jobs_collection(
    *,
    db_path: str,
    collection_name: str = "jobs",
    chroma_client_kwargs: dict[str, Any] | None = None,
) -> tuple[chromadb.PersistentClient, Collection]:
    """
    Step 1 of the online retrieval pipeline: load the Chroma DB + collection.
    """

    if not db_path:
        raise ValueError("db_path must be provided to load the jobs collection.")

    kwargs = dict(chroma_client_kwargs or {})
    client = chromadb.PersistentClient(path=db_path, **kwargs)

    try:
        collection = client.get_collection(collection_name)
    except Exception as exc:  # pragma: no cover - defensive guardrail
        raise RuntimeError(
            f"Failed to load Chroma collection '{collection_name}' from '{db_path}'."
        ) from exc

    return client, collection





def _first_sequence(value: Any) -> list[Any]:
    """Helper: unwrap the first list of results from a Chroma field."""

    if not value:
        return []
    if isinstance(value, list):
        if value and isinstance(value[0], list):
            return value[0]
        return value
    return []

def _format_chroma_query_results(query_payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten the Chroma response into a sequence of matches."""

    ids = _first_sequence(query_payload.get("ids"))
    documents = _first_sequence(query_payload.get("documents"))
    metadatas = _first_sequence(query_payload.get("metadatas"))
    distances = _first_sequence(query_payload.get("distances"))

    matches: list[dict[str, Any]] = []
    for idx, match_id in enumerate(ids):
        match: dict[str, Any] = {"id": match_id}

        if idx < len(documents):
            match["document"] = documents[idx]
        if idx < len(metadatas):
            match["metadata"] = metadatas[idx]
        if idx < len(distances):
            match["distance"] = distances[idx]

        matches.append(match)

    return matches


def query_jobs_collection(
    query_vector: list[float],
    *,
    db_path: str | None = None,
    collection: Collection | None = None,
    collection_name: str = "jobs",
    top_k: int = 100,
    chroma_client_kwargs: dict[str, Any] | None = None,
    include: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Complete the online retrieval pipeline against the jobs collection.
    """

    if not query_vector:
        raise ValueError("query_vector must be a non-empty embedding vector.")
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    if collection is None:
        if not db_path:
            raise ValueError("db_path is required when collection is not provided.")
        _, collection = load_jobs_collection(
            db_path=db_path,
            collection_name=collection_name,
            chroma_client_kwargs=chroma_client_kwargs,
        )

    include_fields = include[:] if include is not None else ["metadatas", "documents", "distances"]
    query_payload = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=include_fields,
    )

    return _format_chroma_query_results(query_payload)


def find_top_jobs_for_profile(
    profile_text: str,
    *,
    profile_id: str | None = None,
    db_path: str,
    top_k: int = 100,
    api_key: str | None = None,
    embedding_model: str = "text-embedding-3-small",
    chroma_client_kwargs: dict | None = None,
    embedding_client_kwargs: dict | None = None,
) -> list[dict]:
    """
    Full online pipeline: profile -> embedding -> top-k job matches.
    Returns a list of match dicts from `query_jobs_collection`.
    """

    # Profile → normalized payload → embedding vector
    embedding_payload = generate_profile_embedding(
        profile_text,
        profile_id=profile_id,
        api_key=api_key,
        model_name=embedding_model,
        embedding_client_kwargs=embedding_client_kwargs,
    )
    profile_embedding = embedding_payload["embedding"]

    # Reuse the same client+collection for multiple calls in real apps
    _, jobs_collection = load_jobs_collection(
        db_path=db_path,
        collection_name="jobs",
        chroma_client_kwargs=chroma_client_kwargs,
    )

    # Execute the vector search and return flattened matches
    return query_jobs_collection(
        profile_embedding,
        collection=jobs_collection,
        top_k=top_k,
    )

def score_profile_for_job_documents(
    profile_document: str,
    job_document: str,
    *,
    api_key: str | None = None,
    model: str = "gpt-4o-mini",
    retries: int = 3,
) -> dict[str, Any]:
    # Normalize whitespace so the model sees clean, comparable blocks of text.
    profile_doc = (profile_document or "").strip()
    job_doc = (job_document or "").strip()

    if not profile_doc or not job_doc:
        raise ValueError("Both profile_document and job_document must be non-empty strings.")

    client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
    required = [
        "overall_score",
        "skill_alignment",
        "experience_alignment",
        "responsibility_alignment",
        "rationale",
        "missing_critical_requirements",
    ]

    # Prompt stays compact; we rely on the standardized documents for context.
    prompt = (
        "Rate how well the candidate fits the job. Scores must be 0-100."
        " Return JSON with keys overall_score, skill_alignment, experience_alignment,"
        " responsibility_alignment, rationale, missing_critical_requirements.\n\n"
        f"Candidate Document:\n{profile_doc}\n\nJob Document:\n{job_doc}"
    )

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "profile_job_match",
                        "schema": {
                            "type": "object",
                            "properties": {
                                field: {"type": "number"} for field in required[:4]
                            } | {
                                "rationale": {"type": "string"},
                                "missing_critical_requirements": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": required,
                        },
                    },
                },
            )
            parsed = json.loads(response.choices[0].message.content)
            if not all(field in parsed for field in required):
                raise ValueError("Incomplete scoring payload.")
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)
            continue

        def clamp(value: Any) -> float:
            """Force every numeric score into the 0-100 range."""
            try:
                return max(0.0, min(100.0, float(value)))
            except (TypeError, ValueError):
                return 0.0

        missing = parsed.get("missing_critical_requirements") or []
        return {
            "overall_score": clamp(parsed.get("overall_score")),
            "skill_alignment": clamp(parsed.get("skill_alignment")),
            "experience_alignment": clamp(parsed.get("experience_alignment")),
            "responsibility_alignment": clamp(parsed.get("responsibility_alignment")),
            "rationale": str(parsed.get("rationale", "")).strip(),
            "missing_critical_requirements": [str(item).strip() for item in missing if str(item).strip()],
            "profile_document": profile_doc,
            "job_document": job_doc,
        }

    raise RuntimeError("Unexpected LLM scoring failure.")



def score_jobs_with_profile_document(
    profile_document: str,
    jobs: list[dict[str, Any]],
    *,
    api_key: str | None = None,
    model: str = "gpt-4o-mini",
    retries: int = 3,
    use_threads: bool = False,
    max_workers: int | None = None,
) -> list[dict[str, Any]]:
    """
    Score every retrieved job against the provided profile document using the LLM scorer.
    """
    # check if jobs is a list
    if not isinstance(jobs, list):
        raise TypeError("jobs must be the list returned by find_top_jobs_for_profile.")
    profile_doc = (profile_document or "").strip()
    # check if profile_doc exists
    if not profile_doc:
        raise ValueError("profile_document must be a non-empty string.")
    # check if jobs exist
    if not jobs:
        return []

    # define a function that returns the original job_payload with LLM scoring
    def _score(job_payload: dict[str, Any]) -> dict[str, Any]:
        job_doc = (job_payload.get("document") or job_payload.get("job_document") or "").strip()
        # check if job_doc exist
        if not job_doc:
            raise ValueError("Job payload missing its document text.")
        # recieve score from the re-ranking model (LLM)
        score = score_profile_for_job_documents(
            profile_doc,
            job_doc,
            api_key=api_key,
            model=model,
            retries=retries,
        )
        # create and return a new "enriched" job_payload with the added scores
        enriched = dict(job_payload)
        enriched["llm_score"] = score
        return enriched

    if not use_threads or len(jobs) == 1: # sequential scoring of jobs
        results: list[dict[str, Any]] = []
        for job in jobs:
            try:
                results.append(_score(job))
            except Exception as exc:
                failed = dict(job)
                failed["llm_scoring_error"] = str(exc)
                results.append(failed)
        return results

    workers = max_workers or min(8, len(jobs)) # parallelized scoring of jobs
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_job = {executor.submit(_score, job): job for job in jobs}
        for future in as_completed(future_to_job):
            job = future_to_job[future]
            try:
                results.append(future.result())
            except Exception as exc:
                failed = dict(job)
                failed["llm_scoring_error"] = str(exc)
                results.append(failed)

    return results


def find_reranked_jobs_for_profile(
    profile_text: str,
    *,
    db_path: str,
    profile_id: str | None = None,
    retrieve_top_k: int = 100,
    reranked_top_n: int = 20,
    api_key: str | None = None,
    embedding_model: str = "text-embedding-3-small",
    scoring_model: str = "gpt-4o-mini",
    scoring_retries: int = 3,
    use_threads: bool = False,
    max_workers: int | None = None,
    chroma_client_kwargs: dict[str, Any] | None = None,
    embedding_client_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Full pipeline: raw profile text -> vector retrieval -> LLM re-ranking -> top N jobs.

    Returns a dict with:
        profile_payload: normalized profile output (includes constructed document).
        retrieved_jobs: the initial top-k matches returned by vector search.
        reranked_jobs: the jobs sorted by LLM score limited to reranked_top_n.
    """

    if retrieve_top_k <= 0:
        raise ValueError("retrieve_top_k must be a positive integer.")
    if reranked_top_n <= 0:
        raise ValueError("reranked_top_n must be a positive integer.")

    # Step 1: normalize + embed the profile text
    embedding_payload = generate_profile_embedding(
        profile_text,
        profile_id=profile_id,
        api_key=api_key,
        model_name=embedding_model,
        embedding_client_kwargs=embedding_client_kwargs,
    )
    profile_document = embedding_payload["document"]
    profile_embedding = embedding_payload["embedding"]

    # Step 2: retrieve top-k jobs from the vector store
    _, jobs_collection = load_jobs_collection(
        db_path=db_path,
        collection_name="jobs",
        chroma_client_kwargs=chroma_client_kwargs,
    )
    retrieved_jobs = query_jobs_collection(
        profile_embedding,
        collection=jobs_collection,
        top_k=retrieve_top_k,
    )

    if not retrieved_jobs:
        return {
            "profile_payload": embedding_payload,
            "retrieved_jobs": [],
            "reranked_jobs": [],
        }

    # Step 3: score each job with the LLM and sort by overall score
    scored_jobs = score_jobs_with_profile_document(
        profile_document,
        retrieved_jobs,
        api_key=api_key,
        model=scoring_model,
        retries=scoring_retries,
        use_threads=use_threads,
        max_workers=max_workers,
    )
    scored_jobs.sort(
        key=lambda match: match.get("llm_score", {}).get("overall_score", -1.0),
        reverse=True,
    )

    return {
        "profile_payload": embedding_payload,
        "retrieved_jobs": retrieved_jobs,
        "reranked_jobs": scored_jobs[:reranked_top_n],
    }
