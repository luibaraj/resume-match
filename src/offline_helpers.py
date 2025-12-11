"""Offline pipeline helpers for building the job vector index."""

from __future__ import annotations

from typing import Any, Iterable

import json
import time
import os

from openai import OpenAI

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
