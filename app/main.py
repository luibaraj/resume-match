# main.py
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Resume-Match",
    description="Recieve the top Data Scientist, AI Engineer, ML Engineer, Data Analyst, and Research Engineer job postings that match the user's query.",
    version="1.0.0",
)

allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "*")
if allowed_origins_env.strip() == "*":
    cors_allow_credentials = False
    allowed_origins = ["*"]
else:
    cors_allow_credentials = True
    allowed_origins = [
        origin.strip()
        for origin in allowed_origins_env.split(",")
        if origin.strip()
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


from typing import List

from typing import List
from pydantic import BaseModel, Field, HttpUrl

from .src.normalize_helpers import clean_text_blob
from .src.online_helpers import find_reranked_jobs_for_profile


class ResumeMatchRequest(BaseModel):


    # Full resume content provided by the user as raw text
    # A minimum length is enforced to ensure sufficient data for matching
    resume_text: str = Field(
        min_length=50,
        description="Full resume content pasted by the user as plain text"
    )


class JobMatchDetail(BaseModel):

    # Unique identifier for the job posting (e.g., database ID or external job ID)
    job_id: str = Field(description="Unique identifier of the job posting")

    # Human-readable job title
    job_title: str = Field(description="Title of the job posting")

    # Name of the company offering the position
    company_name: str = Field(description="Company offering the position")

    # URL directing the user to the job application page
    application_url: HttpUrl = Field(
        description="Direct URL where the user can apply for the job"
    )

    # Overall match score between the resume and the job posting
    # Expressed as a percentage from 0 to 100
    match_score: float = Field(
        ge=0.0,
        le=100.0,
        description="Overall match score as a percentage"
    )

from typing import List
from pydantic import BaseModel, Field, HttpUrl


class ResumeMatchRequest(BaseModel):
    """
    Request payload sent by the UI when a user submits a resume
    for job matching.

    This model represents a single resume pasted into a text box
    on the front end. The resume is evaluated against multiple
    job postings on the server.
    """

    # Raw resume content provided by the user as plain text.
    # A minimum length is enforced to prevent empty or low-quality submissions.
    resume_text: str = Field(
        min_length=50,
        description="Full resume content pasted by the user as plain text"
    )


class JobMatchDetail(BaseModel):
    """
    Match evaluation results for a single job posting.

    Each instance of this model corresponds to one job and
    describes how well the submitted resume matches it.
    """

    # Internal or external identifier for the job posting
    job_id: str = Field(
        description="Unique identifier of the job posting"
    )

    # Human-readable title of the job
    job_title: str = Field(
        description="Title of the job posting"
    )

    # Name of the company offering the role
    company_name: str = Field(
        description="Company offering the position"
    )

    # URL directing the user to the job application page
    # Validated as a proper HTTP/HTTPS URL
    application_url: HttpUrl = Field(
        description="Direct URL where the user can apply for the job"
    )

    # Overall match score between the resume and the job
    # Expressed as a percentage from 0 to 100
    match_score: float = Field(
        ge=0.0,
        le=100.0,
        description="Overall match score as a percentage"
    )


class ResumeMatchResponse(BaseModel):
    """
    API response returned after evaluating a resume against
    multiple job postings.

    This object aggregates all job-level match results for
    a single resume submission.
    """

    # Total number of job postings that were evaluated
    total_jobs_evaluated: int = Field(
        ge=0,
        description="Number of job postings evaluated against the resume"
    )

    # Collection of per-job match results
    job_matches: List[JobMatchDetail] = Field(
        description="List of match results for each job posting"
    )


from concurrent.futures import ThreadPoolExecutor
from typing import Tuple


def _normalize_job(
    idx: int,
    job_payload: dict,
) -> Tuple[int, JobMatchDetail]:
    """
    Normalize a single reranked job payload into a JobMatchDetail.
    Returns the original index to preserve ordering.
    """

    # Safely extract metadata dictionary (may be missing or null)
    metadata = job_payload.get("metadata") or {}

    # Resolve a stable job identifier with fallbacks
    job_id = str(
        job_payload.get("job_id")
        or metadata.get("job_id")
    )

    # Resolve the job title with multiple possible metadata keys
    job_title = str(
        job_payload.get("job_title")
        or metadata.get("job_title")
    )

    # Resolve the company name with multiple possible metadata keys
    company_name = str(
        job_payload.get("employer_name")
        or metadata.get("employer_name")
    )

    # Resolve an application URL
    application_url = str(
        job_payload.get("job_apply_link")
        or metadata.get("job_apply_link")
    )

    # Extract the LLM-derived overall match score (default to 0.0)
    match_score = float(
        (job_payload.get("llm_score") or {}).get("overall_score") or 0.0
    )

    return (
        idx,
        JobMatchDetail(
            job_id=job_id,
            job_title=job_title,
            company_name=company_name,
            application_url=application_url,
            match_score=match_score,
        ),
    )


@app.post("/resume-match", response_model=ResumeMatchResponse)
def match_resume(request: ResumeMatchRequest) -> ResumeMatchResponse:
    """
    Accept a resume payload, normalize the resume text, run the job-matching
    and reranking pipeline, and return structured job match results.
    """

    # Normalize and clean the raw resume text provided by the client
    normalized_resume = clean_text_blob(request.resume_text)

    try:
        # Execute the core matching pipeline:
        # - retrieves candidate jobs from the database
        # - reranks them using an LLM-based scoring step
        matches_payload = find_reranked_jobs_for_profile(
            normalized_resume,
            db_path=os.getenv("VECTOR_DB_PATH"),
            api_key=os.getenv("OPENAI_API_KEY"),
            use_threads=True,
            max_workers=8,
            reranked_top_n=50,
            retrieve_top_k=150
        )

    # Allow explicitly raised HTTPExceptions to propagate unchanged
    except HTTPException:
        raise

    # Catch any unexpected errors and surface them as a 500 response
    # to avoid leaking internal stack traces to clients
    except Exception as exc:  # pragma: no cover - surface pipeline failures
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate resume matches: {exc}",
        ) from exc

    # Extract reranked jobs (final ranked list) from the pipeline output
    reranked_jobs = matches_payload.get("reranked_jobs", [])

    # Extract the originally retrieved jobs to report evaluation volume
    retrieved_jobs = matches_payload.get("retrieved_jobs", [])

    # Container for API-facing job match objects
    job_matches: List[JobMatchDetail] = []

    # Number of concurrent workers
    MAX_WORKERS = 4

    # Execute job normalization concurrently while preserving original order
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all reranked jobs for parallel normalization
        futures = [
            executor.submit(_normalize_job, idx, job_payload)
            for idx, job_payload in enumerate(reranked_jobs)
        ]

        # Collect results and reassemble in original sequence
        job_matches = [
            job_match
            for _, job_match in sorted(
                (future.result() for future in futures),
                key=lambda x: x[0],
            )
        ]

    # Return the final API response:
    # - total jobs evaluated (pre-reranking)
    # - normalized, ranked job matches
    return ResumeMatchResponse(
        total_jobs_evaluated=len(retrieved_jobs),
        job_matches=job_matches,
    )
