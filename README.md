# Resume Match

## Project Summary

Resume Match centers on a two-phase retrieval system that blends fast vector search with high-precision LLM reranking. In the offline phase, job postings are collected, cleaned, and normalized into consistent text fields, then embedded and stored in a local vector database alongside essential metadata (job title, company, apply link, identifiers). This precomputation makes retrieval efficient and keeps query latency low. In the online phase, a user-submitted resume is similarly normalized and embedded, then used to run a nearest-neighbor vector search to fetch a small set of top candidate jobs (top‑K) that are semantically similar. Those candidates are then passed to an LLM reranker, which reads the resume and each job description to produce a more faithful relevance ordering and an interpretable match score. The API returns the reranked, metadata-rich results for display in the UI.

## Retrieval System Architecture

The retrieval system is organized into two phases: an offline phase that prepares job postings ahead of time, and an online phase that runs when a user submits a resume. This keeps online requests fast while preserving quality.

Offline, the system collects job postings and converts each one into a consistent record. It cleans the text (removing boilerplate and standardizing fields) and stores the details needed to show results later, including job id, title, company, and application link. It then produces an embedding for every job description (a compact numeric representation of meaning). These embeddings are stored in a local vector database along with the job metadata and original job text, so the system can quickly find the closest items to a new query.

Online, the frontend sends the user’s resume text to the FastAPI endpoint. The backend applies the same cleanup rules used offline so resumes and job posts are comparable. It generates an embedding for the resume and performs a vector search against the local database to pull back a shortlist of candidates (for example, the top 20–50). This first step is fast and broad: it reduces a large set of postings to a small set that is likely relevant.

Next, the system runs an LLM reranking pass over only the shortlist. The LLM reads the resume together with each candidate job post and reorders them based on fit: required skills and tools, experience level, and domain focus. This second pass helps avoid false positives where a posting looks similar on the surface but is not a practical match. The reranker also produces a match score (and optionally brief reasons) for each job. The output shape is consistent so the UI can sort, filter, and link users straight to applications. Finally, the API returns the reranked list plus metadata (title, company, link, job id, score) for the UI to display.

### Diagrams

![Offline phase architecture](source/offline.png)

![Online phase architecture](source/online.png)

## Conceptual Guide: Embeddings, Vector Search, and LLM Reranking

This system uses a “fast then careful” approach. The fast part turns text into numbers and finds the closest matches. The careful part reads the text and makes a more human-like judgment. Here is what each step is doing at a conceptual level.

### 1) Creating embeddings (turning text into a meaning vector)

An embedding is a list of numbers that represents the meaning of a piece of text. You can think of it as placing the text at a point in a very large coordinate system. If two texts talk about similar things (for example, “build ML pipelines in Python” and “deploy machine learning workflows with Python”), their points tend to land near each other, even if they do not share the exact same words.

Mathematically, an embedding is a vector, usually written as:

```text
e = [e₁, e₂, e₃, …, e_d]
```

where d might be hundreds or thousands of numbers. Each “dimension” is not a human-labeled concept like “Python skill”; it is a learned feature. The embedding model is trained so that similar meanings produce vectors that are close together.

To compare two embeddings (for example, resume `r` and job `j`), the system uses a similarity measure. A common choice is cosine similarity:

```text
cos(r, j) = (r · j) / (||r|| · ||j||)
```

where `r · j` is the dot product (multiply matching coordinates and add them up), and `||r||` is the vector length. Cosine similarity ranges from -1 to 1, and higher values mean “more similar direction,” which loosely corresponds to “more similar meaning.” Many systems first scale vectors to length 1 (normalize them), in which case the dot product `r · j` equals cosine similarity.

### 2) Running vector search (finding the closest jobs quickly)

Once every job has an embedding saved offline, and a resume embedding is created online, the system needs to find which jobs are closest to the resume. The simplest method would be to compute cos(r, j) for every job j and pick the best few. That works for small collections, but it becomes slow as the number of jobs grows.

A vector database speeds this up by using an index that avoids checking every job. Conceptually, it groups “nearby” vectors and navigates the space to quickly reach good candidates. Different databases use different strategies (for example, building a graph of neighbors), but the goal is the same: return the top‑K jobs with the highest similarity scores. This is why the output of the first stage is a shortlist (such as 20–50 jobs): it is fast, and it keeps later steps affordable.

You may also see distance instead of similarity. For normalized vectors, cosine similarity and Euclidean distance are tightly related:

```text
||r − j||² = 2 − 2 · cos(r, j)
```

So “small distance” and “high cosine similarity” produce nearly the same ordering.

### 3) Applying LLM reranking (making a careful, context-aware choice)

Vector search is good at finding “roughly about the same topic,” but it can miss important details. For example, a resume and a job might both mention “data science,” yet the job requires production deployment and the resume is purely academic. This is where reranking helps.

In reranking, the system asks a language model to read the resume and each shortlisted job post and produce a score and an ordering. You can think of it as a scoring function:

```text
score(resume, job) → 0..100
```

Unlike vector search, this step can weigh specific requirements and context: years of experience, must-have tools, domain constraints, seniority level, and what the candidate has actually done (not just what words appear). The reranker can also be asked to explain its score briefly (for example, “strong Python/SQL match, missing cloud deployment experience”), which makes results easier to trust and act on.

Putting it together: embeddings + vector search quickly find the “most likely” matches, and LLM reranking refines those matches into the final ranked list. This two-step design is what makes the system both responsive and accurate.

## Startup Guide

**Prerequisites**

- Python 3.10+ with `pip` and (optionally) `python -m venv`
- Node.js 18+ and npm 9+
- Accounts + API keys for OpenAI (LLM + embeddings) and JSearch, plus AWS credentials with permission to upload to the target S3 bucket

### 1. Backend environment

```bash
cd app
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Create a root-level `.env.local` (or copy `app/env.example` to `app/.env.local`) and fill in the secrets the pipeline expects:

```bash
OPENAI_API_KEY=your-openai-api-key
JSEARCH_API_KEY=your-jsearch-api-key
AWS_ACCESS_KEY_ID=your-aws-access-key-id
AWS_SECRET_ACCESS_KEY=your-aws-secret-access-key
AWS_DEFAULT_REGION=us-east-1
JOB_SCRAPER_S3_BUCKET=your-bucket
JOB_SCRAPER_S3_PREFIX=datalake
VECTOR_DB_PATH=/absolute/path/to/app/vector_db
ALLOWED_ORIGINS=http://localhost:5173
```

### 2. Build the vector database (offline step)

This scrapes fresh jobs, normalizes them with the LLM, uploads a copy to S3, and stores embeddings locally. The script may run for several minutes and consumes OpenAI/JSearch quota.

```bash
cd app
python -m src.build_vector_db
```

After it finishes, confirm that the folder defined by `VECTOR_DB_PATH` now contains a Chroma DB (e.g., `chroma.sqlite3` and `index/`).

### 3. Launch the FastAPI service

```bash
cd app
source .venv/bin/activate
uvicorn app.main:app --reload --port 8000
```

The API will look for the vector DB using `VECTOR_DB_PATH` and expose `http://127.0.0.1:8000/resume-match`.

### 4. Run the frontend

```bash
cd frontend
npm install
echo "VITE_API_BASE_URL=http://127.0.0.1:8000" > .env   # optional override
npm run dev
```

Open the printed Vite URL (default `http://localhost:5173`) and submit a resume. Keep both the backend and frontend terminals running for a complete workflow.
