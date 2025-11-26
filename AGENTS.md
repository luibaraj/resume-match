# Repository Guidelines

## Simple First Implementation Pholosophy

- Implement the simplest solution that fully solves the problem while preserving required behaviors (throttling, retries, normalization).
- Prefer clear, explicit code over abstraction or clever patterns. Add classes/helpers only when they reduce real complexity or duplication.
- Keep solutions simple but scalable: start with straightforward logic, but avoid designs that would block reasonable future growth (e.g., adding new job titles, outputs, or API variants).
- Avoid speculative generalization. Optimize or expand only when a real need appears.
- Maintain readability, testability, and mockability; simplicity must not hide side effects.
- When complexity is required (performance, robustness), document why.

## Project Structure & Modules

- `src/jsearch_client.py`: typed client for the OpenWeb Ninja JSearch API with auth, throttling, and retry/backoff logic.
- `src/job_scraper.py`: fetch/normalize helpers and JSON/CSV persistence; keeps rows flat and CSV-friendly.
- `src/multi_job_scraper.py`: orchestrates multiple job titles, default output `jobs.csv` at `DEFAULT_OUTPUT_PATH`, optional S3 upload.
- `tests/`: `test_*.py` suites mirroring modules, using `unittest` and `mock` for API/S3 isolation.
- `requirements.txt`, `setup.sh`, `env_vars.txt`: dependencies, venv bootstrap, and example env vars (do not commit real secrets).

## Setup, Run, and Test

- Bootstrap: `./setup.sh` (creates `.venv` and installs deps). Activate with `source .venv/bin/activate`.
- Local scrape: `python src/multi_job_scraper.py` (set `JSEARCH_API_KEY`; optionally `JOB_SCRAPER_S3_BUCKET`/`JOB_SCRAPER_S3_PREFIX` and custom `DEFAULT_OUTPUT_PATH`). Import helpers for ad-hoc scripts from `src`.
- Tests: `python -m unittest discover -s tests` or target a file, e.g., `python -m unittest tests.test_job_scraper`. Tests should run offline and only touch temp dirs.

## Coding Style & Naming

- PEP 8, 4-space indents, snake_case for files/functions, UpperCamelCase for classes; keep type hints and logging consistent.
- Preserve retry/rate-limit semantics in `JSearchClient` and avoid bypassing `_throttle`/`_sleep_with_backoff`.
- Use dependency injection for clients/sessions to keep code testable; prefer `pathlib.Path` for file IO.
- Keep defaults/constants near the top of modules; never embed secrets—read from env vars.

## Testing Expectations

- Add or extend `tests/test_*.py` alongside module changes; mock network and S3 to keep runs deterministic.
- Cover error paths (e.g., retries, permission errors, missing fields) and non-ASCII/duplicate handling where applicable.
- Ensure new helpers return shapes compatible with CSV/JSON writers and existing normalization.

## Commit & PR Guidelines

- Commits in this repo are short, imperative statements (e.g., `add logging`, `fix overload handling`); keep each focused.
- PRs should explain behavior changes, config/env vars touched, and include test evidence (`python -m unittest ...`).
- Call out API call volume or output schema impacts; avoid committing `.venv`, generated CSVs, or real keys/credentials.
