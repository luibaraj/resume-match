import { FormEvent, useState } from "react";
import { ResumeMatchResponse, matchResume } from "./services/resumeService";

// Keep client-side validation in sync with the FastAPI model definition.
const MIN_RESUME_LENGTH = 50;

function App() {
  const [resumeText, setResumeText] = useState("");
  const [results, setResults] = useState<ResumeMatchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Submit the resume to the FastAPI endpoint and handle success/error states.
  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    const cleanedResume = resumeText.trim();
    if (cleanedResume.length < MIN_RESUME_LENGTH) {
      setError(`Please paste at least ${MIN_RESUME_LENGTH} characters of resume text.`);
      return;
    }

    setIsSubmitting(true);
    setError(null);

    try {
      const apiResponse = await matchResume(cleanedResume);
      setResults(apiResponse);
    } catch (submissionError) {
      const fallbackMessage =
        submissionError instanceof Error
          ? submissionError.message
          : "Unable to fetch job matches. Please try again.";
      setError(fallbackMessage);
    } finally {
      setIsSubmitting(false);
    }
  };

  // Cached value to simplify rendering logic.
  const jobMatches = results?.job_matches ?? [];

  return (
    <div className="app-shell">
      <header className="app-header">
        <h1>Resume Match</h1>
        <p>Paste your resume to discover the top ranked jobs returned by the FastAPI service.</p>
      </header>

      <main className="app-content">
        <section className="card">
          <form onSubmit={handleSubmit}>
            <label htmlFor="resume-text">Resume text</label>
            <textarea
              id="resume-text"
              name="resume-text"
              placeholder="Paste your resume here..."
              value={resumeText}
              onChange={(event) => setResumeText(event.target.value)}
              rows={12}
              required
            />

            <div className="helper-row">
              <span>{resumeText.trim().length} characters</span>
              <button type="submit" disabled={isSubmitting}>
                {isSubmitting ? "Matching..." : "Find matches"}
              </button>
            </div>
          </form>

          {error && <p className="status status-error">{error}</p>}
          {results && !error && (
            <p className="status status-success">
              Evaluated {results.total_jobs_evaluated} jobs and reranked {jobMatches.length} top matches.
            </p>
          )}
        </section>

        {results && jobMatches.length === 0 && (
          <section className="card">
            <h2>No reranked matches yet</h2>
            <p>
              The API evaluated {results.total_jobs_evaluated} jobs but none passed the reranking threshold.
              Try adding more detail to your resume or running the query again later.
            </p>
          </section>
        )}

        {jobMatches.length > 0 && (
          <section className="card">
            <h2>Top matches</h2>
            <ul className="job-list">
              {jobMatches.map((job) => (
                <li key={job.job_id} className="job-card">
                  <div>
                    <p className="job-score">Match score: {job.match_score.toFixed(1)}%</p>
                    <h3>{job.job_title}</h3>
                    <p className="job-company">{job.company_name}</p>
                  </div>
                  <a
                    href={job.application_url}
                    target="_blank"
                    rel="noreferrer"
                    className="job-link"
                  >
                    Apply
                  </a>
                </li>
              ))}
            </ul>
          </section>
        )}
      </main>

      <footer className="app-footer">
        API base: {import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000"}
      </footer>
    </div>
  );
}

export default App;
