export interface JobMatchDetail {
  job_id: string;
  job_title: string;
  company_name: string;
  application_url: string;
  match_score: number;
}

export interface ResumeMatchResponse {
  total_jobs_evaluated: number;
  job_matches: JobMatchDetail[];
}

const DEFAULT_BASE_URL = "http://127.0.0.1:8000";

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || DEFAULT_BASE_URL).replace(/\/$/, "");

const RESUME_ENDPOINT = "/resume-match";

/**
 * Submit raw resume text to the FastAPI service and return the structured response.
 */
export async function matchResume(resumeText: string): Promise<ResumeMatchResponse> {
  const response = await fetch(`${API_BASE_URL}${RESUME_ENDPOINT}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ resume_text: resumeText }),
  });

  if (!response.ok) {
    let errorMessage = `Request failed with status ${response.status}`;

    try {
      const payload = (await response.json()) as { detail?: string };
      if (payload.detail) {
        errorMessage = payload.detail;
      }
    } catch (_) {
      // Response body was not JSON; keep the default message.
    }

    throw new Error(errorMessage);
  }

  return (await response.json()) as ResumeMatchResponse;
}
