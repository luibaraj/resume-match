import os
import time

import requests





def scrape_jobs(query: str, country: str, start_page: int, num_pages: int = 10, max_attempts: int = 3, backoff_base_seconds: int=1) -> dict:
    """
    Call the JSearch API to fetch job search results in batches and aggregate the raw responses.

    Args:
        query: Job keywords passed directly to the API query parameter.
        start_page: Initial page value for the first request.
        num_pages: Value supplied to the numPages query parameter on each request.

    Returns:
         A dictionary with a single key "batches" containing the raw JSON payload from each
        successful request in the order they were retrieved.
    """
    # Load the required API token and initialize pagination state.
    api_key = os.environ["JSEARCH_API_KEY"]
    all_results = []
    current_page = start_page
    url = "https://api.openwebninja.com/jsearch/search"


    while True:
        # Each request repeats the same parameters except for the advancing page offset.
        params = {
            "page": current_page,
            "num_pages": num_pages,
            "query": query,
            "country": country, 
            "date_posted": "today",
        }
        headers = {
            "x-api-key": api_key,
            "Accept": "application/json",
        }


        # Retry loop guards each request to tolerate transient API or network issues.
        attempt = 1
        response = None
        while attempt <= max_attempts:
            try:
                # print("Attempt: ", attempt)
                candidate = requests.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=(5, 120),
                )
            except requests.RequestException as exc:
                candidate = None
                failure_status = "error"
                failure_reason = str(exc)
                failure_url = url
            else:
                if candidate.status_code == 200:
                    response = candidate
                    break
                failure_status = candidate.status_code
                failure_reason = candidate.reason
                failure_url = candidate.url
                response = candidate

            # Failed batches are reported and retried with exponential backoff.
            print(
                f"Request to {failure_url} failed with status code "
                f"{failure_status}: {failure_reason}"
            )
            if attempt == max_attempts:
                break
            # Exponential backoff to respect API limits and avoid hammering the server.
            backoff_seconds = backoff_base_seconds * (2 ** (attempt - 1))
            time.sleep(backoff_seconds)
            attempt += 1

        if not response or response.status_code != 200:
            # Skip ahead by the chunk size when retries exhausted or response invalid.
            current_page += num_pages
            continue

        parsed_json = response.json()
        # Persist the unfiltered API payload so the caller can handle downstream parsing.
        all_results.append(parsed_json)
        payload = parsed_json.get("data") if isinstance(parsed_json, dict) else parsed_json
        if not payload:
            # API returns empty payload when there are no more jobs to fetch.
            break

        current_page += num_pages
    
    return {"batches": all_results}
