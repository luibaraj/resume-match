import os
import requests


def scrape_jobs(query: str, start_page: int, num_pages: int) -> dict:
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
    url = "https://www.openwebninja.com/api/jsearch/search"

    while True:
        # Each request repeats the same parameters except for the advancing page.
        params = {
            "page": current_page,
            "numPages": num_pages,
            "query": query,
            "location": "US",
            "date_posted": "today",
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        }

        response = requests.get(url, params=params, headers=headers)
        if response.status_code != 200:
            # Failed batches are reported and skipped per spec.
            print(
                f"Request to {response.url} failed with status code "
                f"{response.status_code}: {response.reason}"
            )
            current_page += num_pages
            continue

        parsed_json = response.json()
        all_results.append(parsed_json)
        if isinstance(parsed_json, list) and not parsed_json:
            # Empty list marks the end of available data.
            break

        current_page += num_pages

    return {"batches": all_results}
