import pandas as pd

from .scraper import scrape_jobs


# Hardcoded search terms to iterate over when scraping.
SEARCH_TERMS = [
    "python",
    "statistics",
    "machine learning",
    "deep learning",
    "scikit-learn",
    "pytorch",
    "LLM",
    "Generative AI",
    "pandas, numpy",
    "data scientist",
    "research engineer",
    "machine learning Engineer",
    "AI Engineer",
    "data analyst",
]


def injest_jobs() -> pd.DataFrame:
    """
    Scrape jobs for the predefined search terms and return the raw results as a DataFrame.

    No processing or augmentation is applied; this simply aggregates the raw `data` arrays
    from each scrape_jobs response into a single DataFrame.
    """
    all_records = []
    for term in SEARCH_TERMS:
        scraped = scrape_jobs(term, country="us", start_page=1)
        batches = scraped.get("batches", []) if scraped else []
        for batch in batches:
            data = batch.get("data") if isinstance(batch, dict) else None
            if data:
                all_records.extend(data)

    return pd.DataFrame(all_records)
