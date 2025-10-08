import requests
import json
import os
import logging
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Optional

# Try loading API key from environment first, then from an untracked secrets file.
try:
    from secrets import API_KEY as SECRETS_API_KEY  # secrets.py should be gitignored and created locally
except Exception:
    SECRETS_API_KEY = None

API_KEY = os.getenv('SERPAPI_API_KEY') or SECRETS_API_KEY
if not API_KEY:
    raise RuntimeError(
        "API key not found. Set SERPAPI_API_KEY env var or create a local secrets.py with API_KEY = '...'"
    )

API_URL = 'https://serpapi.com/search'
CACHE_DIR = Path(os.getenv('CACHE_DIR', './cache'))
MAX_RETRIES = int(os.getenv('MAX_RETRIES', '3'))
RETRY_BACKOFF = int(os.getenv('RETRY_BACKOFF', '2'))  # base for exponential backoff

LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)

SESSION = requests.Session()


def make_cache_filename(query: str, page_token: Optional[str] = None) -> str:
    """
    Create a filesystem-safe cache filename by hashing the query + page_token.
    Keep a short human-readable prefix.
    """
    prefix = query.lower().strip().replace(' ', '_')[:40]
    key = f"{query}::{page_token or 'firstpage'}"
    digest = hashlib.sha256(key.encode('utf-8')).hexdigest()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return str(CACHE_DIR / f"{prefix}_{digest}.json")


def fetch_page(query: str, page_token: Optional[str] = None) -> Dict:
    """Fetch a single page from SerpApi with optional pagination, with caching and retries."""
    params = {
        'engine': 'google_jobs',
        'q': query,
        'api_key': API_KEY
    }
    if page_token:
        params['next_page_token'] = page_token

    cache_file = make_cache_filename(query, page_token)
    if Path(cache_file).exists():
        logging.info(f'Loading cached results for query="{query}" page="{page_token}" from {cache_file}')
        with open(cache_file, 'r') as f:
            return json.load(f)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logging.info(f'Fetching query="{query}" page="{page_token}" attempt {attempt}')
            r = SESSION.get(API_URL, params=params, timeout=15)

            if r.status_code == 429:
                retry_after = r.headers.get('Retry-After')
                try:
                    wait = int(retry_after) if retry_after and retry_after.isdigit() else (RETRY_BACKOFF ** attempt)
                except Exception:
                    wait = (RETRY_BACKOFF ** attempt)
                logging.warning(f'Rate limited (429). Sleeping for {wait} seconds before retry.')
                time.sleep(wait)
                raise requests.RequestException("429 Rate limited")

            r.raise_for_status()
            data = r.json()

            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)

            return data

        except requests.RequestException as e:
            logging.error(f'Error fetching data: {e}')
            if attempt < MAX_RETRIES:
                sleep_time = RETRY_BACKOFF ** attempt
                logging.info(f'Retrying after {sleep_time} seconds...')
                time.sleep(sleep_time)
            else:
                logging.error(f'Failed to fetch after {MAX_RETRIES} attempts.')
                raise


def _aggregate_cache_filename(query: str) -> str:
    """Filename for aggregated results (all pages combined)."""
    prefix = query.lower().strip().replace(' ', '_')[:40]
    key = f"{query}::aggregate"
    digest = hashlib.sha256(key.encode('utf-8')).hexdigest()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return str(CACHE_DIR / f"{prefix}_aggregate_{digest}.json")


def fetch_all_pages(query: str,
                    max_pages: int = 5,
                    max_total_results: Optional[int] = 30,
                    stop_if_no_new: bool = True,
                    known_ids: Optional[set] = None) -> List[Dict]:
    """
    Fetch multiple pages up to max_pages or until max_total_results unique jobs collected.
    Stops early if a page yields no new job_ids when stop_if_no_new=True.
    known_ids: optional set of job_ids already seen (useful for incremental polling).
    """
    # try to return aggregate cache if present
    aggregate_file = _aggregate_cache_filename(query)
    if Path(aggregate_file).exists():
        logging.info(f'Loading aggregated cached results for "{query}" from {aggregate_file}')
        try:
            return json.load(open(aggregate_file, 'r'))
        except Exception:
            logging.warning('Failed to read aggregate cache, refetching.')

    results: List[Dict] = []
    seen_ids = set(known_ids or [])
    page_token: Optional[str] = None
    current_page = 0

    while current_page < max_pages:
        page_data = fetch_page(query, page_token)
        jobs = page_data.get('jobs_results', []) or []

        new_jobs = []
        for job in jobs:
            job_id = job.get('job_id')
            # skip entries without job_id or already seen
            if job_id and job_id in seen_ids:
                continue
            if job_id:
                seen_ids.add(job_id)
            new_jobs.append(job)

        # extend results only with unique new items
        results.extend(new_jobs)

        # stop early if limit satisfied
        if max_total_results and len(results) >= max_total_results:
            results = results[:max_total_results]
            logging.info(f'Reached max_total_results={max_total_results} for "{query}", stopping pagination.')
            break

        # stop if page had no new items and stop_if_no_new set
        if stop_if_no_new and not new_jobs:
            logging.info(f'No new jobs on page {current_page+1} for "{query}", stopping pagination.')
            break

        serpapi_pagination = page_data.get('serpapi_pagination', {})
        page_token = serpapi_pagination.get('next_page_token')
        current_page += 1
        if not page_token:
            break

    # write aggregated cache for faster subsequent reads
    try:
        with open(aggregate_file, 'w') as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        logging.warning(f'Failed to write aggregate cache for "{query}": {e}')

    return results

def search_multiple_queries(queries: List[str], max_pages_per_query: int = 5) -> Dict[str, List[Dict]]:
    outputs: Dict[str, List[Dict]] = {}
    for query in queries:
        logging.info(f'Starting search for query: "{query}"')
        try:
            jobs = fetch_all_pages(query, max_pages=max_pages_per_query)
            outputs[query] = jobs
            logging.info(f'Found {len(jobs)} jobs for query "{query}"')
        except Exception as e:
            logging.error(f'Failed to fetch jobs for query "{query}": {e}')
            outputs[query] = []
    return outputs


def print_results(results: Dict[str, List[Dict]]):
    for query, jobs in results.items():
        print(f"\nQuery: {query} -- {len(jobs)} jobs found\n")
        for job in jobs:
            title = job.get('title', 'No title')
            company = job.get('company_name', 'Unknown company')
            location = job.get('location', 'Location not specified')
            print(f"- {title} at {company} ({location})")


if __name__ == '__main__':
    # Example queries (commas fixed)
    queries_to_search = [
        "cloud infrastructure engineer",
        "infrastructure engineer",
        "azure cloud engineer",
        "cloud engineer",
    ]

    results = search_multiple_queries(queries_to_search, max_pages_per_query=3)
    print_results(results)