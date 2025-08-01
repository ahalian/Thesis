"""Functions for scraping https://200.zona.media"""

import time
import requests
import pandas as pd
from random import uniform
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
 

BASE_URL = "https://200.zona.media"
REQUEST_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

def _make_request(url):
    """Helper function to make HTTP requests with proper headers and error handling."""
    time.sleep(uniform(0.5, 1.5))  # Respectful delay between requests
    try:
        response = requests.get(url, headers=REQUEST_HEADERS)
        response.encoding = 'utf-8'
        response.raise_for_status()  # Raise HTTP errors
        return response
    except requests.RequestException as e:
        print(f"Request failed for {url}: {e}")
        return None

def _get_regions():
    """Scrapes and returns a list of region page URLs from the base regions page.

    Returns:
        list[str]: A list of relative URLs (href attributes) pointing to individual 
        region pages. Returns an empty list if no matching links are found.
    """
    regions_url = urljoin(BASE_URL, "все_регионы.html")
    response = _make_request(regions_url)
    if not response:
        return []
    
    soup = BeautifulSoup(response.text, 'html.parser')
    return [a['href'] for a in soup.select('ul.tiles a[href$=".html"]')]

def _get_military_branches(region_url):
    """Scrapes and returns a list of military branch pages URLs from each regional page.

    Args:
        region_url: Full URL to the region page

    Returns:
        list[str]: A list of absolute URLs pointing to individual 
        branch-region pages. Returns an empty list if no matching links are found.
    """
    response = _make_request(region_url)
    if not response:
        return []
    
    soup = BeautifulSoup(response.text, 'html.parser')
    return [urljoin(BASE_URL, a['href']) 
            for a in soup.select('ul.tiles a[href*="/"]')]

def _get_profiles(branch_url):
    """Scrapes and returns a list of individual page URLs from the base branch-region page.

    Args:
        branch_url: Full URL to the military branch page

    Returns:
        list[str]: A list of absolute URLs pointing to individual pages.
        Returns an empty list if no matching links are found.
    """
    response = _make_request(branch_url)
    if not response:
        return []
    
    soup = BeautifulSoup(response.text, 'html.parser')
    tiles_div = soup.select_one('div.tiles')
    return [] if not tiles_div else [
        urljoin(BASE_URL, a['href']) 
        for a in tiles_div.find_all('a', href=True)
    ]

def _extract_profile_data(profile_url):
    """Extracts structured data from an individual profile page.

    Args:
        profile_url: Full URL to the profile page

    Returns:
        dict: Structured data containing profile information
    """
    response = _make_request(profile_url)
    if not response:
        return None

    soup = BeautifulSoup(response.text, 'html.parser')
    
    dates_div = soup.select_one('div.card__dates')
    birth_date = death_date = None
    if dates_div:
        dates_text = dates_div.get_text(strip=True)
        if "—" in dates_text:
            birth_date, death_date = dates_text.split("—", 1)
            birth_date = birth_date.strip()
            death_date = death_date.split("(")[0].strip()

    path = profile_url.replace(BASE_URL, "").strip("/")
    region, branch, filename = path.split("/")[:3]
    
    return {
        'region': region.replace("_", " "),
        'branch': branch.replace("_", " "),
        'name': filename.replace("_", " ").replace(".html", ""),
        'birth_date': birth_date,
        'death_date': death_date,
        'profile_url': profile_url
    }

def scrape_all_profiles():
    
    """Main function to execute the complete scraping workflow."""

    all_profiles = []
    regions = _get_regions()
    for region in tqdm(regions, desc="Collecting profiles"):
        region_url = urljoin(BASE_URL, region)
        branches = _get_military_branches(region_url)
        for branch_url in branches:
            profiles = _get_profiles(branch_url)
            all_profiles.extend(profiles)

    print(f"Total profiles found: {len(all_profiles)}")

    profiles_data = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(_extract_profile_data, url) for url in all_profiles]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Parsing profiles"):
            try: 
                result = future.result()
                if result:
                    profiles_data.append(result)
            except Exception as e:
                print(f"Thread failed: {e}")

    return profiles_data


def save_to_csv(data, filename="../data/daily.csv"):
    """Saves scraped data to a CSV file.
    
    Args:
        data: List of dictionaries containing profile data
        filename: Output file path
    """
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"Data saved to {filename}")

if __name__ == "__main__":
    data = scrape_all_profiles()
    save_to_csv(data)