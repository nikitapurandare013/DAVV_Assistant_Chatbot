import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from tqdm import tqdm
import time

BASE_URL = "https://www.dauniv.ac.in/"
CRAWL_LIMIT = 50  # Max number of pages to crawl

visited = set()
pages_data = []

def is_valid_url(url):
    parsed = urlparse(url)
    return parsed.netloc == urlparse(BASE_URL).netloc

def clean_text(html):
    soup = BeautifulSoup(html, "html.parser")

    # Remove noisy elements
    for tag in soup(["script", "style", "nav", "footer", "header", "form", "noscript"]):
        tag.decompose()

    # Try to focus only on main content blocks
    possible_main_blocks = soup.find_all(["div", "section", "article", "main"], recursive=True)
    
    # Join all their text content
    all_text = "\n".join([
        block.get_text(separator=" ", strip=True)
        for block in possible_main_blocks
        if len(block.get_text(strip=True)) > 100  # Filter out very small snippets
    ])

    # Final cleanup: remove extra blank lines
    lines = [line.strip() for line in all_text.splitlines() if line.strip()]
    return "\n".join(lines)


def crawl(url, limit=CRAWL_LIMIT):
    queue = [url]

    with tqdm(total=limit, desc="Crawling DAVV") as pbar:
        while queue and len(visited) < limit:
            current_url = queue.pop(0)
            if current_url in visited:
                continue

            try:
                response = requests.get(current_url, timeout=10)
                visited.add(current_url)

                if response.status_code == 200:
                    text = clean_text(response.text)
                    pages_data.append({
                        "url": current_url,
                        "content": text
                    })

                    soup = BeautifulSoup(response.text, "html.parser")
                    links = soup.find_all("a", href=True)

                    for link in links:
                        abs_url = urljoin(current_url, link["href"])
                        if is_valid_url(abs_url) and abs_url not in visited:
                            queue.append(abs_url)
                    
                    pbar.update(1)
                time.sleep(0.5)  

            except Exception as e:
                print(f"Error scraping {current_url}: {e}")

    return pages_data

if __name__ == "__main__":
    print("Starting DAVV site scraping...")
    data = crawl(BASE_URL)

    # Save to file
    import json
    with open("davv_scraped_data.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n Scraped {len(data)} pages. Data saved to 'davv_scraped_data.json'.")
