#Script  to scrape Safaricom FAQs and  answers  

import requests
import json
import time
import random
from bs4 import BeautifulSoup

def random_delay(min_seconds=1, max_seconds=3):
    """Sleep for a random interval between min_seconds and max_seconds."""
    delay = random.uniform(min_seconds, max_seconds)
    time.sleep(delay)

def get_category_urls(landing_url):
    """
    Scrapes the landing page to collect all category URLs.
    Adjust the CSS selector if needed.
    """
    response = requests.get(landing_url)
    random_delay()  # Random delay after request
    soup = BeautifulSoup(response.text, 'html.parser')
    # Select anchor tags that contain the FAQ category path
    category_links = soup.select("a[href*='/media-center-landing/frequently-asked-questions/']")
    urls = set()
    for link in category_links:
        href = link.get("href")
        # Build absolute URL if it's relative
        if href.startswith("/"):
            href = "https://www.safaricom.co.ke" + href
        urls.add(href)
    return list(urls)

def scrape_faq_pairs(category_url):
    """
    Scrapes the FAQ pairs from a given category page.
    It uses the <div class="card-header"> element to find the question and
    uses the 'href' attribute to locate the corresponding answer block.
    Now extracts answer content from both paragraphs and list elements.
    """
    response = requests.get(category_url)
    random_delay()  # Random delay after request
    soup = BeautifulSoup(response.text, 'html.parser')
    faq_pairs = []
    
    # Find all FAQ header items
    card_headers = soup.find_all('div', class_='card-header')
    print(f"Found {len(card_headers)} FAQ items on {category_url}")
    
    for header in card_headers:
        # Extract the question text from the <a class="card-title">
        question_elem = header.find('a', class_='card-title')
        question_text = question_elem.get_text(strip=True) if question_elem else ''
        
        # Use the href attribute to find the corresponding answer block.
        # Example: href="#collapse47" => target id is "collapse47"
        answer_ref = header.get('href', '')
        if answer_ref.startswith("#"):
            answer_id = answer_ref[1:]
        else:
            answer_id = answer_ref
        
        answer_div = soup.find('div', id=answer_id)
        answer_text = ""
        if answer_div:
            # Use get_text with a newline separator to include text from paragraphs and list items.
            answer_text = answer_div.get_text(separator="\n", strip=True)
        else:
            print(f"Answer block not found for header with href: {answer_ref}")
        
        if question_text and answer_text:
            faq_pairs.append({"question": question_text, "answer": answer_text})
    
    return faq_pairs

def main():
    # Landing page with all FAQ categories
    landing_page_url = "https://www.safaricom.co.ke/media-center-landing/frequently-asked-questions"
    
    # Get all category URLs
    category_urls = get_category_urls(landing_page_url)
    print("Found category pages:", category_urls)
    
    # Dictionary to hold all data: { category_url: [ {question, answer}, ... ] }
    data = {}
    
    # Loop over each category page and scrape the FAQ pairs
    for category_url in category_urls:
        print("Scraping:", category_url)
        faqs = scrape_faq_pairs(category_url)
        data[category_url] = faqs
        random_delay(2, 4)  # Slightly longer delay between category requests
    
    # Save the collected data into a JSON file
    output_file = "faq_data.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    print("FAQ data saved to", output_file)

if __name__ == "__main__":
    main()