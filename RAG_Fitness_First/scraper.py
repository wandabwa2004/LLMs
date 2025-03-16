import requests
from bs4 import BeautifulSoup
import re
import os
import time  # Added time in order to make some time between requests

print("Script has started!")


def download_faqs(url):
    """
    Downloads FAQs and answers from the Freshdesk support page.

    Args:
        url: The URL of the Freshdesk support homepage.

    Returns:
        A list of dictionaries, where each dictionary represents an FAQ
        and contains the question and answer. Returns None if an error occurs.
    """
    try:
        print(f"Attempting to fetch the base URL: {url}")
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        print(f"Successfully fetched the base URL: {url}")

        soup = BeautifulSoup(response.content, 'html.parser')

        # Find all category links
        category_links = []
        category_divs = soup.find_all('div', class_='list-lead')
        print(f"Number of category divs found: {len(category_divs)}")  # added this print
        for div in category_divs:
            link = div.find('a')
            if link and link.has_attr('href'):
                category_links.append(link['href'])
        print(f"Number of category links found: {len(category_links)}")  # added this print

        all_faqs = []

        for category_link in category_links:
            full_category_url = "https://fitnesspassport.freshdesk.com" + category_link
            print(f"Processing category: {full_category_url}")

            try:
                category_response = requests.get(full_category_url)
                category_response.raise_for_status()
                category_soup = BeautifulSoup(category_response.content, 'html.parser')
                print(f"Successfully fetched category: {full_category_url}")
            except requests.exceptions.RequestException as e:
                print(f"Error fetching category {full_category_url}: {e}")
                continue  # Skip to the next category on error

            # Find all article links within the category
            article_links = []
            article_divs = category_soup.find_all('div', class_='ellipsis') # Corrected line. We find all the div with ellipsis
            print(f"  Number of article divs found: {len(article_divs)}")  # added this print

            for div in article_divs: # We loop in the divs
                link = div.find('a') # We get the link from the div
                if link and link.has_attr('href'):
                    article_links.append(link['href'])
            print(f"  Number of article links found: {len(article_links)}")  # added this print

            for article_link in article_links:
                full_article_url = "https://fitnesspassport.freshdesk.com" + article_link
                print(f"  Processing article: {full_article_url}")
                try:
                    article_response = requests.get(full_article_url)
                    article_response.raise_for_status()
                    article_soup = BeautifulSoup(article_response.content, 'html.parser')
                    print(f"  Successfully fetched article: {full_article_url}")
                    time.sleep(1)  # Wait 1 second after each request (added for respect)
                except requests.exceptions.RequestException as e:
                    print(f"  Error fetching article {full_article_url}: {e}")
                    continue  # Skip to the next article on error

                # Extract question (title)
                try:
                    question_element = article_soup.find('h2', class_='heading')  # Changed this line
                    if question_element:
                        question = question_element.get_text(strip=True)
                    else:
                        print(f"    Warning: Could not find question for {full_article_url}")
                        continue
                except Exception as e:
                    print(
                        f"    Warning: Error while getting the question for {full_article_url}, error: {e}")
                    continue

                # Extract Answer (content)
                try:
                    answer_element = article_soup.find('article', class_='article-body')  # changed this line
                    if answer_element:
                        answer = answer_element.get_text(strip=True)

                        # Clean up the answer text a bit (optional)
                        answer = re.sub(r'\s+', ' ', answer)  # remove extra whitspaces
                        answer = re.sub(r'<[^>]+>', '', answer)  # remove all the tags
                    else:
                        print(f"    Warning: Could not find answer for {full_article_url}")
                        continue
                except Exception as e:
                    print(
                        f"    Warning: Error while getting the answer for {full_article_url}, error: {e}")
                    continue

                all_faqs.append({"question": question, "answer": answer})
                print(f"Successfully extracted Question and Answer from: {full_article_url}")

        print(f"Total FAQs extracted: {len(all_faqs)}")  # Added this print
        if len(all_faqs) == 0:
            return None
        else:
            return all_faqs

    except requests.exceptions.ConnectionError as e:
        print(f"Connection error occurred: {e}")
        return None
    except requests.exceptions.Timeout as e:
        print(f"Timeout error occurred: {e}")
        return None
    except requests.exceptions.TooManyRedirects as e:
        print(f"Too many redirects occurred: {e}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"A requests error occurred: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


def save_faqs_to_file(faqs, filename="faqs.txt"):
    """
    Saves the extracted FAQs to a text file.

    Args:
        faqs: A list of dictionaries, where each dictionary contains a question and answer.
        filename: The name of the file to save the data to.
    """
    if faqs:
        try:
            if not os.path.exists("output"):
                os.makedirs("output")
            with open(os.path.join("output", filename), 'w', encoding='utf-8') as f:
                for faq in faqs:
                    f.write(f"Question: {faq['question']}\n")
                    f.write(f"Answer: {faq['answer']}\n")
                    f.write("-" * 20 + "\n")
            print(f"Successfully saved FAQs to {filename}")
        except Exception as e:
            print(f"Error saving to file: {e}")
    else:
        print("No FAQs to save.")


def save_faqs_to_json(faqs, filename="faqs.json"):
    """
    Saves the extracted FAQs to a JSON file.

    Args:
        faqs: A list of dictionaries, where each dictionary contains a question and answer.
        filename: The name of the file to save the data to.
    """
    import json

    if faqs:
        try:
            if not os.path.exists("output"):
                os.makedirs("output")

            with open(os.path.join("output", filename), 'w', encoding='utf-8') as f:
                json.dump(faqs, f, indent=4, ensure_ascii=False)
            print(f"Successfully saved FAQs to {filename}")
        except Exception as e:
            print(f"Error saving to JSON file: {e}")
    else:
        print("No FAQs to save.")


if __name__ == "__main__":
    base_url = "https://fitnesspassport.freshdesk.com/support/home"
    extracted_faqs = download_faqs(base_url)
    if extracted_faqs:
        print("Saving  to file ....")
        save_faqs_to_file(extracted_faqs)
        print("Saving  to file .... done!")
        print("-----------------------------")
        print("Saving  to JSON ..........")
        save_faqs_to_json(extracted_faqs)
        print("Saving  to JSON .... done!")
    else:
        print("Nothing to save")

