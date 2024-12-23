# match_pipeline/text_cleaning.py

import re

def clean_text(raw_text: str) -> str:
    if not raw_text:
        return ""
    # Strip whitespace, remove unwanted chars
    cleaned = raw_text.strip()
    # Example: remove non-ASCII
    cleaned = re.sub(r'[^\x00-\x7F]+', ' ', cleaned)
    return cleaned
