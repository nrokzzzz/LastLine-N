import re
from dateutil import parser as date_parser
import dateparser
from datetime import datetime

def extract_deadline_date(email_text):
    text = email_text.lower()
    deadline_keywords = [
        "deadline", "last date", "apply by", "submit by", "registration closes",
        "registration ends", "valid till", "due by", "ends on"
    ]

    # Match many formats like 05-08-2025, 2025/08/05, Aug 5, 2025
    date_patterns = re.findall(
        r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|\w+\s+\d{1,2},?\s+\d{4})\b',
        text
    )

    relative_patterns = re.findall(
        r'(last|past|next|within|in)\s+(\d+)\s+(day|days|week|weeks|month|months)',
        text, re.IGNORECASE
    )

    deadline_sentences = []
    for line in text.split('.'):
        if any(kw in line for kw in deadline_keywords):
            deadline_sentences.append(line.strip())

    for sentence in deadline_sentences:
        for d in date_patterns:
            if d in sentence:
                try:
                    return date_parser.parse(d, dayfirst=True).date().isoformat()
                except:
                    continue
        for rel in relative_patterns:
            phrase = ' '.join(rel)
            if phrase in sentence:
                parsed = dateparser.parse(phrase, settings={'RELATIVE_BASE': datetime.now()})
                if parsed:
                    return parsed.date().isoformat()

    # Fallback to first date
    for d in date_patterns:
        try:
            return date_parser.parse(d, dayfirst=True).date().isoformat()
        except:
            continue

    for rel in relative_patterns:
        phrase = ' '.join(rel)
        parsed = dateparser.parse(phrase, settings={'RELATIVE_BASE': datetime.now()})
        if parsed:
            return parsed.date().isoformat()

    return None
