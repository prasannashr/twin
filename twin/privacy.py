"""Deterministic privacy checks shared by retrieval, prompts, and output.

Public portfolio identity and career facts remain available. These checks cover
common contact/identifier formats, not universal detection of every form of PII.
"""
import html
import re
import unicodedata
from urllib.parse import unquote

REDACTED = '[private detail removed]'
PRIVACY_RESPONSE = (
    'I can discuss professional experience and skills, but I cannot share private '
    'contact details, addresses, personal identifiers, or hidden profile data.'
)

PATTERNS = [
    # Personal profile links are unnecessary for career Q&A; remove all URLs.
    re.compile(r'(?i)(?:https?://|www\.)[^\s<>\)\]]+'),
    re.compile(r'(?i)\b(?:linkedin\.com|github\.com|facebook\.com|instagram\.com|twitter\.com|x\.com)/[^\s<>\)\]]+'),
    re.compile(r'(?i)(?:mailto:)?[a-z0-9.!#$%&\'*+/=?^_`{|}~-]+\s*@\s*[a-z0-9-]+(?:\s*\.\s*[a-z0-9-]+)+'),
    re.compile(r'(?i)\b[a-z0-9._%+-]+\s*(?:\[at\]|\(at\)| at )\s*[a-z0-9.-]+\s*(?:\[dot\]|\(dot\)| dot )\s*[a-z]{2,}\b'),
    # North American numbers, including dotted and space-separated PDF text.
    re.compile(r'(?<!\w)(?:\+?1[\s.\-]*)?\(?\d{3}\)?[\s.\-]*\d{3}[\s.\-]*\d{4}(?!\d)'),
    # International phone numbers with an explicit country prefix.
    re.compile(r'(?<!\w)\+\d(?:[\s().\-]*\d){7,14}(?!\d)'),
    re.compile(r'\b\d{3}[ -]\d{2}[ -]\d{4}\b'),  # SSN
    re.compile(r'(?<!\d)(?:\d[ -]?){13,19}(?!\d)'),  # cards / long identifiers
    re.compile(r'(?i)\b\d{1,6}\s+(?:[a-z0-9]+[ .-]+){1,6}(?:street|st|avenue|ave|road|rd|lane|ln|drive|dr|court|ct|boulevard|blvd|way)\b(?:\s+(?:apt|unit|suite|#)\s*[\w-]+)?'),
    re.compile(r'(?i)\bP\.?\s*O\.?\s+Box\s+\d+\b'),
    re.compile(r'(?i)\b(?:date of birth|dob|born on|passport(?: number)?|social security(?: number)?|ssn|driver.?s? licen[cs]e(?: number)?|bank account(?: number)?)\s*[:#=-]?\s*[^\n,;]+'),
]

PRIVATE_REQUEST = re.compile(
    r'(?i)\b(?:phone|telephone|mobile|cellphone|whatsapp|email|e-mail|contact|'
    r'address|postcode|zip code|birthday|birthdate|date of birth|dob|passport|'
    r'ssn|social security|credit card|bank account|driver.?s? licen[cs]e|'
    r'personal (?:data|details|information)|private (?:data|details|information)|'
    r'linkedin|social media|profile link|reach you|reach him|call you|call him|'
    r'where (?:do you|does he) live|home location)\b'
)
EXTRACTION_REQUEST = re.compile(
    r'(?i)\b(?:dump|reveal|print|repeat|show|output|encode|base64|rot13|translate)\b'
    r'.{0,100}\b(?:raw|full|entire|original|hidden|unredacted|system prompt|documents?|pdf|profile|context|sources?|digits)\b'
)


def normalized_text(text):
    text = html.unescape(unquote(unicodedata.normalize('NFKC', text)))
    return re.sub(r'[\u200b-\u200f\u202a-\u202e\u2060\ufeff]', '', text)


def redact_private_data(text):
    text = normalized_text(text)
    for pattern in PATTERNS:
        text = pattern.sub(REDACTED, text)
    return text


def is_private_request(text):
    normalized = normalized_text(text)
    return bool(PRIVATE_REQUEST.search(normalized) or EXTRACTION_REQUEST.search(normalized))
