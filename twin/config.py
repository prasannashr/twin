"""Paths, model settings, and server bindings resolved once at import time."""
import os
from pathlib import Path

from dotenv import load_dotenv

# twin/config.py -> twin/ -> the application root that holds data/ and app.py.
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
SUMMARY_PATH = DATA_DIR / 'summary.txt'
PROFILE_PDF_PATH = DATA_DIR / 'linkedin.pdf'

# App-specific values win; the course repository root is a fallback for shared keys.
load_dotenv(BASE_DIR / '.env', override=False)
load_dotenv(BASE_DIR.parent.parent / '.env', override=False)

MODEL_NAME = os.getenv('TWIN_MODEL', 'gpt-5.6-luna')
PROVIDER_BASE_URL = 'https://api.experientiallabs.ai/v1'
REQUEST_TIMEOUT = 90
MAX_RETRIES = 1


def server_name():
    return '0.0.0.0' if os.getenv('RENDER') == 'true' else '127.0.0.1'


def server_port():
    return int(os.getenv('PORT', '7861'))
