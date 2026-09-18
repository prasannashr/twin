"""Document ingestion: privacy filtering, chunking, and a cached index build."""
from functools import lru_cache
import re

from pypdf import PdfReader

from twin.config import PROFILE_PDF_PATH, SUMMARY_PATH
from twin.privacy import redact_private_data
from twin.retrieval.knowledge_base import KnowledgeBase, Passage

CHUNK_SIZE = 180
CHUNK_OVERLAP = 35

# Personal biography and contact sections are never part of the public index.
EXCLUDED_SECTIONS = ('IDENTITY AND BACKGROUND', 'PERSONAL INTERESTS', 'CONTACT AND INFORMATION GAPS')


def chunk_text(text, source, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = redact_private_data(text).split()
    passages = []
    for start in range(0, len(words), size - overlap):
        passages.append(Passage(f'{source}, passage {len(passages) + 1}', ' '.join(words[start:start + size])))
        if start + size >= len(words):
            break
    return passages


def build_passages():
    passages = []
    summary = SUMMARY_PATH.read_text(encoding='utf-8')
    for number, paragraph in enumerate(re.split(r'\n\s*\n', summary), 1):
        if paragraph.startswith(EXCLUDED_SECTIONS):
            continue
        passages.extend(chunk_text(paragraph, f'summary.txt, section {number}'))
    for number, page in enumerate(PdfReader(PROFILE_PDF_PATH).pages, 1):
        passages.extend(chunk_text(page.extract_text() or '', f'linkedin.pdf, page {number}'))
    return passages


@lru_cache(maxsize=1)
def load_knowledge_base():
    return KnowledgeBase(build_passages())
