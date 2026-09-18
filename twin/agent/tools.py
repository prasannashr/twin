"""The agent's two read-only tools, their schemas, and strict argument validation."""
import json

from twin.privacy import PRIVACY_RESPONSE, is_private_request, redact_private_data

MAX_SOURCES = 12
PROFILE_SEARCH_LIMIT = 4
REQUIREMENT_SEARCH_LIMIT = 2
MAX_REQUIREMENTS = 6
MAX_ARGUMENT_CHARS = 4000
MAX_QUERY_CHARS = 500
MAX_REQUIREMENT_CHARS = 160

ALLOWED_TOOLS = ('search_profile', 'find_requirement_evidence')

TOOL_SCHEMAS = [
    {'type': 'function', 'function': {
        'name': 'search_profile',
        'description': 'Search documented career experience. Refine or narrow the query when evidence is missing. Returns cited passages, not verified conclusions.',
        'parameters': {'type': 'object', 'properties': {'query': {'type': 'string'}},
                       'required': ['query'], 'additionalProperties': False},
    }},
    {'type': 'function', 'function': {
        'name': 'find_requirement_evidence',
        'description': 'Retrieve profile evidence separately for up to six job requirements. Matching passages are candidates: assess support and report gaps; never invent a fit score.',
        'parameters': {'type': 'object', 'properties': {'requirements': {
            'type': 'array', 'items': {'type': 'string'}, 'minItems': 1, 'maxItems': MAX_REQUIREMENTS}},
            'required': ['requirements'], 'additionalProperties': False},
    }},
]

TOOL_ACTION_LABELS = {
    'search_profile': 'Searched the professional profile',
    'find_requirement_evidence': 'Looked up evidence for job requirements',
}


def register_sources(passages, sources):
    """Assign stable request-local IDs, de-duplicating and honouring the source cap."""
    result = []
    for passage in passages:
        text = redact_private_data(passage.text)
        existing = next((key for key, value in sources.items()
                         if value['source'] == passage.source and value['text'] == text), None)
        if existing is None:
            if len(sources) >= MAX_SOURCES:
                continue
            existing = f'S{len(sources) + 1}'
            sources[existing] = {'source': passage.source, 'text': text}
        result.append({'id': existing, **sources[existing]})
    return result


def _search_profile(args, sources, knowledge_base):
    if set(args) != {'query'} or not isinstance(args['query'], str):
        return {'error': 'Expected a query string.'}
    query = args['query'].strip()
    if not query or len(query) > MAX_QUERY_CHARS:
        return {'error': f'Query must contain 1–{MAX_QUERY_CHARS} characters.'}
    if is_private_request(query):
        return {'error': PRIVACY_RESPONSE}
    hits = knowledge_base.search(redact_private_data(query), limit=PROFILE_SEARCH_LIMIT)
    return {'passages': register_sources(hits, sources),
            'note': 'No matches.' if not hits else 'Assess evidence before making claims.'}


def _find_requirement_evidence(args, sources, knowledge_base):
    requirements = args.get('requirements')
    if (set(args) != {'requirements'} or not isinstance(requirements, list)
            or not 1 <= len(requirements) <= MAX_REQUIREMENTS
            or any(not isinstance(r, str) or not r.strip() or len(r) > MAX_REQUIREMENT_CHARS
                   for r in requirements)):
        return {'error': f'Provide 1–{MAX_REQUIREMENTS} requirements of at most {MAX_REQUIREMENT_CHARS} characters each.'}
    if any(is_private_request(r) for r in requirements):
        return {'error': PRIVACY_RESPONSE}
    results = []
    for requirement in requirements:
        requirement = redact_private_data(requirement)
        hits = knowledge_base.search(requirement, limit=REQUIREMENT_SEARCH_LIMIT)
        results.append({'requirement': requirement, 'candidate_evidence': register_sources(hits, sources)})
    return {'requirements': results,
            'note': 'A keyword match is not proof. Mark unsupported requirements as not documented.'}


def execute_readonly_tool(name, arguments, sources, knowledge_base):
    """Explicit allowlist and schema validation; never resolve arbitrary functions."""
    if name not in ALLOWED_TOOLS:
        return {'error': 'Tool is not permitted.'}
    try:
        if len(arguments) > MAX_ARGUMENT_CHARS:
            return {'error': 'Tool arguments are too long.'}
        args = json.loads(arguments)
        if not isinstance(args, dict):
            return {'error': 'Expected an argument object.'}
        if name == 'search_profile':
            return _search_profile(args, sources, knowledge_base)
        return _find_requirement_evidence(args, sources, knowledge_base)
    except (ValueError, TypeError):
        return {'error': 'Invalid tool arguments.'}
