"""Fake provider responses so the suite never calls a model or the network."""
import json
from types import SimpleNamespace


def completion(text='Answer [S1]', calls=None):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text, tool_calls=calls))])


def call(name='search_profile', args=None, call_id='test'):
    return SimpleNamespace(id=call_id, function=SimpleNamespace(
        name=name, arguments=json.dumps(args or {'query': 'Intuit'})))


def fake_research(text='Prasanna worked at Intuit. [S1]'):
    return [completion(calls=[call()]), completion(text)]
