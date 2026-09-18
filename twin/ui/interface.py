"""Chat entry point: input limits, history normalization, and evidence rendering."""
import logging

import gradio as gr
from openai import OpenAIError

from twin.agent import run_agent
from twin.privacy import PRIVACY_RESPONSE, is_private_request, redact_private_data
from twin.provider import call_genai
from twin.ui.styles import EXAMPLES

MAX_HISTORY_MESSAGES = 12
MAX_MESSAGE_CHARS = 6000

TITLE = 'Prasanna’s Digital Twin'
DESCRIPTION = ('An AI career assistant that can search my profile and compare job requirements. '
               'Ask about experience, skills, or job fit, and inspect the sources behind each answer.')


def normalize_history(history):
    """Accept Gradio text and text-block messages; discard client system roles."""
    normalized = []
    for item in (history or [])[-MAX_HISTORY_MESSAGES:]:
        role = item.get('role')
        if role not in ('user', 'assistant'):
            continue
        content = item.get('content', '')
        if isinstance(content, list):
            content = '\n'.join(block.get('text', '') for block in content
                                if isinstance(block, dict) and block.get('type') == 'text')
        if isinstance(content, str) and content.strip():
            # Source excerpts are rebuilt for each turn, not replayed as chat history.
            content = content.split('<details>', 1)[0]
            normalized.append({'role': role, 'content': redact_private_data(content[:MAX_MESSAGE_CHARS])})
    return normalized


def _render_sources(sources):
    excerpts = '\n\n'.join(f"**[{key}] {p['source']}**\n\n> " + p['text'].replace('\n', '\n> ')
                           for key, p in sources.items())
    return '\n\n<details><summary>Retrieved sources (privacy-filtered)</summary>\n\n' + excerpts + '\n\n</details>'


def _render_actions(actions):
    return '\n\n<details><summary>Research actions</summary>\n\n' + '\n'.join(
        f'- {action}' for action in actions) + '\n\n</details>'


def chat(message, history):
    if not message or not message.strip():
        return 'Please ask a question about Prasanna.'
    if len(message) > MAX_MESSAGE_CHARS:
        return f'Please shorten your question or job description to {MAX_MESSAGE_CHARS:,} characters.'
    if is_private_request(message):
        return PRIVACY_RESPONSE
    try:
        message = redact_private_data(message)
        history = normalize_history(history)
        result = run_agent(message, history, call_genai)
        text = result['answer']
        if result['sources']:
            text += _render_sources(result['sources'])
        if result['actions']:
            text += _render_actions(result['actions'])
        return redact_private_data(text)
    except RuntimeError as exc:
        logging.warning('Twin configuration failure (%s)', type(exc).__name__)
        return 'The model service is not configured. Please check the server settings.'
    except OpenAIError as exc:
        logging.warning('Twin model request failed (%s)', type(exc).__name__)
        return 'The model service is unavailable or rejected the request. Check the server API key, model access, and quota, then try again.'
    except (OSError, ValueError) as exc:
        logging.warning('Twin knowledge request failed (%s)', type(exc).__name__)
        return 'The profile documents could not be processed. Please check data/summary.txt and data/linkedin.pdf on the server.'


def build_ui():
    return gr.ChatInterface(
        chat, examples=EXAMPLES, title=TITLE, description=DESCRIPTION,
        chatbot=gr.Chatbot(show_label=False),
    )
