"""Experiential Labs chat client. OpenAI's SDK is used only for wire compatibility."""
import os

from openai import OpenAI

from twin.config import MAX_RETRIES, MODEL_NAME, PROVIDER_BASE_URL, REQUEST_TIMEOUT


def call_genai(messages, model=None, tools=None, tool_choice=None):
    api_key = os.getenv('EXPLABS_API_KEY')
    if not api_key:
        raise RuntimeError('EXPLABS_API_KEY is not configured. Add it to the app environment and restart.')
    with OpenAI(api_key=api_key, base_url=PROVIDER_BASE_URL,
                timeout=REQUEST_TIMEOUT, max_retries=MAX_RETRIES) as client:
        payload = {'model': model or MODEL_NAME, 'messages': messages}
        if tools:
            payload['tools'] = tools
        if tool_choice:
            payload['tool_choice'] = tool_choice
        return client.chat.completions.create(**payload)
