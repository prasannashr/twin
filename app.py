
from context import TWIN_SYSTEM_PROMPT
from tools import tools, handle_tool_calls
from styles import CSS, JS, EXAMPLES
from dotenv import load_dotenv
import gradio as gr
import httpx
from openrouter import OpenRouter
import os

load_dotenv(override=True)

MODEL_NAME = "gpt-5.4-mini"


system = [{"role": "system", "content": TWIN_SYSTEM_PROMPT}]

def call_genai(messages, model: str = "openai/gpt-4o-mini", tools=None, tool_choice=None) -> str:
    #load_dotenv()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set. Add it to your environment or a .env file before running this script.")

    with OpenRouter(api_key=api_key) as client:
        try:
            payload = {
                "model": model,
                "messages": messages,
            }

            if tools is not None:
                payload["tools"] = tools

            if tool_choice is not None:
                payload["tool_choice"] = tool_choice

            response = client.chat.send(**payload)
            
        except (httpx.HTTPError, RuntimeError, ValueError) as exc:
            raise RuntimeError(f"OpenRouter request failed: {exc}") from exc

        #print(response.choices[0])
        return response

def chat(message, history):
    messages = system + history + [{"role": "user", "content": message}]
    #response = openai.chat.completions.create(model=MODEL_NAME, messages=messages, tools=tools)
    response = call_genai(messages, model="deepseek/deepseek-v4-flash-0731", tools=tools)
    while response.choices[0].finish_reason == "tool_calls":
        message = response.choices[0].message
        tool_calls = message.tool_calls
        results = handle_tool_calls(tool_calls)
        messages.append(message)
        messages.extend(results)
        response = call_genai(messages, model="deepseek/deepseek-v4-flash-0731", tools=tools)
    return response.choices[0].message.content


if __name__ == "__main__":
    gr.ChatInterface(
        chat,
        examples=EXAMPLES,
        title="Digital Twin",
        description="Talk to my AI twin about my career",
        chatbot=gr.Chatbot(show_label=False),
    ).launch(css=CSS, js=JS, theme=gr.themes.Base())
