"""Bounded LangGraph tool loop with privacy and citation validation at the exit."""
import json
import re
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from twin.agent.prompts import AGENT_INSTRUCTIONS, TWIN_SYSTEM_PROMPT
from twin.agent.tools import (TOOL_ACTION_LABELS, TOOL_SCHEMAS, execute_readonly_tool)
from twin.privacy import PRIVACY_RESPONSE, is_private_request, redact_private_data
from twin.retrieval import load_knowledge_base

MAX_TOOL_ROUNDS = 3
MAX_CALLS_PER_ROUND = 2
RECURSION_LIMIT = 12

BUDGET_EXHAUSTED_NOTICE = ('Research budget exhausted. Answer using the evidence already retrieved, '
                           'or explain what is not documented. Do not call any tools.')
NO_EVIDENCE_ANSWER = ('I do not have enough retrieved profile evidence to answer that. '
                      'Please ask a more specific career question.')
BAD_CITATION_ANSWER = 'I could not validate the source references in this answer. Please rephrase the question.'
UNCITED_ANSWER = 'I found candidate evidence but could not produce a source-cited answer. Please narrow the question.'
LIMIT_REACHED_ANSWER = 'The research limit was reached. Please narrow the question.'

GREETING = re.compile(r'(hi|hello|hey|thanks|thank you)[.! ]*', re.I)
CITATION = re.compile(r'\[(S\d+)\]')


class AgentState(TypedDict, total=False):
    messages: list
    sources: dict
    actions: list[str]
    rounds: int
    calls: list
    answer: str
    done: bool


def build_agent(model_call, knowledge_base=None):
    """Use an injectable client for offline tests; no persisted graph checkpoints."""
    def reason(state):
        messages = state['messages']
        at_limit = state['rounds'] >= MAX_TOOL_ROUNDS
        if at_limit:
            messages = messages + [{'role': 'system', 'content': BUDGET_EXHAUSTED_NOTICE}]
        response = model_call(messages, tools=TOOL_SCHEMAS,
                              tool_choice='none' if at_limit else 'auto')
        answer = response.choices[0].message
        calls = answer.tool_calls or []
        if calls and at_limit:
            return {'answer': LIMIT_REACHED_ANSWER, 'done': True, 'calls': []}
        if calls:
            # Serialize only permitted protocol fields and redact model-produced arguments.
            serialized = [{'id': c.id, 'type': 'function', 'function': {
                'name': c.function.name, 'arguments': redact_private_data(c.function.arguments)}} for c in calls]
            return {'messages': messages + [{'role': 'assistant', 'content': None, 'tool_calls': serialized}],
                    'calls': serialized, 'done': False}
        return {'answer': redact_private_data(answer.content or 'Please try rephrasing the question.'),
                'calls': [], 'done': True}

    def use_tools(state):
        sources = dict(state['sources'])
        actions = list(state['actions'])
        messages = list(state['messages'])
        kb = knowledge_base or load_knowledge_base()
        for number, call in enumerate(state['calls']):
            name = call['function']['name']
            if number >= MAX_CALLS_PER_ROUND:
                result = {'error': 'Tool call budget for this round exceeded.'}
            else:
                result = execute_readonly_tool(name, call['function']['arguments'], sources, kb)
                if 'error' not in result:
                    actions.append(TOOL_ACTION_LABELS[name])
            messages.append({'role': 'tool', 'tool_call_id': call['id'],
                             'content': redact_private_data(json.dumps(result))})
        return {'messages': messages, 'sources': sources, 'actions': actions,
                'rounds': state['rounds'] + 1, 'calls': []}

    def finish(state):
        answer = state['answer']
        cited = set(CITATION.findall(answer))
        question = next(m['content'] for m in reversed(state['messages']) if m['role'] == 'user')
        greeting = GREETING.fullmatch(question.strip())
        if not state['sources'] and not greeting:
            answer = NO_EVIDENCE_ANSWER
        elif cited - state['sources'].keys():
            answer = BAD_CITATION_ANSWER
        elif state['sources'] and not cited:
            answer = UNCITED_ANSWER
        return {'answer': redact_private_data(answer)}

    graph = StateGraph(AgentState)
    graph.add_node('agent', reason)
    graph.add_node('tools', use_tools)
    graph.add_node('privacy_and_citations', finish)
    graph.add_edge(START, 'agent')
    graph.add_conditional_edges('agent', lambda state: 'finish' if state['done'] else 'tools',
                                {'finish': 'privacy_and_citations', 'tools': 'tools'})
    graph.add_edge('tools', 'agent')
    graph.add_edge('privacy_and_citations', END)
    return graph.compile()


def run_agent(question, history, model_call, knowledge_base=None):
    # Enforce privacy even for callers that bypass the UI.
    if is_private_request(question):
        return {'answer': PRIVACY_RESPONSE, 'sources': {}, 'actions': [], 'rounds': 0}
    clean_history = [{'role': item['role'], 'content': redact_private_data(item['content'])}
                     for item in history if item.get('role') in ('user', 'assistant')]
    messages = [{'role': 'system', 'content': TWIN_SYSTEM_PROMPT + '\n\n' + AGENT_INSTRUCTIONS},
                *clean_history, {'role': 'user', 'content': redact_private_data(question)}]
    # Explicitly disable environment-enabled LangSmith tracing for profile data.
    from langsmith import tracing_context
    with tracing_context(enabled=False):
        return build_agent(model_call, knowledge_base).invoke(
            {'messages': messages, 'sources': {}, 'actions': [], 'rounds': 0, 'calls': [], 'done': False},
            config={'recursion_limit': RECURSION_LIMIT},
        )
