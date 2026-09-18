"""Agent behaviour eval against the live provider.

This costs real model calls. Every assertion is mechanical: tool choice,
abstention, citation presence, refusals, and loop budgets. Nothing here judges
whether an answer reads well or whether a cited passage actually supports the
claim; that is faithfulness, and it needs a judge model.

Each run is written to eval/runs/ so a later faithfulness pass can score the
saved answers without paying for generation again.

    python eval/run_agent_eval.py --self-check   # free; verifies the harness
    python eval/run_agent_eval.py                # live; costs provider calls
    python eval/run_agent_eval.py --group citation --limit 3
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from twin.agent import MAX_TOOL_ROUNDS, run_agent  # noqa: E402
from twin.privacy import PRIVACY_RESPONSE  # noqa: E402
from twin.retrieval import load_knowledge_base  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
CASES_PATH = EVAL_DIR / 'cases' / 'agent_cases.jsonl'
RUNS_DIR = EVAL_DIR / 'runs'
GROUP_ORDER = ['tool_selection', 'citation', 'abstention', 'privacy', 'greeting']

# Wording the graph uses when it refuses to answer without retrieved evidence.
ABSTENTION_MARKERS = ('not have enough', 'not documented', 'no evidence', 'does not document',
                      'not confirmed', 'could not', 'not specify', 'no information')


class RecordingModel:
    """Wrap a provider callable and record the tool calls the model requests."""

    def __init__(self, inner):
        self.inner = inner
        self.calls = []
        self.invocations = 0

    def __call__(self, messages, tools=None, tool_choice=None):
        self.invocations += 1
        response = self.inner(messages, tools=tools, tool_choice=tool_choice)
        for call in (response.choices[0].message.tool_calls or []):
            self.calls.append({'name': call.function.name, 'arguments': call.function.arguments})
        return response

    @property
    def tool_names(self):
        return [call['name'] for call in self.calls]

    def queries(self):
        """Every free-text query the model sent to a retrieval tool."""
        found = []
        for call in self.calls:
            try:
                args = json.loads(call['arguments'])
            except ValueError:
                continue
            if isinstance(args.get('query'), str):
                found.append(args['query'])
            found.extend(r for r in args.get('requirements', []) if isinstance(r, str))
        return found


def scripted_provider():
    """A canned provider so the harness can be exercised without spending money."""
    def respond(messages, tools=None, tool_choice=None):
        question = next(m['content'] for m in messages if m['role'] == 'user')
        already_searched = any(m.get('role') == 'tool' for m in messages)
        wants_requirements = 'requirement' in question.lower() or 'job description' in question.lower()
        greeting = question.strip().lower() in ('hello', 'thanks!', 'hi', 'hey')
        if greeting:
            return _message('Hello. Ask me about documented career experience.')
        if not already_searched and tool_choice != 'none':
            name = 'find_requirement_evidence' if wants_requirements else 'search_profile'
            arguments = (json.dumps({'requirements': ['Java', 'Kafka']}) if wants_requirements
                         else json.dumps({'query': question[:200]}))
            return _message(None, [SimpleNamespace(id='selfcheck', function=SimpleNamespace(
                name=name, arguments=arguments))])
        return _message('Documented evidence indicates relevant experience. [S1]')
    return respond


def _message(content, tool_calls=None):
    return SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=content, tool_calls=tool_calls))])


def check_case(case, result, model, knowledge_base):
    """Mechanical assertions only. Returns a list of {check, passed, detail}."""
    answer = result['answer']
    lowered = answer.lower()
    cited = '[s' in lowered
    checks = []

    def add(name, passed, detail=''):
        checks.append({'check': name, 'passed': bool(passed), 'detail': detail})

    if case.get('expect_tool'):
        wanted = case['expect_tool']
        add('tool_selection', wanted in model.tool_names,
            f"wanted {wanted}, called {model.tool_names or 'nothing'}")

    if case.get('expect_citation'):
        add('cites_a_source', cited and result['sources'],
            f"{len(result['sources'])} source(s), cited={cited}")
        missing = [t for t in case.get('expect_terms', []) if t.lower() not in lowered]
        if case.get('expect_terms'):
            add('grounded_terms', not missing, f'missing {missing}' if missing else 'all present')

    if case.get('expect_abstention'):
        abstained = any(marker in lowered for marker in ABSTENTION_MARKERS)
        add('abstains', abstained, 'answered instead of flagging the gap' if not abstained else '')

    if case.get('expect_refusal'):
        add('refuses', answer == PRIVACY_RESPONSE, '' if answer == PRIVACY_RESPONSE else 'did not return the refusal')
        add('no_provider_call', model.invocations == 0, f'{model.invocations} model call(s)')

    if case.get('expect_no_tools'):
        add('no_tools_used', not model.tool_names, f'called {model.tool_names}')

    add('within_budget', result['rounds'] <= MAX_TOOL_ROUNDS, f"{result['rounds']} round(s)")

    # Diagnostic, not a pass/fail: did the model's own phrasing retrieve anything?
    queries = model.queries()
    barren = [q for q in queries if not knowledge_base.search(q)]
    return checks, {'queries': queries, 'queries_with_no_hits': barren}


def run_case(case, provider, knowledge_base):
    model = RecordingModel(provider)
    error = None
    try:
        result = run_agent(case['question'], [], model)
    except Exception as exc:  # noqa: BLE001 - a provider failure is a result, not a crash
        error = f'{type(exc).__name__}: {exc}'
        result = {'answer': '', 'sources': {}, 'actions': [], 'rounds': 0}
    checks, diagnostics = check_case(case, result, model, knowledge_base)
    if error:
        checks = [{'check': 'provider_call', 'passed': False, 'detail': error}]
    return {'id': case['id'], 'group': case['group'], 'question': case['question'],
            'answer': result['answer'], 'sources': result['sources'], 'actions': result['actions'],
            'rounds': result['rounds'], 'model_calls': model.invocations,
            'tools_called': model.tool_names, 'checks': checks, **diagnostics,
            'passed': all(c['passed'] for c in checks), 'error': error}


def report(results):
    print(f"{'group':<15} {'cases':>5} {'passed':>7}")
    print('-' * 30)
    for group in GROUP_ORDER:
        rows = [r for r in results if r['group'] == group]
        if rows:
            print(f"{group:<15} {len(rows):>5} {sum(r['passed'] for r in rows):>7}")
    print('-' * 30)
    print(f"{'ALL':<15} {len(results):>5} {sum(r['passed'] for r in results):>7}")

    failures = [r for r in results if not r['passed']]
    if failures:
        print(f'\n{len(failures)} failing case(s):')
        for row in failures:
            print(f"  {row['id']} ({row['group']}) — {row['question'][:60]}")
            for check in row['checks']:
                if not check['passed']:
                    print(f"      {check['check']}: {check['detail']}")

    barren = [(r['id'], q) for r in results for q in r['queries_with_no_hits']]
    if barren:
        print(f'\n{len(barren)} model-written query/queries retrieved nothing:')
        for case_id, query in barren:
            print(f'  {case_id}: {query[:70]}')

    total_calls = sum(r['model_calls'] for r in results)
    print(f'\n{total_calls} provider calls across {len(results)} cases.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-check', action='store_true',
                        help='run against a canned provider to verify the harness; costs nothing')
    parser.add_argument('--group', choices=GROUP_ORDER, help='run one group only')
    parser.add_argument('--limit', type=int, help='run at most this many cases')
    parser.add_argument('--no-save', action='store_true', help='do not write a run file')
    args = parser.parse_args()

    with CASES_PATH.open(encoding='utf-8') as handle:
        cases = [json.loads(line) for line in handle if line.strip()]
    if args.group:
        cases = [c for c in cases if c['group'] == args.group]
    if args.limit:
        cases = cases[:args.limit]

    if args.self_check:
        provider = scripted_provider()
        print('HARNESS SELF-CHECK — canned provider. Case outcomes are NOT an evaluation.\n')
    else:
        from twin.provider import call_genai
        provider = call_genai
        print(f'Live run against the provider: {len(cases)} cases.\n')

    knowledge_base = load_knowledge_base()
    results = [run_case(case, provider, knowledge_base) for case in cases]
    report(results)

    if not args.no_save and not args.self_check:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        path = RUNS_DIR / f'agent-{stamp}.json'
        path.write_text(json.dumps({'cases': results}, indent=2) + '\n', encoding='utf-8')
        print(f'\nSaved {path} for a later faithfulness pass.')

    if args.self_check:
        return 0
    return 0 if all(r['passed'] for r in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
