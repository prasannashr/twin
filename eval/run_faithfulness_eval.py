"""Faithfulness pass: does the cited passage actually support the claim?

The graph validates that a citation ID exists in the request's evidence. It
cannot tell whether the passage backs the sentence it is attached to, so a real
source can be cited for a claim it does not support. This scores that gap.

It reads a saved run from eval/runs/, so generation is not paid for twice; only
the judge calls cost anything.

A judge model is itself unreliable. Treat the output as a triage queue, not a
score: read the unsupported claims it prints and decide for yourself. Spot-check
a handful of "supported" verdicts too, or the judge is rubber-stamping unchecked.

    python eval/run_faithfulness_eval.py --self-check   # free; verifies the harness
    python eval/run_faithfulness_eval.py                # judges the latest run
    python eval/run_faithfulness_eval.py --run eval/runs/agent-....json
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

EVAL_DIR = Path(__file__).resolve().parent
RUNS_DIR = EVAL_DIR / 'runs'

CITATION = re.compile(r'\[(S\d+)\]')
# Text followed by the citation run that closes it. Splitting on sentence
# boundaries instead orphans the citation, because answers end lines as
# "...per second**. [S1]" and the marker lands in its own fragment.
CITED_SPAN = re.compile(r'(?P<text>[^\n]*?)(?P<ids>(?:\s*\[S\d+\])+)')
VERDICTS = ('supported', 'partial', 'unsupported')

JUDGE_SYSTEM = """
You check whether a source passage supports a claim. You are given one claim and
the passage(s) it cites.

Judge ONLY against the supplied passage text. Ignore anything you know from
outside it. A claim is not supported merely because it sounds plausible.

Reply with JSON and nothing else:
{"verdict": "supported" | "partial" | "unsupported", "reason": "<one short sentence>"}

supported   - the passage states the claim, or the claim follows directly from it.
partial     - the passage is related but does not establish the whole claim.
unsupported - the passage does not establish the claim.

A claim that only reports an absence of evidence ("the profile does not mention
X") is supported when the passage indeed does not contain X.
""".strip()


def extract_claims(answer, sources):
    """Each cited span, paired with the passages it cites.

    A claim is the text running up to a citation run, which is what that
    citation is standing behind. For bullet answers that is the bullet; in a
    paragraph it is the text since the previous citation.
    """
    claims = []
    for line in answer.splitlines():
        line = line.strip()
        if not line:
            continue
        for match in CITED_SPAN.finditer(line):
            ids = CITATION.findall(match.group('ids'))
            text = match.group('text').strip().lstrip('-*\u2022 ').replace('**', '').strip(' .,;:')
            cited = [{'id': i, **sources[i]} for i in dict.fromkeys(ids) if i in sources]
            if len(text) > 15 and cited:
                claims.append({'claim': text, 'cited_ids': [c['id'] for c in cited],
                               'passages': cited})
    return claims


def judge_claim(claim, provider):
    evidence = '\n\n'.join(f"[{p['id']}] {p['source']}\n{p['text']}" for p in claim['passages'])
    messages = [{'role': 'system', 'content': JUDGE_SYSTEM},
                {'role': 'user', 'content': f"CLAIM:\n{claim['claim']}\n\nPASSAGE(S):\n{evidence}"}]
    raw = provider(messages).choices[0].message.content or ''
    match = re.search(r'\{.*\}', raw, re.S)
    if not match:
        return {'verdict': 'unparsed', 'reason': raw[:120]}
    try:
        parsed = json.loads(match.group(0))
    except ValueError:
        return {'verdict': 'unparsed', 'reason': raw[:120]}
    verdict = str(parsed.get('verdict', '')).lower().strip()
    return {'verdict': verdict if verdict in VERDICTS else 'unparsed',
            'reason': str(parsed.get('reason', ''))[:200]}


def scripted_judge():
    """Canned judge so the harness can be exercised without spending money."""
    def respond(messages, model=None, tools=None, tool_choice=None):
        from types import SimpleNamespace
        text = messages[-1]['content']
        claim = text.split('PASSAGE(S):')[0]
        passage = text.split('PASSAGE(S):')[-1]
        # Crude token overlap, only so the self-check exercises each verdict branch.
        words = {w.lower() for w in re.findall(r'[A-Za-z0-9,]{4,}', claim)}
        hits = sum(1 for w in words if w in passage.lower())
        ratio = hits / max(1, len(words))
        verdict = 'supported' if ratio > 0.5 else 'partial' if ratio > 0.25 else 'unsupported'
        body = json.dumps({'verdict': verdict, 'reason': f'self-check overlap {ratio:.2f}'})
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=body))])
    return respond


def latest_run():
    runs = sorted(RUNS_DIR.glob('agent-*.json'))
    if not runs:
        raise SystemExit(f'No saved runs in {RUNS_DIR}. Run eval/run_agent_eval.py first.')
    return runs[-1]


def report(rows, structural_errors):
    counts = {v: sum(1 for r in rows if r['verdict'] == v) for v in (*VERDICTS, 'unparsed')}
    total = len(rows)
    print(f'{total} cited claim(s) judged\n')
    print(f"{'verdict':<14} {'count':>5} {'share':>7}")
    print('-' * 28)
    for verdict in (*VERDICTS, 'unparsed'):
        share = counts[verdict] / total if total else 0
        print(f'{verdict:<14} {counts[verdict]:>5} {share:>6.0%}')
    print('-' * 28)
    supported = counts['supported'] / total if total else 0
    print(f'faithfulness (supported / all): {supported:.0%}')

    flagged = [r for r in rows if r['verdict'] in ('partial', 'unsupported', 'unparsed')]
    if flagged:
        print(f'\n{len(flagged)} claim(s) to read yourself:')
        for row in flagged:
            print(f"\n  [{row['verdict']}] {row['case_id']} cites {row['cited_ids']}")
            print(f"    claim:  {row['claim'][:150]}")
            print(f"    judge:  {row['reason'][:150]}")

    if structural_errors:
        print(f'\n{len(structural_errors)} citation ID(s) not present in the run evidence:')
        for case_id, missing in structural_errors:
            print(f'  {case_id}: {missing}')
    else:
        print('\nEvery citation ID resolves to a passage in its request (structural check).')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, help='saved agent run to judge (default: newest)')
    parser.add_argument('--self-check', action='store_true', help='canned judge; costs nothing')
    parser.add_argument('--limit', type=int, help='judge at most this many claims')
    parser.add_argument('--no-save', action='store_true', help='do not write a verdict file')
    args = parser.parse_args()

    run_path = args.run or latest_run()
    data = json.loads(run_path.read_text(encoding='utf-8'))

    claims, structural_errors = [], []
    for case in data['cases']:
        missing = [i for i in set(CITATION.findall(case['answer'])) if i not in case['sources']]
        if missing:
            structural_errors.append((case['id'], missing))
        for claim in extract_claims(case['answer'], case['sources']):
            claims.append({'case_id': case['id'], **claim})
    if args.limit:
        claims = claims[:args.limit]

    if args.self_check:
        provider = scripted_judge()
        print('HARNESS SELF-CHECK - canned judge. Verdicts are NOT an evaluation.\n')
    else:
        from twin.provider import call_genai
        provider = call_genai
        print(f'Judging {len(claims)} claims from {run_path.name} ({len(claims)} judge calls).\n')

    rows = []
    for claim in claims:
        verdict = judge_claim(claim, provider)
        rows.append({**{k: claim[k] for k in ('case_id', 'claim', 'cited_ids')}, **verdict})
    report(rows, structural_errors)

    print('\nA judge model is not ground truth. Read the flagged claims, and spot-check\n'
          'a few "supported" verdicts, before trusting this number.')

    if not args.no_save and not args.self_check:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        path = RUNS_DIR / f'faithfulness-{stamp}.json'
        path.write_text(json.dumps({'run': run_path.name, 'verdicts': rows}, indent=2) + '\n',
                        encoding='utf-8')
        print(f'\nSaved {path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
