"""Offline retrieval eval: no provider calls, no cost, runnable on every commit.

Cases assert required substrings rather than passage labels, because chunk source
labels ("summary.txt, section 7, passage 2") shift whenever the documents are
edited. Substrings survive re-chunking, and they match what the agent actually
needs to see.

    python eval/run_retrieval_eval.py                 # report
    python eval/run_retrieval_eval.py --check         # compare against the baseline
    python eval/run_retrieval_eval.py --write-baseline
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from twin.retrieval import load_knowledge_base  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
CASES_PATH = EVAL_DIR / 'cases' / 'retrieval_cases.jsonl'
BASELINE_PATH = EVAL_DIR / 'baselines' / 'retrieval.json'
GROUP_ORDER = ['direct', 'synonym', 'multihop', 'negative']
# A drop larger than this against the baseline fails --check.
TOLERANCE = 0.01


def load_cases(path=CASES_PATH):
    with path.open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def score_case(case, knowledge_base, k):
    """Return the per-case result. `hit` is the headline pass/fail."""
    ranked = knowledge_base.ranked(case['question'])
    top = [passage for _, passage in ranked[:k]]

    if case.get('expect_empty'):
        return {'id': case['id'], 'group': case['group'], 'hit': not ranked,
                'hit_at_1': not ranked, 'reciprocal_rank': 0.0,
                'retrieved': len(ranked), 'missing': []}

    expected = case['expect_all']
    haystack = '\n'.join(p.text for p in top).lower()
    missing = [term for term in expected if term.lower() not in haystack]

    first_relevant = next((rank for rank, p in enumerate(top, 1)
                           if any(term.lower() in p.text.lower() for term in expected)), 0)
    top_one = top[:1] and any(term.lower() in top[0].text.lower() for term in expected)

    return {'id': case['id'], 'group': case['group'], 'hit': not missing,
            'hit_at_1': bool(top_one), 'reciprocal_rank': 1 / first_relevant if first_relevant else 0.0,
            'retrieved': len(ranked), 'missing': missing}


def summarize(results):
    def rate(rows, key):
        return round(sum(row[key] for row in rows) / len(rows), 4) if rows else 0.0

    summary = {'overall': {'cases': len(results),
                           'hit_rate': rate(results, 'hit'),
                           'hit_at_1': rate(results, 'hit_at_1'),
                           'mrr': rate(results, 'reciprocal_rank')},
               'groups': {}}
    for group in GROUP_ORDER:
        rows = [r for r in results if r['group'] == group]
        if rows:
            summary['groups'][group] = {'cases': len(rows), 'hit_rate': rate(rows, 'hit'),
                                        'hit_at_1': rate(rows, 'hit_at_1'), 'mrr': rate(rows, 'reciprocal_rank')}
    return summary


def report(results, summary, k):
    failures = [r for r in results if not r['hit']]
    print(f'Retrieval eval — k={k}, {len(results)} cases\n')
    print(f"{'group':<10} {'cases':>5} {'hit@k':>7} {'hit@1':>7} {'MRR':>7}")
    print('-' * 40)
    for group, stats in summary['groups'].items():
        print(f"{group:<10} {stats['cases']:>5} {stats['hit_rate']:>7.2f} "
              f"{stats['hit_at_1']:>7.2f} {stats['mrr']:>7.2f}")
    print('-' * 40)
    overall = summary['overall']
    print(f"{'ALL':<10} {overall['cases']:>5} {overall['hit_rate']:>7.2f} "
          f"{overall['hit_at_1']:>7.2f} {overall['mrr']:>7.2f}")

    if failures:
        print(f'\n{len(failures)} failing case(s):')
        for row in failures:
            detail = f"missing {row['missing']}" if row['missing'] else f"expected no hits, got {row['retrieved']}"
            print(f"  {row['id']:<5} ({row['group']}) {detail}")


def check_against_baseline(summary):
    if not BASELINE_PATH.exists():
        print(f'\nNo baseline at {BASELINE_PATH}. Run with --write-baseline first.')
        return 1
    baseline = json.loads(BASELINE_PATH.read_text(encoding='utf-8'))
    regressions = []
    for metric, value in summary['overall'].items():
        if metric == 'cases':
            continue
        previous = baseline['overall'].get(metric, 0)
        if value < previous - TOLERANCE:
            regressions.append(f'  overall {metric}: {previous:.2f} -> {value:.2f}')
    for group, stats in summary['groups'].items():
        previous = baseline['groups'].get(group, {}).get('hit_rate', 0)
        if stats['hit_rate'] < previous - TOLERANCE:
            regressions.append(f'  {group} hit_rate: {previous:.2f} -> {stats["hit_rate"]:.2f}')
    if regressions:
        print('\nREGRESSION against baseline:')
        print('\n'.join(regressions))
        return 1
    print('\nNo regression against baseline.')
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--k', type=int, default=4, help='passages per query (default: the app 4)')
    parser.add_argument('--check', action='store_true', help='fail if metrics dropped below the baseline')
    parser.add_argument('--write-baseline', action='store_true', help='record current metrics as the baseline')
    parser.add_argument('--json', type=Path, help='write the full result to this path')
    args = parser.parse_args()

    knowledge_base = load_knowledge_base()
    cases = load_cases()
    results = [score_case(case, knowledge_base, args.k) for case in cases]
    summary = summarize(results)
    report(results, summary, args.k)

    payload = {'k': args.k, 'overall': summary['overall'], 'groups': summary['groups'], 'cases': results}
    if args.json:
        args.json.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
        print(f'\nWrote {args.json}')
    if args.write_baseline:
        BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
        BASELINE_PATH.write_text(json.dumps({'k': args.k, 'overall': summary['overall'],
                                             'groups': summary['groups']}, indent=2) + '\n', encoding='utf-8')
        print(f'\nWrote baseline {BASELINE_PATH}')
    return check_against_baseline(summary) if args.check else 0


if __name__ == '__main__':
    raise SystemExit(main())
