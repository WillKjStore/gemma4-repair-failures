"""Recompute the final review tables from the ZIP's pseudonymous rows."""

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

PASS = 'pass'
KNOWN = {PASS, 'test_failure', 'context_limit_before_patch'}
GROUP = ('model', 'arm', 'cohort', 'seed', 'execution_mode')


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def summary(rows):
    groups = defaultdict(Counter)
    for row in rows:
        group = groups[tuple(row[k] for k in GROUP)]
        status, outcome = row['attempt_status'], row['outcome']
        group['planned_slots'] += 1
        group['tasks_attempted'] += status == 'attempted'
        group['pending_or_unstarted'] += status != 'attempted'
        if status == 'attempted':
            group['in_progress'] += outcome == 'pending_or_running'
            group['verified_repairs'] += outcome == PASS
            group['test_failures'] += outcome == 'test_failure'
            group['context_limits'] += outcome == 'context_limit_before_patch'
            group['incomplete_verifications'] += outcome == 'incomplete_verification'
            group['other_incomplete'] += outcome.startswith('incomplete') and outcome != 'incomplete_verification'
            group['nonrepair_measurements'] += outcome.startswith('file_prediction_')
    fields = ('planned_slots', 'tasks_attempted', 'verified_repairs', 'test_failures', 'context_limits',
              'incomplete_verifications', 'other_incomplete', 'pending_or_unstarted', 'in_progress',
              'nonrepair_measurements')
    return [{**dict(zip(GROUP, key)), **{k: str(value[k]) for k in fields}}
            for key, value in sorted(groups.items())]


def compare(left, right):
    paired = [(left[k], right[k]) for k in sorted(left.keys() & right.keys())]
    known = [(a, b) for a, b in paired if a in KNOWN and b in KNOWN]
    return {'pairs': len(paired), 'classifiable': len(known),
            'left_verified': sum(a == PASS for a, _ in paired),
            'right_verified': sum(b == PASS for _, b in paired),
            'both_pass': sum(a == b == PASS for a, b in known),
            'left_only_pass': sum(a == PASS and b != PASS for a, b in known),
            'right_only_pass': sum(a != PASS and b == PASS for a, b in known),
            'category_agreement': sum(a == b for a, b in known),
            'left_minus_right_bounds': [
                sum((a == PASS) - (b == PASS or b not in KNOWN) for a, b in paired),
                sum((a == PASS or a not in KNOWN) - (b == PASS) for a, b in paired)]}


def report(rows, ledger):
    def choose(model, arm, cohorts, seed='20261109'):
        selected = [r for r in rows if r['model'] == model and r['arm'] == arm
                    and r['cohort'] in cohorts and r['seed'] == seed]
        result = {r['task_key']: r['outcome'] for r in selected}
        assert len(result) == len(selected)
        return result

    core = {'matched_core32', 'matched_extension10'}
    e4b = choose('e4b', 'graph_on', core)
    e2b = choose('e2b', 'graph_on', core)
    larger = choose('31b', 'graph_on', {'matched42'})
    assert len(e4b) == len(e2b) == len(larger) == 42
    comparisons = {
        'e4b_vs_e2b': compare(e4b, e2b),
        'e4b_graph_on_vs_g0': compare(e4b, choose('e4b', 'graph_off', {'matched_core32'})),
        '31b_graph_on_vs_g0': compare(larger, choose('31b', 'graph_off', {'matched42'})),
        '31b_vs_e4b': compare(larger, e4b),
        'e4b_default_vs_e6': compare(e4b, choose('e4b', 'notes_e6', {'matched42'})),
        'original_e4b_repeat': compare(e4b, choose('e4b', 'graph_on', {'repeat_original'})),
        'additional_e4b_repeat': compare(e4b, choose('e4b', 'graph_on', {'repeat_additional'})),
        'additional_e2b_repeat': compare(e2b, choose('e2b', 'graph_on', {'repeat_additional'})),
    }
    repeats = {}
    for model, original in [('e4b', e4b), ('e2b', e2b), ('31b', larger)]:
        seeds = ['20261109', '20261112'] + (['20261113'] if model != '31b' else [])
        outcomes = [original] + [choose(model, 'graph_on', {'matched42'}, s) for s in seeds[1:]]
        assert all(x.keys() == original.keys() for x in outcomes)
        repeats[model] = {'verified_by_seed': dict(zip(seeds, [sum(v == PASS for v in x.values()) for x in outcomes])),
                          'task_pass_count_histogram': {str(k): v for k, v in sorted(Counter(
                              sum(x[key] == PASS for x in outcomes) for key in original).items())}}
    return {
        'row_counts': {'outcomes': len(rows), 'consolidated': len(summary(rows)), 'screening': len(ledger)},
        'screening_by_repo_and_status': {
            field: dict(sorted(Counter(f"{r['repo']}/{r[field]}" for r in ledger).items()))
            for field in ('original_status', 'final_status')},
        'screening_controls': dict(Counter(r['rescue_attempted'] for r in ledger)),
        'paired_comparisons': comparisons, 'seed_repeats': repeats,
    }


def main():
    directory = Path(__file__).resolve().parent
    manifest = {}
    for line in (directory / 'SHA256SUMS.txt').read_text(encoding='utf-8').splitlines():
        digest, name = line.split('  ', 1)
        assert Path(name).name == name and name not in manifest
        manifest[name] = digest
    for name, digest in manifest.items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest, name
    rows = read_csv(directory / 'task_level_outcomes.csv')
    ledger = read_csv(directory / 'screening_ledger.csv')
    assert summary(rows) == read_csv(directory / 'consolidated_results.csv')
    result = report(rows, ledger)
    assert result == json.loads((directory / 'recompute_output.json').read_bytes())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
