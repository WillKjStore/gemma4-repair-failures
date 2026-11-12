"""Run the frozen graph policy against issue-only manifests inside Quartz."""

import argparse
import json
from pathlib import Path

from swegemma.budget import EvaluationBudget
from swegemma.context import SwegemmaContext
from swegemma.models.task import Task
from swegemma.tools import graph

from query_policy import MAX_QUERIES, replay


ALLOWED_FIELDS = {'instance_id', 'repo', 'base_commit', 'problem_statement'}
TOOLS = {name: getattr(graph, name) for name in ('search_similar_code', 'get_code_neighbors', 'get_code_subgraph')}


def run(inputs, output):
    output.mkdir(mode=0o700)
    with (inputs / 'issues.jsonl').open(encoding='utf-8') as stream:
        rows = [json.loads(line) for line in stream]
    for index, row in enumerate(rows, 1):
        if row.keys() != ALLOWED_FIELDS or not row['problem_statement']:
            raise ValueError('unexpected replay manifest fields')
        task = Task(**row)
        context = SwegemmaContext(task=task, repo=task.repo, task_id=task.instance_id,
                                  problem_statement=task.problem_statement, graph_dir=str(inputs / 'graphs'),
                                  embeddings_dir=str(inputs / 'embeddings'),
                                  budget=EvaluationBudget(tool_calls=MAX_QUERIES, time_minutes=None))
        calls = []

        def invoke(name, arguments):
            response = TOOLS[name](context, **arguments)
            calls.append({'tool': name, 'arguments': arguments, 'response': response})
            return response

        status = 'complete'
        try:
            replay(task.problem_statement, invoke)
        except (ValueError, TypeError, KeyError, AttributeError, OSError, RuntimeError):
            status = 'replay_error'
        with (output / f'replay-{index:03}.json').open('x', encoding='utf-8') as stream:
            json.dump({'task_id': task.instance_id, 'repo': task.repo, 'base_commit': task.base_commit,
                       'status': status, 'calls': calls}, stream, default=str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    run(args.inputs, args.output)
