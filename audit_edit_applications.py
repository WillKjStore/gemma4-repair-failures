"""Count explicitly observed edit tool replies across the frozen saved traces."""
import json
from collections import Counter,defaultdict
from pathlib import Path

from label_wed_provisional_v3 import ROOT, attempts, new_attempts
from overnight_trace_audit_v2 import check,read,records,trace_path
from trace_labels_v2 import tool_observations
from wed_trace_evidence import task_key

FROZEN = ROOT/'run-wed-results-freeze/final-labels/runs.csv'
TOOLS = {'edit_file','write_file'}


def classify(raw):
    value = raw
    if isinstance(raw,str):
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = None
    if isinstance(value,dict):
        status = value.get('status')
        if status=='ok':
            return 'success',''
        if status=='error':
            return 'failure',reason(json.dumps(value))
        if value.get('error'):
            return 'failure',reason(json.dumps(value['error']),fallback='error_only_reply')
    if isinstance(raw,str) and raw.lstrip().startswith(('FileEditError:','FileWriteError:')):
        return 'failure',reason(raw)
    return 'unknown',''


def reason(text, fallback='other_explicit_error'):
    normalized = text.casefold()
    for marker in ('old_string not found','old_string is not unique','multiple occurrences',
                   'file not found','permission denied'):
        if marker in normalized:
            return marker
    return fallback


def replies(trace):
    calls = {call['tool_call_id']:call['function_name'] for step in trace['steps']
             for call in step.get('tool_calls') or () if call['function_name'] in TOOLS}
    observed = defaultdict(set)
    for call,raw,_ in tool_observations(trace):
        key = call['tool_call_id']
        if key in calls:
            observed[key].add(classify(raw))
    statuses = defaultdict(Counter)
    reasons = Counter()
    for key,tool in calls.items():
        known = {entry for entry in observed[key] if entry[0]!='unknown'}
        states = {entry[0] for entry in known}
        state = next(iter(states)) if len(states)==1 else 'unknown'
        statuses[tool][state]+=1
        if tool=='edit_file' and state=='failure':
            errors = {category for _,category in known}
            reasons[next(iter(errors)) if len(errors)==1 else 'other_explicit_error']+=1
    return statuses,reasons


def main(output):
    check(output.parent)
    frozen = {row['run_id']:row for row in records(FROZEN) if row['trace_saved']=='True'}
    covered = set()
    aggregates = defaultdict(Counter)
    all_reasons = Counter()
    first_trace_confirmed = False
    for attempt in (*attempts(),*new_attempts()):
        key = f"{attempt['job_id']}:{attempt['slot'] or '0'}:{task_key(attempt['task_id'])}"
        if key not in frozen:
            continue
        if key in covered:
            raise ValueError('duplicate frozen trace identity')
        covered.add(key)
        row = frozen[key]
        trace = read(trace_path(attempt['directory']))
        status,reasons = replies(trace)
        edit,write = status['edit_file'],status['write_file']
        all_reasons.update(reasons)
        if attempt['job_id']=='12764570':
            first_trace_confirmed = any(call['function_name']=='edit_file' and str(step)=='6'
                                        and classify(raw)==('failure','old_string not found')
                                        for call,raw,step in tool_observations(trace))
        for group in ('all',row['model'],row['model']+'/'+row['outcome']):
            counts = aggregates[group]
            counts['saved_traces']+=1
            counts['edit_file_calls']+=edit.total()
            counts['edit_file_failures']+=edit['failure']
            counts['edit_file_successes']+=edit['success']
            counts['edit_file_unknown_replies']+=edit['unknown']
            counts['runs_with_edit_file_call']+=bool(edit)
            counts['runs_with_edit_failure']+=bool(edit['failure'])
            counts['runs_with_failure_and_edit_success']+=bool(edit['failure'] and edit['success'])
            counts['runs_with_failure_and_all_edit_calls_failed']+=bool(edit['failure'] and not edit['success'] and not edit['unknown'])
            counts['runs_with_failure_no_edit_success_and_unknown_reply']+=bool(edit['failure'] and not edit['success'] and edit['unknown'])
            counts['runs_with_all_edits_failed_and_successful_write']+=bool(edit['failure'] and not edit['success'] and not edit['unknown'] and write['success'])
            counts['runs_with_all_edits_failed_and_no_observed_successful_write']+=bool(edit['failure'] and not edit['success'] and not edit['unknown'] and not write['success'])
    if covered!=frozen.keys() or not first_trace_confirmed:
        raise ValueError('frozen trace membership or reported step could not be reconciled')
    report = {'groups':aggregates,'failure_reasons':all_reasons,'reported_step_confirmed':first_trace_confirmed,
              'scope':'All frozen saved repair traces, including repeats, G0 and larger-model diagnostics. Never succeeded means all edit_file calls had explicit failure replies. A success is any successful edit_file in the same trace, on any file. Shell edits and submitted patch correctness are outside this measure.'}
    with output.open('x',encoding='utf-8') as stream:
        json.dump(report,stream,indent=2,sort_keys=True)
    print(json.dumps({'all':aggregates['all'],'failure_reasons':all_reasons}))


if __name__=='__main__':
    import sys
    main(Path(sys.argv[1]))
