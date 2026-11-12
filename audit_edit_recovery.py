"""Measure ordered edit recovery and post-failure trace activity."""
import json
import math
from collections import Counter,defaultdict
from pathlib import Path
from statistics import median

from audit_edit_applications_v2 import FROZEN, attempts, check, classify, new_attempts, read, records, tool_observations, trace_path
from wed_trace_evidence import task_key


def arguments(call):
    value = next((call[field] for field in ('arguments','args','parameters') if field in call),None)
    if isinstance(value,str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if not isinstance(value,dict):
        return None
    path = value.get('filepath',value.get('path',value.get('file_path')))
    fields = (path,value.get('old_string'),value.get('new_string'))
    return fields if all(isinstance(part,str) for part in fields) else None


def wilson(successes,total):
    z = 1.959963984540054
    fraction = successes/total
    scale = 1+z*z/total
    center = (fraction+z*z/(2*total))/scale
    half = z*math.sqrt(fraction*(1-fraction)/total+z*z/(4*total*total))/scale
    return {'numerator':successes,'denominator':total,'rate':fraction,'wilson95':[center-half,center+half]}


def ordered(trace):
    steps = trace['steps']
    positions = {step['step_id']:index for index,step in enumerate(steps)}
    calls = {call['tool_call_id']:(index,arguments(call)) for index,step in enumerate(steps)
             for call in step.get('tool_calls') or () if call['function_name']=='edit_file'}
    outcomes = defaultdict(set)
    for call,raw,step in tool_observations(trace):
        if call['tool_call_id'] in calls:
            state,_ = classify(raw)
            if state!='unknown':
                outcomes[call['tool_call_id']].add((state,positions[step]))
    failures,successes = [],[]
    for key,(index,signature) in calls.items():
        options = outcomes[key]
        if len({state for state,_ in options})!=1:
            continue
        state,reply_index = min(options,key=lambda event:event[1])
        (failures if state=='failure' else successes).append((index,reply_index,signature))
    return calls,failures,successes


def post_failure(trace,calls,failures):
    first = min(reply for _,reply,_ in failures)
    retries = [(pos,signature) for pos,signature in calls.values() if pos>first]
    repeated = [pos for pos,signature in retries if signature is not None and
                any(original==signature and reply<pos for _,reply,original in failures)]
    after = trace['steps'][first+1:]
    tokens = [(step.get('metrics') or {}).get('completion_tokens') for step in after]
    saved_tokens = [value for value in tokens if isinstance(value,int) and value>=0]
    prompts = [(step.get('metrics') or {}).get('prompt_tokens') for step in after]
    prompts = [value for value in prompts if isinstance(value,int) and value>=0]
    retry_tokens = [(trace['steps'][pos].get('metrics') or {}).get('completion_tokens') for pos in set(repeated)]
    retry_tokens = [value for value in retry_tokens if isinstance(value,int) and value>=0]
    return {'later_agent_steps':sum(step.get('source')=='agent' for step in after),
            'later_edit_calls':len(retries),'identical_retries':len(repeated),
            'later_edit_arguments_unavailable':sum(signature is None for _,signature in retries),
            'completion_tokens_after_failure':sum(saved_tokens) if saved_tokens else None,
            'peak_recorded_prompt_after_failure':max(prompts) if prompts else None,
            'completion_tokens_on_identical_retry_steps':sum(retry_tokens) if retry_tokens else None}


def distribution(values):
    known = [value for value in values if value is not None]
    return {'observations':len(known),'median':median(known) if known else None,
            'min':min(known) if known else None,'max':max(known) if known else None}


def main(output):
    check(output.parent)
    frozen = {row['run_id']:row for row in records(FROZEN) if row['trace_saved']=='True'}
    covered = set()
    counts = defaultdict(Counter)
    context = defaultdict(list)
    for attempt in (*attempts(),*new_attempts()):
        key = f"{attempt['job_id']}:{attempt['slot'] or '0'}:{task_key(attempt['task_id'])}"
        if key not in frozen:
            continue
        assert key not in covered
        covered.add(key)
        row = frozen[key]
        model = row['model']
        trace = read(trace_path(attempt['directory']))
        calls,failures,successes = ordered(trace)
        group = counts[model]
        group['runs']+=1
        group['edit_calls']+=len(calls)
        group['failed_edit_calls']+=len(failures)
        group['unclassified_edit_calls']+=len(calls)-len(failures)-len(successes)
        if failures:
            group['failed_runs']+=1
            group['failed_runs_with_any_edit_success']+=bool(successes)
            group['failed_runs_with_later_success']+=any(call>reply for call,_,_ in successes for _,reply,_ in failures)
        if row['outcome']=='context_limit_before_patch':
            group['context_limit_runs']+=1
            group['context_limit_edit_calls']+=len(calls)
            group['context_limit_failed_edit_calls']+=len(failures)
            if failures:
                context[model].append(post_failure(trace,calls,failures))
    assert covered==frozen.keys()
    report = {}
    for model,c in counts.items():
        rows = context[model]
        report[model] = {'counts':dict(c),'call_failure':wilson(c['failed_edit_calls'],c['edit_calls']),
                         'ordered_recovery':wilson(c['failed_runs_with_later_success'],c['failed_runs']),
                         'context_after_first_failure':{'runs':len(rows),
                            'runs_with_later_edit_call':sum(row['later_edit_calls']>0 for row in rows),
                            'runs_with_identical_retry':sum(row['identical_retries']>0 for row in rows),
                            'runs_with_two_or_more_identical_retries':sum(row['identical_retries']>=2 for row in rows),
                            'runs_with_unavailable_later_edit_arguments':sum(
                                row['later_edit_arguments_unavailable']>0 for row in rows),
                            **{field:distribution([row[field] for row in rows])
                               for field in ('later_agent_steps','later_edit_calls','completion_tokens_after_failure',
                                             'peak_recorded_prompt_after_failure',
                                             'completion_tokens_on_identical_retry_steps')},
                            'completion_tokens_by_identical_retry':{
                                name:distribution([row['completion_tokens_after_failure'] for row in rows
                                                   if bool(row['identical_retries'])==is_retry])
                                for name,is_retry in [('retry_observed',True),('retry_not_observed',False)]}}}
    summary = {'saved_traces':len(covered),'models':report,
               'method':'Recovery requires a successful edit_file call in a later trace step than an explicit failed edit_file reply. It may edit another file. An identical retry repeats the exact file path, old_string and new_string of an earlier failed call. Later agent steps and recorded completion tokens describe the observed tail. Completion tokens are not window occupancy. Wilson intervals are descriptive and do not adjust for repeated tasks.'}
    with output.open('x',encoding='utf-8') as stream:
        json.dump(summary,stream,indent=2,sort_keys=True)
    print(json.dumps(summary,sort_keys=True))


if __name__=='__main__':
    import sys
    main(Path(sys.argv[1]))
