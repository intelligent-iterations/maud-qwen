"""Deterministic scoring with fixed task and label universes."""
from collections import Counter, defaultdict
import math

import numpy as np


def mean(values):
    return float(np.mean(values)) if values else None


def validate_predictions(rows, predictions, schema):
    truth = {r['example_id']: r for r in rows}
    assert len(truth) == len(rows), 'Duplicate gold example ID'
    ids = [p['example_id'] for p in predictions]
    if len(set(ids)) != len(ids) or set(ids) != set(truth):
        raise ValueError('Missing, duplicate or foreign prediction IDs')
    for p in predictions:
        r = truth[p['example_id']]
        for key in ('agreement_id', 'task_id', 'label', 'split'):
            if p[key] != r[key]:
                raise ValueError(f'Prediction metadata differs for {key}: {p["example_id"]}')
        label = p['prediction']
        if type(label) is not int or label < -1 or label >= len(schema[r['task_id']]['answers']):
            raise ValueError('Out-of-range predicted label')
    return [next_p for next_p in sorted(predictions, key=lambda x: x['example_id'])]


def task_metrics(rows, answers):
    n = len(rows)
    per_label = []
    for i, answer in enumerate(answers):
        tp = sum(r['label'] == i and r['prediction'] == i for r in rows)
        fp = sum(r['label'] != i and r['prediction'] == i for r in rows)
        fn = sum(r['label'] == i and r['prediction'] != i for r in rows)
        per_label.append({'label': i, 'answer': answer, 'support': tp+fn,
            'precision': tp/(tp+fp) if tp+fp else 0.0,
            'recall': tp/(tp+fn) if tp+fn else 0.0,
            'f1': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0})
    confusion = Counter((r['label'],r['prediction']) for r in rows)
    return {'n': n, 'accuracy': sum(r['label'] == r['prediction'] for r in rows)/n,
            'macro_f1': mean([v['f1'] for v in per_label]),
            'invalid_output_rate': sum(r['prediction'] == -1 for r in rows)/n, 'per_label': per_label,
            'confusions': [{'label':y,'prediction':p,'n':count} for (y,p),count in sorted(confusion.items())]}


def score(predictions, schema, coverage, strata=True):
    if not predictions:
        return {'n': 0, 'question_macro_f1': None, 'accuracy': None, 'invalid_output_rate': None,
                'eligible_tasks_scored': 0, 'missing_eligible_tasks': [k for k,v in coverage.items() if v['eligible']]}
    by_task = defaultdict(list)
    for row in predictions:
        by_task[row['task_id']].append(row)
    per_task = {tid: {**task_metrics(rows, schema[tid]['answers']), 'eligible': coverage[tid]['eligible'],
                       'question': schema[tid]['question'], 'subquestion': schema[tid]['subquestion']} for tid, rows in sorted(by_task.items())}
    eligible = [tid for tid in per_task if coverage[tid]['eligible']]
    rare_rows = [r for r in predictions if r['label'] in coverage[r['task_id']]['rare_labels']]
    rare_class_recall = [v['recall'] for tid, result in per_task.items() for v in result['per_label']
                         if v['support'] and v['label'] in coverage[tid]['rare_labels']]
    parents = defaultdict(list)
    for tid in eligible:
        parents[schema[tid]['question']].append(per_task[tid]['macro_f1'])
    result = {'n': len(predictions), 'question_macro_f1': mean([per_task[t]['macro_f1'] for t in eligible]),
              'parent_question_macro_f1': mean([mean(v) for v in parents.values()]),
              'accuracy': sum(r['label'] == r['prediction'] for r in predictions)/len(predictions),
              'invalid_output_rate': sum(r['prediction'] == -1 for r in predictions)/len(predictions),
              'eligible_tasks_scored': len(eligible),
              'missing_eligible_tasks': sorted(t for t,v in coverage.items() if v['eligible'] and t not in by_task),
              'rare_answer_n': len(rare_rows),
              'rare_answer_accuracy': mean([r['label'] == r['prediction'] for r in rare_rows]),
              'rare_class_macro_recall': mean(rare_class_recall), 'per_question': per_task}
    categories = sorted({v['category'] for v in schema.values()})
    result['per_category'] = {}
    for category in categories:
        tids = [t for t in eligible if schema[t]['category'] == category]
        rs = [r for r in predictions if schema[r['task_id']]['category'] == category]
        result['per_category'][category] = {'question_macro_f1': mean([per_task[t]['macro_f1'] for t in tids]),
            'accuracy': mean([r['label'] == r['prediction'] for r in rs]), 'n': len(rs), 'eligible_tasks': len(tids)}
    if strata:
        for name, key in [('per_agreement', 'agreement_id'), ('per_data_type', 'data_type'), ('per_length_bucket', 'length_bucket')]:
            groups = defaultdict(list)
            for r in predictions:
                groups[r.get(key, 'unknown')].append(r)
            result[name] = {g: {k:v for k,v in score(rs, schema, coverage, False).items()
                                if k not in ('per_question', 'per_category')} for g,rs in sorted(groups.items())}
        if all('train_overlap' in r for r in predictions):
            result['without_train_overlap'] = {k:v for k,v in score([r for r in predictions if not r['train_overlap']], schema, coverage, False).items() if k not in ('per_question','per_category')}
    return result


def bootstrap(left, right, schema, coverage, replicates=2000, seed=42):
    lmap, rmap = {r['example_id']:r for r in left}, {r['example_id']:r for r in right}
    if lmap.keys() != rmap.keys() or len(lmap) != len(left) or len(rmap) != len(right):
        raise ValueError('Paired bootstrap needs identical, unique example IDs')
    for eid in lmap:
        if any(lmap[eid][k] != rmap[eid][k] for k in ('agreement_id', 'task_id', 'label')):
            raise ValueError('Paired metadata differs')
    groups = defaultdict(list)
    for r in left:
        groups[r['agreement_id']].append(r['example_id'])
    agreements = sorted(groups)
    rng = np.random.default_rng(seed)
    values, missing = [], 0
    # Build agreement/task confusion counts once; resample complete clusters.
    tasks = [t for t in schema if coverage[t]['eligible'] and any(r['task_id'] == t for r in left)]
    if not tasks or not agreements:
        raise ValueError('No eligible tasks or agreements for bootstrap')
    counts = []
    for preds in (lmap, rmap):
        mats = []
        for tid in tasks:
            n = len(schema[tid]['answers'])
            arr = np.zeros((len(agreements), n, n+1), dtype=np.int64)
            for ai, agreement in enumerate(agreements):
                for eid in groups[agreement]:
                    r = preds[eid]
                    if r['task_id'] == tid:
                        arr[ai, r['label'], r['prediction'] if r['prediction'] >= 0 else n] += 1
            mats.append(arr)
        counts.append(mats)
    for _ in range(replicates):
        weights = np.bincount(rng.integers(0, len(agreements), len(agreements)), minlength=len(agreements))
        pair = []
        omitted = False
        for mats in counts:
            fs = []
            for arr in mats:
                cm = np.tensordot(weights, arr, axes=(0,0))
                if cm.sum() == 0:
                    omitted = True
                    continue
                n = cm.shape[0]
                tp = np.diag(cm[:,:n])
                denom = cm.sum(1) + cm[:,:n].sum(0)
                fs.append(float(np.divide(2*tp, denom, out=np.zeros(n, dtype=float), where=denom != 0).mean()))
            pair.append(mean(fs))
        missing += omitted
        if all(v is not None for v in pair):
            values.append([pair[0], pair[1], pair[1]-pair[0]])
    v = np.asarray(values)
    return {'unit': 'agreement', 'paired': True, 'replicates': replicates, 'seed': seed,
            'agreements': len(agreements), 'valid_replicates': len(values), 'replicates_missing_tasks': missing,
            'left_ci95': np.percentile(v[:,0],[2.5,97.5]).tolist(),
            'right_ci95': np.percentile(v[:,1],[2.5,97.5]).tolist(),
            'difference_right_minus_left_ci95': np.percentile(v[:,2],[2.5,97.5]).tolist()}
