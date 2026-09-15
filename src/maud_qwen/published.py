"""Audit already-published predictions without loading a model or selecting one."""
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path

from .common import file_sha, read_json, read_jsonl, sha, verify_freeze, write_json
from .metrics import bootstrap, score, validate_predictions
from .prompt import parse_answer


def read_archive(entry):
    path = Path(entry['path'])
    if file_sha(path) != entry['gzip_sha256']:
        raise ValueError(f'Compressed prediction hash mismatch: {path}')
    raw = gzip.decompress(path.read_bytes())
    if sha(raw) != entry['sha256']:
        raise ValueError(f'Original JSONL hash mismatch: {path}')
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if len(rows) != entry['rows']:
        raise ValueError(f'Prediction row count mismatch: {path}')
    return rows


def verify_prediction_rows(rows, predictions, schema, overlap):
    predictions = validate_predictions(rows, predictions, schema)
    gold = {r['example_id']: r for r in rows}
    for prediction in predictions:
        row = gold[prediction['example_id']]
        expected = {
            'valid_answers': schema[row['task_id']]['answers'],
            'text_sha256': sha(row['text']),
            'data_type': row['data_type'],
            'train_overlap': row['example_id'] in overlap,
        }
        for key, value in expected.items():
            if prediction.get(key) != value:
                raise ValueError(f'Published {key} mismatch: {row["example_id"]}')
        if parse_answer(prediction['raw_output'], len(expected['valid_answers'])) != prediction['prediction']:
            raise ValueError(f'Raw answer does not match parsed label: {row["example_id"]}')
    return predictions


def rare_stats(predictions, coverage):
    gold = [r for r in predictions if r['label'] in coverage[r['task_id']]['rare_labels']]
    chosen = [r for r in predictions if r['prediction'] in coverage[r['task_id']]['rare_labels']]
    correct = sum(r['prediction'] == r['label'] for r in gold)
    return {'n': len(gold), 'correct': correct, 'recall': correct / len(gold) if gold else None,
            'precision': correct / len(chosen) if chosen else None, 'predictions': len(chosen),
            'incorrect_predictions': len(chosen) - correct}


def check_equal(actual, expected, label):
    if actual != expected:
        raise ValueError(f'Published result mismatch: {label}')


def main():
    published = Path('results/published')
    verify_freeze()
    schema = read_json('manifests/questions.json')
    coverage = read_json('manifests/coverage.json')
    flags = read_json('manifests/train_overlap.json')
    overlap = set(flags['exact']) | set(flags['near'])
    manifest = read_json(published / 'predictions.json')
    gold = {s: read_jsonl(f'data/heldout/{s}.jsonl') for s in ('train', 'validation', 'test')}
    final = read_json(published / 'final-test-verification.json')
    validation = read_json(published / 'all-checkpoints-validation-summary.json')
    weighted = read_json(published / 'weighted-vs-control-epoch3.json')
    selection = read_json(published / 'final_selection.json')
    check_equal(file_sha(published / 'final_selection.json'), final['selection_ledger']['selection_sha256'], 'selection hash')
    check_equal(file_sha('data/heldout/test.jsonl'), final['test_file_sha256'], 'test population')
    check_equal(selection['selected_epoch'], 5, 'recorded selected epoch')
    counts = defaultdict(Counter)
    for row in gold['train']:
        counts[row['task_id']][row['label']] += 1
    majority = {task: min(range(len(info['answers'])), key=lambda i: (-counts[task][i], i))
                for task, info in schema.items()}
    predictions, scores, rare = {}, {}, {}
    for name, entry in manifest.items():
        rows = verify_prediction_rows(gold[entry['split']], read_archive(entry), schema, overlap)
        if name.startswith('majority-'):
            check_equal([r['prediction'] for r in rows], [majority[r['task_id']] for r in rows], name + ' training counts')
        predictions[name], scores[name], rare[name] = rows, score(rows, schema, coverage), rare_stats(rows, coverage)
        print(f'{name}: {len(rows):,} rows; macro-F1={scores[name]["question_macro_f1"]:.4f}', flush=True)
    for system in ('majority', 'base', 'epoch5'):
        name = system + '-test'
        check_equal(manifest[name]['sha256'], final['systems'][system]['prediction_sha256'], name + ' original hash')
        check_equal(scores[name], final['systems'][system]['scores'], name + ' full metrics')
        check_equal(rare[name], final['systems'][system]['rare'], name + ' rare metrics')
    names = {'Base': 'base-validation', 'Weighted epoch 3': 'weighted3-validation',
             **{f'Epoch {i}': f'epoch{i}-validation' for i in range(1, 7)}}
    for row in validation['rows']:
        name = names[row['checkpoint']]
        for key in ('question_macro_f1', 'accuracy', 'rare_class_macro_recall'):
            check_equal(scores[name][key], row[key], name + ' ' + key)
        for source, key in (('correct', 'rare_correct'), ('n', 'rare_n'), ('recall', 'rare_recall'), ('precision', 'rare_precision')):
            check_equal(rare[name][source], row[key], name + ' ' + key)
    for name in names.values():
        left, right = predictions['base-validation'], predictions[name]
        check_equal([r['prompt_tokens_sha256'] for r in left], [r['prompt_tokens_sha256'] for r in right], name + ' prompt identity')
    check_equal([r['prompt_tokens_sha256'] for r in predictions['base-test']],
                [r['prompt_tokens_sha256'] for r in predictions['epoch5-test']], 'test prompt identity')
    check_equal(scores['epoch3-validation'], weighted['control_scores'], 'weighted control full metrics')
    check_equal(scores['weighted3-validation'], weighted['weighted_scores'], 'weighted full metrics')
    print('Recomputing paired agreement bootstrap intervals (2,000 draws each)...', flush=True)
    final_interval = bootstrap(predictions['base-test'], predictions['epoch5-test'], schema, coverage, 2000, 42)
    weighted_interval = bootstrap(predictions['epoch3-validation'], predictions['weighted3-validation'], schema, coverage, 2000, 42)
    check_equal(final_interval, final['primary_bootstrap'], 'final primary interval')
    check_equal(weighted_interval, weighted['bootstrap'], 'weighted primary interval')
    receipt = {'passed': True, 'prediction_sets': len(predictions),
               'prediction_rows': sum(len(r) for r in predictions.values()),
               'freeze_sha256': file_sha('manifests/freeze.json'),
               'selection_sha256': file_sha(published / 'final_selection.json'),
               'test_scores': {s: scores[s + '-test'] for s in ('majority', 'base', 'epoch5')},
               'validation_scores': {s: scores[s] for s in names.values()},
               'final_bootstrap': final_interval, 'weighted_bootstrap': weighted_interval,
               'model_loaded': False, 'new_training': False, 'purpose': 'Rescore already-published predictions'}
    write_json('outputs/reproduction/verification.json', receipt)
    print('PASS: published results reproduced. Receipt: outputs/reproduction/verification.json')


if __name__ == '__main__':
    main()
