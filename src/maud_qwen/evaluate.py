from collections import Counter, defaultdict
import json
from pathlib import Path
import time

from .common import file_sha, metadata, read_json, read_jsonl, sha, tracking, verify_freeze, write_json, write_jsonl
from .metrics import score, validate_predictions
from .prompt import parse_answer, prompt_ids


def prediction_record(row, prediction, raw_output, schema, overlap, **extra):
    n = extra.get('prompt_tokens', 0)
    bucket = 'unknown' if not n else '0-1024' if n <= 1024 else '1025-2048' if n <= 2048 else '2049-4096' if n <= 4096 else '4097+'
    return {k:row[k] for k in ('example_id', 'agreement_id', 'task_id', 'label', 'split', 'data_type')} | {
        'prediction': prediction, 'raw_output': raw_output, 'valid_answers': schema[row['task_id']]['answers'],
        'text_sha256': sha(row['text']), 'train_overlap': row['example_id'] in overlap,
        'length_bucket': bucket, **extra}


def load_eval_data(split):
    verify_freeze()
    rows = read_jsonl(f'data/heldout/{split}.jsonl')
    schema = read_json('manifests/questions.json')
    coverage = read_json('manifests/coverage.json')
    overlap = read_json('manifests/train_overlap.json')
    return rows, schema, coverage, set(overlap['exact']) | set(overlap['near'])


def save_report(rows, predictions, schema, coverage, output):
    validate_predictions(rows, predictions, schema)
    report = score(predictions, schema, coverage)
    write_json(str(output) + '.metrics.json', report)
    errors = []
    for r in predictions:
        if r['prediction'] != r['label']:
            errors.append({**r, 'error_tags': [tag for condition, tag in [
                (r['prediction'] == -1, 'invalid_format'),
                (r['label'] in coverage[r['task_id']]['rare_labels'], 'rare_label'),
                (r['label'] in coverage[r['task_id']]['unsupported_train_labels'], 'unsupported_training_label'),
                (r.get('prompt_tokens',0) > 2048, 'long_passage'),
                (r.get('train_overlap',False), 'train_lexical_overlap')]
                if condition]})
    write_jsonl(str(output) + '.errors.jsonl', errors)
    return report


def majority(config, split='validation'):
    rows, schema, coverage, overlap = load_eval_data(split)
    train = read_jsonl('data/heldout/train.jsonl')
    counts = defaultdict(Counter)
    for r in train:
        counts[r['task_id']][r['label']] += 1
    chosen = {t:min(range(len(schema[t]['answers'])), key=lambda i: (-counts[t][i],i)) for t in schema}
    predictions = [prediction_record(r, chosen[r['task_id']], chr(65+chosen[r['task_id']]), schema, overlap) for r in rows]
    output = Path(f'outputs/majority/{split}.jsonl')
    write_jsonl(output, predictions)
    write_json(str(output)+'.metadata.json', {**metadata(config), 'baseline':'training_majority', 'counts':dict(counts)})
    report = save_report(rows, predictions, schema, coverage, output)
    mlflow, run = tracking('majority-'+split, config)
    with run:
        mlflow.log_artifacts(str(output.parent), 'evaluation')
        mlflow.log_metrics({k: v for k,v in report.items() if type(v) in (float,int)})
    print(json.dumps({k:v for k,v in report.items() if not isinstance(v,(dict,list))}, indent=2))
    return report


def predict(model, tokenizer, rows, schema, overlap, config, output, identity):
    import torch
    from transformers import GenerationConfig
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    meta_path = str(output)+'.metadata.json'
    expected = {**metadata(config), 'identity':identity, 'rows_sha256':sha('\n'.join(r['example_id'] for r in rows))}
    predictions = []
    if output.exists():
        if read_json(meta_path) != expected:
            raise ValueError('Cannot resume predictions with different experiment metadata')
        predictions = read_jsonl(output)
        ids = [p['example_id'] for p in predictions]
        assert len(set(ids)) == len(ids) and ids == [r['example_id'] for r in rows[:len(ids)]]
    else:
        write_json(meta_path, expected)
    generation = GenerationConfig(do_sample=False, max_new_tokens=config['max_new_tokens'],
                                   eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
                                   use_cache=True)
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    with output.open('a') as f, torch.inference_mode():
        for row in rows[len(predictions):]:
            ids = prompt_ids(tokenizer, row, schema)
            if len(ids)+config['max_new_tokens'] > config['context_length']:
                raise ValueError('Evaluation context overflow: '+row['example_id'])
            tensor = torch.tensor([ids], device='cuda')
            torch.cuda.synchronize()
            start = time.perf_counter()
            out = model.generate(input_ids=tensor, attention_mask=torch.ones_like(tensor), generation_config=generation)
            torch.cuda.synchronize()
            latency = time.perf_counter() - start
            new = out[0, len(ids):].tolist()
            content = new[:-1] if new and new[-1] == tokenizer.eos_token_id else new
            raw = tokenizer.decode(content, skip_special_tokens=False)
            p = prediction_record(row, parse_answer(raw, len(schema[row['task_id']]['answers'])), raw, schema, overlap,
                generated_token_ids=new, prompt_tokens=len(ids), prompt_tokens_sha256=sha(json.dumps(ids)),
                latency_seconds=latency, generated_tokens=len(new), truncated=False)
            predictions.append(p)
            f.write(json.dumps(p, ensure_ascii=False)+'\n')
            f.flush()
            if len(predictions) % 100 == 0:
                print(f'prediction progress {len(predictions)}/{len(rows)}', flush=True)
    durations = [r['latency_seconds'] for r in predictions]
    import numpy as np
    perf = {'inference_seconds':sum(durations), 'examples_per_second':len(durations)/sum(durations),
        'generated_tokens_per_second':sum(r['generated_tokens'] for r in predictions)/sum(durations),
        'latency_p50_seconds':float(np.percentile(durations,50)), 'latency_p95_seconds':float(np.percentile(durations,95)),
        'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
        'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,
        'memory_scope':'current process segment; latency includes all prediction rows'}
    write_json(str(output)+'.performance.json', perf)
    return predictions, perf


def evaluate(config, adapter=None, split='validation', limit=None, output=None):
    from .model import load_model
    rows, schema, coverage, overlap = load_eval_data(split)
    if limit:
        rows = rows[:limit]
    name = 'untuned-qwen' if adapter is None else 'tuned-qwen'
    if limit:
        name += '-smoke'
    output = output or f'outputs/{name}/{split}.jsonl'
    mlflow, run = tracking(name+'-'+split, config)
    with run:
        model, tokenizer, hardware = load_model(config, adapter=adapter)
        mlflow.log_dict(hardware, 'hardware.json')
        identity = {'adapter':str(adapter) if adapter else None,
                    'adapter_sha256':file_sha(Path(adapter)/'adapter_model.safetensors') if adapter else None}
        predictions, perf = predict(model, tokenizer, rows, schema, overlap, config, output, identity)
        report = save_report(rows, predictions, schema, coverage, output)
        mlflow.log_metrics({k:v for k,v in report.items() if type(v) in (float,int)})
        mlflow.log_metrics({k:v for k,v in perf.items() if type(v) in (float,int)})
        for path in Path(output).parent.glob(Path(output).name+'*'):
            mlflow.log_artifact(str(path), 'evaluation')
        write_json(Path(output).parent/'run.json', {'mlflow_run_id':run.info.run_id, **identity})
    print(json.dumps({k:v for k,v in report.items() if not isinstance(v,(dict,list))}, indent=2))
    return report
