"""One controlled, resumable weighted third epoch from a pinned epoch-two parent."""
import argparse
import copy
import gc
import json
import math
from pathlib import Path
import statistics

from .common import file_sha, metadata, read_json, read_jsonl, sha, tracking, verify_freeze, write_json
from .evaluate import load_eval_data, predict, save_report
from .metrics import bootstrap, score, validate_predictions
from .prompt import encode
from .train import restore_rng, save_checkpoint, training_step, verify_checkpoint


def registration(parent_project):
    path = Path('outputs/weighted-registration.json')
    if not path.exists():
        raise ValueError('Register this reproduction parent first: python scripts/register_weighted.py --parent-project PATH')
    record = read_json(path)
    if Path(record['parent_project']).resolve() != Path(parent_project).resolve():
        raise ValueError('Registered parent project differs')
    return record['plan'], record['parent_receipt']


def context(parent_project):
    verify_freeze()
    plan, receipt = registration(parent_project)
    base = read_json('configs/resolved.json')
    parent = Path(parent_project) / plan['parent_checkpoint']
    assert file_sha(parent/'checkpoint_hashes.json') == receipt['checkpoint_hashes_sha256']
    for name, digest in read_json(parent/'checkpoint_hashes.json').items():
        assert Path(name).name == name
        assert file_sha(parent/name) == digest, name
    old = read_json(parent/'experiment.json')
    current = metadata(base)
    assert old['config'] == base
    assert old['code_commit'] == plan['parent_code_commit']
    for key in ('freeze_sha256', 'context_manifest_sha256', 'config_sha256', 'runtime_config_sha256'):
        assert old[key] == current[key], key
    state = read_json(parent/'trainer_state.json')
    assert state['epoch'] == plan['parent_epoch'] == 2 and state['row_offset'] == 0
    assert state['global_step'] == plan['parent_global_step'] == 3154
    assert state['planned_steps'] == plan['original_planned_steps'] == 9462
    assert state['steps_per_epoch'] == 1577
    assert plan['target_epoch'] == 3 and plan['target_global_step'] == 4731
    assert plan['rare_multiplier'] == 3.0 and plan['ordinary_multiplier'] == 1.0
    config = {**base, 'weighted_branch': {**plan, 'parent_checkpoint_hashes_sha256': receipt['checkpoint_hashes_sha256']}}
    return config, plan, parent, state


def train_branch(parent_project, output='outputs/initial', resume=None, smoke_steps=None, stop_after=None, unweighted_replay=False):
    import numpy as np
    import torch
    from transformers import get_scheduler
    from .model import load_model

    config, plan, parent, parent_state = context(parent_project)
    if not smoke_steps:
        proof = read_json('outputs/smoke/resume_verified.json')
        assert proof['passed'] and proof['config_sha256'] == metadata(config)['config_sha256']
        assert proof['code_commit'] == metadata(config)['code_commit']
        assert proof['parent_checkpoint_hashes_sha256'] == config['weighted_branch']['parent_checkpoint_hashes_sha256']
    if resume:
        verify_checkpoint(resume, config)
    output = Path(output)
    if output.exists() and any(output.iterdir()) and not resume:
        raise ValueError('Nonempty branch output requires an explicit verified resume')
    output.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl('data/heldout/train.jsonl')
    coverage, schema = read_json('manifests/coverage.json'), read_json('manifests/questions.json')
    accumulation = config['gradient_accumulation_steps']
    order = np.random.default_rng(config['seed'] + plan['parent_epoch']).permutation(len(rows)).tolist()
    assert len(rows) == 25225 and math.ceil(len(rows) / accumulation) == 1577
    weights = [plan['rare_multiplier'] if r['label'] in coverage[r['task_id']]['rare_labels'] else 1.0 for r in rows]
    assert sum(w > 1 for w in weights) == 716
    if smoke_steps and not unweighted_replay:
        assert any(weights[i] > 1 for i in order[:smoke_steps * accumulation]), 'Smoke must exercise rare weighting'
    state = copy.deepcopy(parent_state)
    state.update(best_score=None, best_checkpoint=None, bad_epochs=0, completed_epoch_validation=[], stop_reason=None,
                 branch_parent=str(parent), branch_start_step=plan['parent_global_step'])
    if resume:
        state = read_json(Path(resume)/'trainer_state.json')
    assert state['epoch'] == 2 and 0 <= state['row_offset'] <= len(rows)
    assert state['global_step'] <= plan['target_global_step']
    source = Path(resume) if resume else parent
    last_step = plan['parent_global_step'] + smoke_steps if smoke_steps else plan['target_global_step']
    name = 'rare3x-smoke' if smoke_steps else 'rare3x-alternative-epoch3'
    if unweighted_replay:
        name = 'unweighted-parent-replay-proof'
    mlflow, run = tracking(name, config)
    stats_rows = []
    with run:
        mlflow.set_tags({'parent_checkpoint': str(parent), 'parent_code_commit': plan['parent_code_commit'],
                         'branch_kind': 'explicit experimental fork; preserved optimizer, scheduler and RNG'})
        mlflow.log_dict({'epoch_index': 2, 'order_seed': config['seed'] + 2,
            'ordered_example_ids_sha256': sha('\n'.join(rows[i]['example_id'] for i in order)),
            'examples_per_epoch': len(rows), 'rare_examples_per_epoch': sum(w > 1 for w in weights),
            'parent_checkpoint_hashes_sha256': config['weighted_branch']['parent_checkpoint_hashes_sha256']}, 'branch_design.json')
        model, tokenizer, hardware = load_model(config, adapter=str(source), training=True)
        mlflow.log_dict(hardware, 'hardware.json')
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=config['learning_rate'], weight_decay=config['weight_decay'],
            betas=(config['adam_beta1'], config['adam_beta2']), eps=config['adam_epsilon'], foreach=False, fused=False)
        scheduler = get_scheduler(config['scheduler'], optimizer=optimizer,
            num_warmup_steps=math.ceil(plan['original_planned_steps'] * config['warmup_ratio']),
            num_training_steps=plan['original_planned_steps'])
        # The trusted local checkpoint bytes and parent identity are verified above.
        optimizer.load_state_dict(torch.load(source/'optimizer.pt', map_location='cpu', weights_only=False))
        scheduler.load_state_dict(torch.load(source/'scheduler.pt', map_location='cpu', weights_only=False))
        assert scheduler.last_epoch == state['global_step']
        restore_rng(torch.load(source/'rng.pt', map_location='cpu', weights_only=False))
        mlflow.log_dict({'source': str(source), 'global_step': state['global_step'],
            'scheduler_last_epoch': scheduler.last_epoch, 'optimizer_lr': optimizer.param_groups[0]['lr'],
            'planned_steps_preserved': plan['original_planned_steps']}, 'restored_state.json')
        cache = {}
        torch.cuda.reset_peak_memory_stats()
        with (output/'steps.jsonl').open('a') as history:
            for offset in range(state['row_offset'], len(rows), accumulation):
                indices = order[offset:offset + accumulation]
                for index in indices:
                    if index not in cache:
                        cache[index] = encode(tokenizer, rows[index], schema, config['context_length'])
                stats = training_step(model, optimizer, scheduler, [cache[i] for i in indices], config,
                    example_weights=None if unweighted_replay else [weights[i] for i in indices])
                state['global_step'] += 1
                state['row_offset'] = min(offset + accumulation, len(rows))
                stats.update(global_step=state['global_step'], epoch=2 + state['row_offset'] / len(rows),
                    peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                    peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
                stats_rows.append(stats)
                history.write(json.dumps(stats) + '\n'); history.flush()
                mlflow.log_metrics({k: v for k, v in stats.items() if k != 'global_step'}, step=state['global_step'])
                if state['global_step'] % 10 == 0 or smoke_steps:
                    print(json.dumps(stats), flush=True)
                due = state['global_step'] % config['save_steps'] == 0 or state['global_step'] in (last_step, stop_after)
                if due:
                    save_checkpoint(output/f"checkpoint-{state['global_step']}", model, tokenizer, optimizer, scheduler, state, config)
                if state['global_step'] >= last_step or (stop_after and state['global_step'] >= stop_after):
                    break
        if not smoke_steps and state['global_step'] == plan['target_global_step']:
            assert state['row_offset'] == len(rows)
            valid, _, coverage, overlap = load_eval_data('validation')
            torch.cuda.empty_cache()
            pred_path = output/'validation-epoch-3.jsonl'
            preds, performance = predict(model, tokenizer, valid, schema, overlap, config, pred_path,
                {'training_run': str(output.resolve()), 'global_step': state['global_step'], 'parent_checkpoint': str(parent)})
            metrics = save_report(valid, preds, schema, coverage, pred_path)
            for artifact in output.glob(pred_path.name + '*'):
                mlflow.log_artifact(str(artifact), 'epoch-3')
            mlflow.log_metrics({'validation_' + k: v for k, v in metrics.items() if type(v) in (float, int)})
            epoch_path = output/'epoch-3'
            state.update(epoch=3, row_offset=0, best_score=metrics['question_macro_f1'], best_checkpoint=str(epoch_path),
                completed_epoch_validation=[{'epoch': 3, 'score': metrics['question_macro_f1'], 'checkpoint': str(epoch_path)}],
                stop_reason='completed_one_registered_weighted_epoch')
            save_checkpoint(epoch_path, model, tokenizer, optimizer, scheduler, state, config)
            write_json(output/'selection_progress.json', state)
            write_json(output/'completed.json', state)
        write_json(output/'performance.json', {'measured_optimizer_steps': len(stats_rows),
            'steady_mean_seconds_per_step': statistics.mean(s['seconds_per_optimizer_step'] for s in stats_rows[2:]) if len(stats_rows) > 2 else None,
            'sampling': 'natural_frequency', 'weight_normalization': plan['normalization']})
        mlflow.log_artifact(str(output/'steps.jsonl'), 'training')
        write_json(output/'run.json', {'mlflow_run_id': run.info.run_id, 'state': state})
    del model, tokenizer, optimizer, scheduler, params
    gc.collect(); torch.cuda.empty_cache()
    return stats_rows


def smoke(parent_project):
    import numpy as np
    import torch
    from safetensors.torch import load_file
    config, plan, parent, _ = context(parent_project)
    # Replaying the unchanged kernel checks the fork's parent/order/schedule before changing its objective.
    replay = train_branch(parent_project, 'outputs/smoke/control-replay', smoke_steps=4, unweighted_replay=True)
    reference = {r['global_step']: r for r in read_jsonl(Path(parent_project)/'outputs/initial/steps.jsonl')}
    control_differences = {key: max(abs(r[key] - reference[r['global_step']][key]) for r in replay)
                          for key in ('loss', 'grad_norm', 'learning_rate', 'input_tokens', 'examples')}
    assert all(value <= 1e-6 for value in control_differences.values()), control_differences
    left, right = Path('outputs/smoke/uninterrupted'), Path('outputs/smoke/resumed')
    train_branch(parent_project, str(left), smoke_steps=8)
    train_branch(parent_project, str(right), smoke_steps=8, stop_after=3158)
    train_branch(parent_project, str(right), smoke_steps=8, resume=str(right/'checkpoint-3158'))
    a, b = left/'checkpoint-3162', right/'checkpoint-3162'
    wa, wb = load_file(str(a/'adapter_model.safetensors')), load_file(str(b/'adapter_model.safetensors'))
    maximum = max(float((wa[k] - wb[k]).abs().max()) for k in wa)
    assert maximum <= 1e-6
    def compare(x, y):
        if isinstance(x, torch.Tensor):
            assert torch.allclose(x, y, rtol=0, atol=1e-6)
        elif isinstance(x, np.ndarray):
            assert np.array_equal(x, y)
        elif isinstance(x, dict):
            assert x.keys() == y.keys()
            for key in x: compare(x[key], y[key])
        elif isinstance(x, (list, tuple)):
            assert len(x) == len(y)
            for xx, yy in zip(x, y): compare(xx, yy)
        else:
            assert x == y
    for name in ('optimizer.pt', 'scheduler.pt', 'rng.pt'):
        compare(torch.load(a/name, map_location='cpu', weights_only=False), torch.load(b/name, map_location='cpu', weights_only=False))
    assert read_json(a/'trainer_state.json') == read_json(b/'trainer_state.json')
    ha, hb = read_jsonl(left/'steps.jsonl'), read_jsonl(right/'steps.jsonl')
    assert len(ha) == len(hb) == 8
    assert sum(row['upweighted_examples'] for row in ha) > 0
    differences = {key: max(abs(x[key] - y[key]) for x, y in zip(ha, hb))
                   for key in ('loss', 'weighted_loss', 'grad_norm', 'learning_rate', 'input_tokens')}
    assert all(value <= 1e-6 for value in differences.values())
    proof = {'passed': True, 'code_commit': metadata(config)['code_commit'], 'config_sha256': metadata(config)['config_sha256'],
        'freeze_sha256': metadata(config)['freeze_sha256'], 'parent_checkpoint_hashes_sha256': config['weighted_branch']['parent_checkpoint_hashes_sha256'],
        'max_adapter_absolute_difference': maximum, 'max_loss_absolute_difference': differences['loss'],
        'control_replay_differences': control_differences, 'weighted_resume_differences': differences,
        'optimizer_scheduler_trainer_rng_equal': True, 'atol': 1e-6, 'rtol': 0}
    write_json('outputs/smoke/resume_verified.json', proof)
    mlflow, run = tracking('weighted-resume-proof', config)
    with run:
        mlflow.log_dict(proof, 'resume_verified.json')
    return proof


def compare_branches(parent_project):
    config, plan, _, _ = context(parent_project)
    paths = [Path(parent_project)/plan['control_predictions'], Path('outputs/initial/validation-epoch-3.jsonl')]
    rows, schema, coverage, _ = load_eval_data('validation')
    predictions = [validate_predictions(rows, read_jsonl(path), schema) for path in paths]
    metas = [read_json(str(path)+'.metadata.json') for path in paths]
    for meta in metas:
        for key in ('freeze_sha256', 'context_manifest_sha256', 'runtime_config_sha256'):
            assert meta[key] == metadata(config)[key]
    assert metas[0]['config'] == {k: v for k, v in config.items() if k != 'weighted_branch'}
    assert metas[1]['config'] == config
    for left, right in zip(*predictions):
        for key in ('example_id', 'prompt_tokens_sha256', 'valid_answers', 'text_sha256'):
            assert left[key] == right[key], key
    scores = [score(preds, schema, coverage) for preds in predictions]
    diagnostics = []
    for preds, metrics in zip(predictions, scores):
        rare_predictions = [r for r in preds if r['prediction'] in coverage[r['task_id']]['rare_labels']]
        correct = sum(r['prediction'] == r['label'] for r in rare_predictions)
        diagnostics.append({'rare_recall': metrics['rare_answer_accuracy'], 'rare_correct': correct,
            'rare_predictions': len(rare_predictions), 'rare_precision': correct/len(rare_predictions) if rare_predictions else None,
            'incorrect_rare_predictions': len(rare_predictions)-correct})
    result = {'comparison': 'same epoch-two parent; ordinary versus rare3x third epoch',
        'parent_receipt': registration(parent_project)[1], 'plan': plan,
        'control_scores': scores[0], 'weighted_scores': scores[1],
        'control_rare': diagnostics[0], 'weighted_rare': diagnostics[1],
        'bootstrap': bootstrap(*predictions, schema, coverage, config['bootstrap_replicates'], config['seed']),
        'promising_under_registered_rule': scores[1]['rare_answer_accuracy'] > scores[0]['rare_answer_accuracy'] and scores[1]['question_macro_f1'] >= scores[0]['question_macro_f1'],
        'prediction_sha256': [file_sha(path) for path in paths], 'test_access': 'sealed'}
    write_json('reports/weighted-vs-control-epoch3.json', result)
    mlflow, run = tracking('weighted-vs-control-epoch3', config)
    with run:
        mlflow.log_artifact('reports/weighted-vs-control-epoch3.json', 'comparison')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['preflight', 'smoke', 'train', 'compare'])
    p.add_argument('--parent-project', required=True)
    p.add_argument('--resume')
    args = p.parse_args()
    if args.command == 'preflight':
        config, _, parent, state = context(args.parent_project)
        print(json.dumps({'parent_verified': str(parent), 'state': state, 'branch_config_sha256': metadata(config)['config_sha256']}))
    elif args.command == 'smoke':
        print(json.dumps(smoke(args.parent_project), indent=2))
    elif args.command == 'train':
        train_branch(args.parent_project, resume=args.resume)
    else:
        result = compare_branches(args.parent_project)
        print(json.dumps({'promising': result['promising_under_registered_rule'], 'control_rare': result['control_rare'], 'weighted_rare': result['weighted_rare']}, indent=2))
