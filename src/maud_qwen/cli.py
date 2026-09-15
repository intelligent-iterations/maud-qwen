import argparse
import json
from pathlib import Path

from .common import file_sha, metadata, read_json, read_jsonl, verify_freeze, write_json


def select(config, run_dir, out):
    from .train import verify_checkpoint
    completed = read_json(Path(run_dir)/'completed.json')
    scores = completed['completed_epoch_validation']
    best = max(scores, key=lambda r:(r['score'],-r['epoch']))
    assert completed['best_checkpoint'] == best['checkpoint']
    adapter = Path(best['checkpoint'])
    verify_checkpoint(adapter,config)
    record = {**metadata(config), 'adapter':str(adapter), 'adapter_sha256':file_sha(adapter/'adapter_model.safetensors'),
              'selected_epoch':best['epoch'],'validation_score':best['score'],'selection_metric':config['selection_metric'],
              'training_completed':completed,'test_access':'final comparison only, selection fixed before scoring'}
    if Path(out).exists():
        raise ValueError('Selection record already exists; do not overwrite a final selection')
    write_json(out,record)
    print(json.dumps(record,indent=2))


def require_final_record(config, path, adapter=None):
    if not path:
        raise ValueError('Test is sealed: create a selection record first, then pass --final-record')
    record = read_json(path)
    for key in ('freeze_sha256','context_manifest_sha256','config_sha256','runtime_config_sha256'):
        assert record[key] == metadata(config)[key]
    assert file_sha(Path(record['adapter'])/'adapter_model.safetensors') == record['adapter_sha256']
    if adapter:
        assert str(Path(adapter)) == record['adapter']
    return record


def main():
    p = argparse.ArgumentParser(description='Frozen MAUD agreement-held-out experiments')
    p.add_argument('command', choices=['prepare','lengths','majority','evaluate','train','smoke','rescore','compare','select','estimate','verify'])
    p.add_argument('--config', default='configs/resolved.json')
    p.add_argument('--split',choices=['train','validation','test'],default='validation')
    p.add_argument('--adapter')
    p.add_argument('--output')
    p.add_argument('--resume')
    p.add_argument('--limit',type=int)
    p.add_argument('--predictions')
    p.add_argument('--left')
    p.add_argument('--right')
    p.add_argument('--run-dir',default='outputs/initial')
    p.add_argument('--final-record')
    a = p.parse_args()
    config_path = a.config
    if a.command in ('prepare','lengths') and config_path == 'configs/resolved.json':
        config_path = 'configs/initial.json'
    config = read_json(config_path)
    if a.split == 'test' and a.command in ('majority','evaluate','rescore','compare'):
        require_final_record(config,a.final_record,a.adapter)
    if a.command == 'prepare':
        from .data import prepare
        prepare(config)
    elif a.command == 'lengths':
        from .data import measure_lengths
        measure_lengths(config)
    elif a.command == 'majority':
        from .evaluate import majority
        majority(config,a.split)
    elif a.command == 'evaluate':
        from .evaluate import evaluate
        evaluate(config,a.adapter,a.split,a.limit,a.output)
    elif a.command == 'train':
        from .train import train
        train(config,output=a.output or 'outputs/initial',resume=a.resume)
    elif a.command == 'smoke':
        from .train import verify_resume
        verify_resume(config)
    elif a.command == 'verify':
        verify_freeze()
        print('Frozen data and protocol verified')
    elif a.command == 'select':
        select(config,a.run_dir,a.output or 'manifests/final_selection.json')
    elif a.command == 'rescore':
        from .evaluate import load_eval_data, save_report
        rows,schema,coverage,_ = load_eval_data(a.split)
        expected = read_json(a.predictions+'.metadata.json')
        assert expected['freeze_sha256'] == file_sha('manifests/freeze.json')
        result = save_report(rows,read_jsonl(a.predictions),schema,coverage,a.output or a.predictions)
        print(json.dumps({k:v for k,v in result.items() if type(v) in (int,float)},indent=2))
    elif a.command == 'compare':
        from .evaluate import load_eval_data
        from .metrics import bootstrap, score, validate_predictions
        rows,schema,coverage,_ = load_eval_data(a.split)
        left,right = read_jsonl(a.left),read_jsonl(a.right)
        for path,preds in [(a.left,left),(a.right,right)]:
            validate_predictions(rows,preds,schema)
            meta = read_json(path+'.metadata.json')
            assert meta['freeze_sha256'] == file_sha('manifests/freeze.json')
            assert meta['config_sha256'] == metadata(config)['config_sha256']
            assert meta['runtime_config_sha256'] == metadata(config)['runtime_config_sha256']
        result = {'left':a.left,'right':a.right,
            'left_scores':score(left,schema,coverage),'right_scores':score(right,schema,coverage),
            'bootstrap':bootstrap(left,right,schema,coverage,config['bootstrap_replicates'],config['seed'])}
        write_json(a.output or f'reports/comparison-{a.split}.json',result)
        print(json.dumps(result['bootstrap'],indent=2))
    elif a.command == 'estimate':
        proof = read_json('outputs/smoke/resume_verified.json')
        perf = proof['throughput']
        evaluation = read_json('outputs/untuned-qwen/validation.jsonl.performance.json')
        seconds = perf['steady_mean_seconds_per_step']
        train_steps = perf['steps_per_epoch']*config['epochs']
        result = {'measured_seconds_per_optimizer_step':seconds,'optimizer_steps_per_epoch':perf['steps_per_epoch'],
            'six_epoch_training_steps':train_steps, 'training_hours':train_steps*seconds/3600,
            'six_validation_hours':config['epochs']*evaluation['inference_seconds']/3600,
            'total_hours_excluding_load_checkpoint_io':(train_steps*seconds+config['epochs']*evaluation['inference_seconds'])/3600,
            'basis':'8-step smoke, discard first 2; full untuned validation latency. Short pilot uncertainty; update from training steps.',
            'peak_training_allocated_gib':perf['peak_allocated_gib']}
        write_json('reports/runtime_estimate.json',result)
        print(json.dumps(result,indent=2))

if __name__ == '__main__':
    main()
