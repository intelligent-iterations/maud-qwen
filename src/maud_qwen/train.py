"""Single-GPU answer-only QLoRA with explicit, auditable optimizer boundaries."""
import copy
import gc
import json
import math
from pathlib import Path
import random
import statistics
import time

from .common import file_sha, metadata, read_json, read_jsonl, sha, tracking, verify_freeze, write_json
from .prompt import AnswerCollator, encode


def rng_state():
    import numpy as np
    import torch
    return {'python':random.getstate(), 'numpy':np.random.get_state(), 'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all()}


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda'])


def save_checkpoint(path, model, tokenizer, optimizer, scheduler, state, config):
    import torch
    path = Path(path)
    if path.exists():
        raise ValueError(f'Checkpoint already exists: {path}')
    tmp = path.with_name(path.name+'.writing')
    tmp.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(tmp, safe_serialization=True)
    tokenizer.save_pretrained(tmp)
    torch.save(optimizer.state_dict(), tmp/'optimizer.pt')
    torch.save(scheduler.state_dict(), tmp/'scheduler.pt')
    torch.save(rng_state(), tmp/'rng.pt')
    write_json(tmp/'trainer_state.json', state)
    write_json(tmp/'experiment.json', metadata(config))
    write_json(tmp/'checkpoint_hashes.json', {p.name:file_sha(p) for p in sorted(tmp.iterdir()) if p.is_file()})
    tmp.rename(path)
    write_json(path.parent/'latest.json', {'checkpoint':str(path), 'global_step':state['global_step']})
    import mlflow
    if mlflow.active_run():
        for name in ['checkpoint_hashes.json','trainer_state.json','experiment.json']:
            mlflow.log_artifact(str(path/name),'checkpoints/'+path.name)
        mlflow.log_dict({'on_premises_path':str(path.resolve())},'checkpoints/'+path.name+'/location.json')
    return path


def verify_checkpoint(path, config):
    path = Path(path)
    for name, digest in read_json(path/'checkpoint_hashes.json').items():
        assert file_sha(path/name) == digest, f'Corrupt checkpoint file {name}'
    expected = read_json(path/'experiment.json')
    current = metadata(config)
    for key in ('config_sha256','freeze_sha256','context_manifest_sha256','code_commit','runtime_config_sha256'):
        if current.get(key) != expected.get(key):
            raise ValueError(f'Resume identity mismatch: {key}')


def answer_loss(model, batch):
    import torch.nn.functional as F
    # At microbatch=1, the last answer+EOS tokens are the only supervised targets.
    labels = batch['labels']
    assert labels.shape[0] == 1
    n = int((labels[0] != -100).sum())
    assert n > 0 and (labels[0,:-n] == -100).all() and (labels[0,-n:] >= 0).all()
    outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                    use_cache=False, logits_to_keep=n+1)
    # Keeping only the answer positions avoids a context-length x vocabulary logits allocation.
    logits = outputs.logits[:,:-1,:].float()
    return F.cross_entropy(logits.reshape(-1,logits.shape[-1]), labels[:,-n:].reshape(-1))


def normalized_example_loss(loss, weight, total_weight):
    # Normalize across the effective batch, not within each one-example
    # microbatch (which would cancel a class weight under mean reduction).
    return loss / total_weight if weight == 1 else (loss * weight) / total_weight


def training_step(model, optimizer, scheduler, encoded_rows, config, example_weights=None):
    import torch
    model.train()
    model.config.use_cache = False
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    start = time.perf_counter()
    losses = []
    weights = [1.0] * len(encoded_rows) if example_weights is None else list(example_weights)
    if len(weights) != len(encoded_rows) or not weights or any(not math.isfinite(w) or w <= 0 for w in weights):
        raise ValueError('One finite positive weight is required for each example')
    total_weight = sum(weights)
    collate = AnswerCollator(0)
    for item, weight in zip(encoded_rows, weights):
        batch = {k:v.to('cuda') for k,v in collate([item]).items()}
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = answer_loss(model, batch)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite training loss')
        losses.append(float(loss.detach()))
        if example_weights is None:
            (loss / len(encoded_rows)).backward()
        else:
            normalized_example_loss(loss, weight, total_weight).backward()
    norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], config['max_grad_norm'])
    if not torch.isfinite(norm):
        raise FloatingPointError('Nonfinite gradient norm')
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    elapsed=time.perf_counter()-start
    tokens=sum(len(x['input_ids']) for x in encoded_rows)
    result = {'loss':statistics.mean(losses), 'seconds_per_optimizer_step':elapsed,
            'examples_per_second':len(encoded_rows)/elapsed, 'input_tokens_per_second':tokens/elapsed,
            'examples':len(encoded_rows), 'input_tokens':sum(len(x['input_ids']) for x in encoded_rows),
            'grad_norm':float(norm), 'learning_rate':scheduler.get_last_lr()[0]}
    if example_weights is not None:
        result.update(weighted_loss=sum(w*l for w,l in zip(weights,losses))/total_weight,
                      effective_batch_weight_sum=total_weight,
                      upweighted_examples=sum(w > 1 for w in weights))
    return result


def train(config, output='outputs/initial', resume=None, smoke=False, max_steps=None, stop_after=None):
    import numpy as np
    import torch
    from transformers import get_scheduler
    from .model import load_model
    from .evaluate import predict, save_report, load_eval_data
    verify_freeze()
    if config['per_device_train_batch_size'] != 1 or config['sampling'] != 'natural_frequency':
        raise ValueError('v1 implements natural frequency and microbatch 1 only')
    if not smoke:
        proof = read_json('outputs/smoke/resume_verified.json')
        assert proof['passed'] and proof['freeze_sha256'] == file_sha('manifests/freeze.json')
        assert proof['code_commit'] == metadata(config)['code_commit']
        assert proof['runtime_config_sha256'] == metadata(config)['runtime_config_sha256']
        baseline_path = 'outputs/untuned-qwen/validation.jsonl'
        baseline = read_json(baseline_path+'.metadata.json')
        assert baseline['config_sha256'] == metadata(config)['config_sha256']
        assert baseline['runtime_config_sha256'] == metadata(config)['runtime_config_sha256']
        valid, schema, coverage, _ = load_eval_data('validation')
        from .metrics import validate_predictions
        validate_predictions(valid, read_jsonl(baseline_path), schema)
    if resume:
        verify_checkpoint(resume, config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl('data/heldout/train.jsonl')
    schema = read_json('manifests/questions.json')
    accumulation = config['gradient_accumulation_steps']
    steps_per_epoch = math.ceil(len(rows) / accumulation)
    planned_steps = max_steps or steps_per_epoch * config['epochs']
    state = {'epoch':0, 'row_offset':0, 'global_step':0, 'best_score':None, 'best_checkpoint':None,
             'bad_epochs':0, 'planned_steps':planned_steps, 'steps_per_epoch':steps_per_epoch,
             'completed_epoch_validation':[], 'stop_reason':None}
    mlflow, run = tracking('qlora-smoke' if smoke else 'qlora-lr3e-5-natural', config)
    with run:
        model, tokenizer, hardware = load_model(config, adapter=resume, training=True)
        write_json(output/'hardware.json', hardware)
        mlflow.log_dict(hardware, 'hardware.json')
        assert len(hardware['lora_modules']) == model.config.num_hidden_layers * len(config['lora']['target_modules'])
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=config['learning_rate'], weight_decay=config['weight_decay'],
             betas=(config['adam_beta1'],config['adam_beta2']), eps=config['adam_epsilon'], foreach=False, fused=False)
        scheduler = get_scheduler(config['scheduler'], optimizer=optimizer,
             num_warmup_steps=math.ceil(planned_steps*config['warmup_ratio']), num_training_steps=planned_steps)
        if resume:
            # Only load files produced by this project and verified above. Pickle state is not safe for untrusted checkpoints.
            optimizer.load_state_dict(torch.load(Path(resume)/'optimizer.pt', map_location='cpu', weights_only=False))
            scheduler.load_state_dict(torch.load(Path(resume)/'scheduler.pt', weights_only=False))
            state = read_json(Path(resume)/'trainer_state.json')
            assert state['planned_steps'] == planned_steps
            restore_rng(torch.load(Path(resume)/'rng.pt', map_location='cpu', weights_only=False))
            mlflow.set_tag('resumed_from',str(resume))
        encoded_cache = {}
        def encoded(i):
            if i not in encoded_cache:
                encoded_cache[i] = encode(tokenizer, rows[i], schema, config['context_length'])
            return encoded_cache[i]
        timings = []
        torch.cuda.reset_peak_memory_stats()
        history = output/'steps.jsonl'
        with history.open('a') as f:
            while (state['epoch'] < config['epochs'] and not state['stop_reason'] and
                   (state['global_step'] < planned_steps or state['row_offset'] == len(rows))):
                order = np.random.default_rng(config['seed']+state['epoch']).permutation(len(rows)).tolist()
                for offset in range(state['row_offset'], len(rows), accumulation):
                    items = [encoded(i) for i in order[offset:offset+accumulation]]
                    stats = training_step(model, optimizer, scheduler, items, config)
                    state['global_step'] += 1
                    state['row_offset'] = min(offset+accumulation,len(rows))
                    timings.append(stats['seconds_per_optimizer_step'])
                    stats.update(global_step=state['global_step'], epoch=state['epoch']+state['row_offset']/len(rows))
                    stats['peak_allocated_gib'] = torch.cuda.max_memory_allocated()/2**30
                    stats['peak_reserved_gib'] = torch.cuda.max_memory_reserved()/2**30
                    f.write(json.dumps(stats)+'\n'); f.flush()
                    mlflow.log_metrics({k:v for k,v in stats.items() if k != 'global_step'}, step=state['global_step'])
                    if state['global_step'] % 10 == 0 or smoke:
                        print(json.dumps(stats),flush=True)
                    checkpoint_due = (state['global_step'] == 10 or state['global_step'] % config['save_steps'] == 0 or
                                       state['global_step'] == planned_steps or state['global_step'] == stop_after)
                    if checkpoint_due:
                        cp = output/f"checkpoint-{state['global_step']}"
                        save_checkpoint(cp, model, tokenizer, optimizer, scheduler, state, config)
                    if (stop_after and state['global_step'] >= stop_after) or state['global_step'] >= planned_steps:
                        break
                if smoke or (stop_after and state['global_step'] >= stop_after):
                    break
                if state['row_offset'] == len(rows):
                    epoch_number = state['epoch']+1
                    valid, _, coverage, overlap = load_eval_data('validation')
                    pred_path = output/f'validation-epoch-{epoch_number}.jsonl'
                    torch.cuda.empty_cache()
                    predictions, perf = predict(model, tokenizer, valid, schema, overlap, config, pred_path,
                        {'training_run':str(output), 'global_step':state['global_step']})
                    report = save_report(valid,predictions,schema,coverage,pred_path)
                    metric = report[config['selection_metric']]
                    assert metric is not None
                    mlflow.log_metric('validation_question_macro_f1',metric,step=state['global_step'])
                    mlflow.log_metric('validation_accuracy',report['accuracy'],step=state['global_step'])
                    for artifact in output.glob(pred_path.name+'*'):
                        mlflow.log_artifact(str(artifact),f'epoch-{epoch_number}')
                    improved = state['best_score'] is None or metric > state['best_score']+config['early_stopping_min_delta']
                    epoch_checkpoint = output/f'epoch-{epoch_number}'
                    if improved:
                        state['best_score'] = metric
                        state['best_checkpoint'] = str(epoch_checkpoint)
                        state['bad_epochs'] = 0
                    else:
                        state['bad_epochs'] += 1
                    state['completed_epoch_validation'].append({'epoch':epoch_number,'score':metric,'checkpoint':str(epoch_checkpoint)})
                    state['epoch'] += 1
                    state['row_offset'] = 0
                    if state['bad_epochs'] >= config['early_stopping_patience']:
                        state['stop_reason'] = 'two_consecutive_epochs_without_improvement'
                    save_checkpoint(epoch_checkpoint,model,tokenizer,optimizer,scheduler,state,config)
                    write_json(output/'selection_progress.json',state)
                    print('EPOCH VALIDATION '+json.dumps(state),flush=True)
                    if state['stop_reason']:
                        break
            if not smoke and not stop_after:
                state['stop_reason'] = state['stop_reason'] or 'completed_six_epochs'
                write_json(output/'completed.json',state)
        steady = timings[2:]
        performance = {'measured_optimizer_steps':len(timings), 'discarded_warmup_steps':min(2,len(timings)),
            'steady_mean_seconds_per_step':statistics.mean(steady) if steady else None,
            'steady_median_seconds_per_step':statistics.median(steady) if steady else None,
            'peak_allocated_gib':max((s.get('peak_allocated_gib',0) for s in read_jsonl(history)), default=0),
            'peak_reserved_gib':max((s.get('peak_reserved_gib',0) for s in read_jsonl(history)), default=0),
            'steps_per_epoch':steps_per_epoch, 'planned_steps':planned_steps, 'smoke':smoke}
        write_json(output/'performance.json',performance)
        mlflow.log_dict(performance,'performance.json')
        write_json(output/'run.json',{'mlflow_run_id':run.info.run_id,'state':state})
        mlflow.log_artifact(str(history),'training')
    del model, optimizer, scheduler, params
    gc.collect(); torch.cuda.empty_cache()
    return performance


def verify_resume(config, steps=8, boundary=4):
    import numpy as np
    import torch
    from safetensors.torch import load_file
    verify_freeze()
    # Stress the longest training passage before benchmarking the natural-frequency pilot.
    from .model import load_model
    model,tokenizer,_ = load_model(config,training=True)
    schema = read_json('manifests/questions.json')
    rows = read_jsonl('data/heldout/train.jsonl')
    lengths = read_jsonl('reports/token_lengths.jsonl')
    longest = max((r for r in lengths if r['split']=='train'),key=lambda r:r['prompt_tokens'])
    row = next(r for r in rows if r['example_id']==longest['example_id'])
    item = encode(tokenizer,row,schema,config['context_length'])
    batch = {k:v.to('cuda') for k,v in AnswerCollator(tokenizer.pad_token_id)([item]).items()}
    torch.cuda.reset_peak_memory_stats()
    model.train()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        loss=answer_loss(model,batch)
    loss.backward()
    assert torch.isfinite(loss)
    stress={'example_id':row['example_id'],'input_tokens':len(item['input_ids']),
            'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
            'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,'loss':float(loss.detach())}
    del model,tokenizer,loss,batch
    gc.collect();torch.cuda.empty_cache()
    write_json('outputs/smoke/longest_passage.json',stress)
    left = Path('outputs/smoke/uninterrupted')
    right = Path('outputs/smoke/resumed')
    perf = train(config, output=left, smoke=True, max_steps=steps)
    train(config, output=right, smoke=True, max_steps=steps, stop_after=boundary)
    train(config, output=right, smoke=True, max_steps=steps, resume=right/f'checkpoint-{boundary}')
    a,b = left/f'checkpoint-{steps}',right/f'checkpoint-{steps}'
    weights_a,weights_b = load_file(str(a/'adapter_model.safetensors')),load_file(str(b/'adapter_model.safetensors'))
    assert weights_a.keys() == weights_b.keys()
    maximum = max(float((weights_a[k]-weights_b[k]).abs().max()) for k in weights_a)
    assert maximum <= 1e-6, f'Resumed weights differ by {maximum}'
    ha,hb = read_jsonl(left/'steps.jsonl'), read_jsonl(right/'steps.jsonl')
    loss_difference = max(abs(x['loss']-y['loss']) for x,y in zip(ha,hb))
    assert len(ha)==len(hb)==steps and loss_difference <= 1e-6
    assert read_json(a/'trainer_state.json') == read_json(b/'trainer_state.json')
    # Compare optimizer and scheduler values recursively, including moments and counters.
    def compare(x,y):
        if isinstance(x,torch.Tensor):
            assert torch.allclose(x,y,rtol=0,atol=1e-6)
        elif isinstance(x,np.ndarray):
            assert np.array_equal(x,y)
        elif isinstance(x,dict):
            assert x.keys()==y.keys()
            for k in x: compare(x[k],y[k])
        elif isinstance(x,(list,tuple)):
            assert len(x)==len(y)
            for xx,yy in zip(x,y): compare(xx,yy)
        else:
            assert x==y
    for name in ['optimizer.pt','scheduler.pt','rng.pt']:
        compare(torch.load(a/name,map_location='cpu',weights_only=False),torch.load(b/name,map_location='cpu',weights_only=False))
    proof = {'passed':True, 'optimizer_steps':steps,'resume_boundary':boundary,
        'max_adapter_absolute_difference':maximum,'max_loss_absolute_difference':loss_difference,
        'optimizer_scheduler_trainer_rng_equal':True,
        'state_comparison_tolerance':{'floating_tensor_atol':1e-6,'floating_tensor_rtol':0,'non_tensor_state':'exact'},
        'freeze_sha256':file_sha('manifests/freeze.json'),
        'code_commit':metadata(config)['code_commit'],'throughput':perf,'longest_passage_stress':stress}
    proof['runtime_config_sha256'] = metadata(config)['runtime_config_sha256']
    write_json('outputs/smoke/resume_verified.json',proof)
    mlflow,run=tracking('smoke-verification-proof',config)
    with run:
        mlflow.log_dict(proof,'resume_verified.json')
        mlflow.log_dict(stress,'longest_passage.json')
    print(json.dumps(proof,indent=2))
    return proof
