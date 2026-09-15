"""Rebuildable official-data audit and deterministic agreement partitions."""
from collections import Counter, defaultdict
import csv
import itertools
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import unicodedata
import zipfile

from .common import file_sha, read_json, read_jsonl, sha, verify_freeze, write_json, write_jsonl

SPLITS = ('train', 'validation', 'test')


def normalize(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def mapped_answer(row):
    """Match pinned specs.MultiBinaryDatasetSubQuestionSpec exactly; retain raw answers."""
    if row['subquestion'] != '<NONE>':
        expected = int(row['subquestion'] in row['answer'].split(', '))
        assert int(row['label']) == expected, ('Binary membership mismatch',row['id'],row['answer'])
        return row['subquestion'] if expected else '<OTHER>'
    return row['answer']


def fetch(config):
    repo = Path('vendor/maud')
    if not repo.exists():
        subprocess.run(['git', 'clone', config['dataset_repository'], str(repo)], check=True)
    current = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if current != config['dataset_commit']:
        subprocess.run(['git', '-C', str(repo), 'fetch', 'origin', config['dataset_commit']], check=True)
        subprocess.run(['git', '-C', str(repo), 'checkout', '--detach', config['dataset_commit']], check=True)
    if not (repo / 'data/MAUD_train.csv').exists():
        with zipfile.ZipFile(repo / 'data.zip') as z:
            for item in z.infolist():
                if item.filename.startswith('/') or '..' in Path(item.filename).parts:
                    raise ValueError('Unsafe upstream archive path')
            z.extractall(repo)
    return repo


def build_schema(rows):
    schema = {}
    for r in rows:
        tid = r['id']
        info = {k: r[k] for k in ('question', 'subquestion', 'text_type', 'category')}
        if tid not in schema:
            schema[tid] = {**info, 'answer_map': {}}
        assert all(schema[tid][k] == v for k, v in info.items()), (tid, info)
        labels = schema[tid]['answer_map']
        label = int(r['label'])
        answer = mapped_answer(r)
        assert label not in labels or labels[label] == answer, (tid, label)
        labels[label] = answer
    seen_questions = set()
    for tid, info in schema.items():
        keys = sorted(info['answer_map'])
        assert keys == list(range(len(keys))) and 1 < len(keys) <= 26, (tid, keys)
        info['answers'] = [info['answer_map'][i] for i in keys]
        assert len(set(info['answers'])) == len(keys)
        del info['answer_map']
        qkey = (info['question'], info['subquestion'])
        assert qkey not in seen_questions
        seen_questions.add(qkey)
    return dict(sorted(schema.items()))


def duplicate_audit(rows, config):
    """Exact normalized overlap; exhaustive sparse 5-word Jaccard within text type.

    All pairs above threshold are checked, with no probabilistic LSH recall claim.
    Near matching keeps numbers and words; it does not infer semantic equivalence.
    """
    import numpy as np
    from scipy.sparse import csr_matrix
    by_text = defaultdict(list)
    for r in rows:
        by_text[normalize(r['text'])].append(r)
    cross_exact = []
    exact_seen = set()
    conflict = []
    near_seen = set()
    for text, records in by_text.items():
        if len({r['agreement_id'] for r in records}) > 1:
            tids = defaultdict(list)
            for r in records:
                tids[r['task_id']].append(r)
            for tid, rs in tids.items():
                if len({r['agreement_id'] for r in rs}) <= 1:
                    continue
                splits = sorted({r['split'] for r in rs})
                cross_exact.append({'task_id': tid, 'text_sha256': sha(text),
                    'agreement_ids': sorted({r['agreement_id'] for r in rs}), 'splits': splits,
                    'example_ids': [r['example_id'] for r in rs], 'labels': sorted({r['label'] for r in rs})})
                if len({r['label'] for r in rs}) > 1:
                    conflict.append(cross_exact[-1])
                if any(r['split'] == 'train' for r in rs):
                    exact_seen.update(r['example_id'] for r in rs if r['split'] != 'train')
    by_type = defaultdict(set)
    for r in rows:
        by_type[r['text_type']].add(normalize(r['text']))
    near_pairs = []
    for text_type, text_set in sorted(by_type.items()):
        texts = sorted(text_set)
        vocab = {}
        indices, indptr, sizes = [], [0], []
        for text in texts:
            words = re.findall(r'\w+', text)
            shingles = {tuple(words[i:i+5]) for i in range(max(0, len(words)-4))}
            cols = []
            for s in sorted(shingles):
                if s not in vocab:
                    vocab[s] = len(vocab)
                cols.append(vocab[s])
            indices.extend(cols)
            indptr.append(len(indices))
            sizes.append(len(cols))
        m = csr_matrix((np.ones(len(indices), dtype=np.int32), indices, indptr), shape=(len(texts), len(vocab)))
        overlaps = (m @ m.T).tocoo()
        for a, b, intersection in zip(overlaps.row, overlaps.col, overlaps.data):
            if a >= b or min(sizes[a], sizes[b]) < 26:
                continue
            sim = float(intersection / (sizes[a] + sizes[b] - intersection))
            if sim < config['near_duplicate_jaccard']:
                continue
            ra, rb = by_text[texts[a]], by_text[texts[b]]
            common_tasks = {r['task_id'] for r in ra} & {r['task_id'] for r in rb}
            for tid in sorted(common_tasks):
                aa = [r for r in ra if r['task_id'] == tid]
                bb = [r for r in rb if r['task_id'] == tid]
                if not any(x['agreement_id'] != y['agreement_id'] for x in aa for y in bb):
                    continue
                near_pairs.append({'task_id': tid, 'text_type': text_type, 'jaccard': sim,
                    'left_sha256': sha(texts[a]), 'right_sha256': sha(texts[b]),
                    'left_example_ids': [r['example_id'] for r in aa],
                    'right_example_ids': [r['example_id'] for r in bb],
                    'splits': sorted({r['split'] for r in aa + bb})})
                for left, right in [(aa, bb), (bb, aa)]:
                    if any(r['split'] == 'train' for r in left):
                        near_seen.update(r['example_id'] for r in right if r['split'] != 'train')
    write_jsonl('reports/exact_overlap.jsonl', cross_exact)
    write_jsonl('reports/near_overlap.jsonl', near_pairs)
    write_jsonl('reports/conflicting_duplicate_labels.jsonl', conflict)
    write_json('manifests/train_overlap.json', {'exact': sorted(exact_seen), 'near': sorted(near_seen)})
    return {'cross_agreement_exact_task_groups': len(cross_exact),
            'cross_split_exact_task_groups': sum(len(x['splits']) > 1 for x in cross_exact),
            'cross_agreement_near_task_pairs': len(near_pairs),
            'cross_split_near_task_pairs': sum(len(x['splits']) > 1 for x in near_pairs),
            'conflicting_exact_label_groups': len(conflict),
            'validation_test_examples_with_train_exact_overlap': len(exact_seen),
            'validation_test_examples_with_train_near_overlap': len(near_seen)}


def prepare(config):
    if Path('manifests/freeze.json').exists():
        # A fresh checkout carries frozen manifests but not redistributed contract passages.
        # Rehydrate only missing data from pinned official bytes and the EXISTING assignments.
        repo = fetch(config)
        assignments = read_json('manifests/agreements.json')['assignments']
        excluded = {r['example_id'] for r in read_jsonl('manifests/excluded.jsonl')}
        recovered = []
        for split,name in zip(SPLITS,('train','dev','test')):
            source = repo/f'data/MAUD_{name}.csv'
            dst = Path('data/official')/source.name
            if not dst.exists():
                dst.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(source,dst)
            for i,row in enumerate(csv.DictReader(dst.open())):
                row['official_split']=split
                row['official_row']=i+2
                row['example_id']=f'{name}:{i+2}:{sha(json.dumps(row,sort_keys=True))[:16]}'
                if row['example_id'] in excluded:
                    continue
                recovered.append({**row,'task_id':row['id'],'agreement_id':row['contract_name'],
                                  'split':assignments[row['contract_name']],'label':int(row['label']),
                                  'target_answer':mapped_answer(row)})
        for split in SPLITS:
            path=Path(f'data/heldout/{split}.jsonl')
            if not path.exists():
                write_jsonl(path,sorted((r for r in recovered if r['split']==split),key=lambda r:r['example_id']))
        verify_freeze()
        print('Existing frozen preparation verified; refusing to repartition.')
        return
    repo = fetch(config)
    all_rows = []
    official = {}
    hashes = {'vendor/maud/data.zip': file_sha(repo / 'data.zip')}
    for split, name in zip(SPLITS, ('train', 'dev', 'test')):
        source = repo / f'data/MAUD_{name}.csv'
        dst = Path('data/official') / source.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dst)
        hashes[str(dst)] = file_sha(dst)
        rows = list(csv.DictReader(source.open()))
        main_agreements = sorted({r['contract_name'] for r in rows if r['data_type'] == 'main'})
        official[split] = {'examples': len(rows), 'types': dict(Counter(r['data_type'] for r in rows)),
                           'main_agreements': main_agreements, 'main_agreement_count': len(main_agreements)}
        for i, row in enumerate(rows):
            row['official_split'] = split
            row['official_row'] = i + 2
            row['example_id'] = f'{name}:{i+2}:{sha(json.dumps(row, sort_keys=True))[:16]}'
            all_rows.append(row)
    assert [official[s]['examples'] for s in SPLITS] == [25827, 6753, 6651]
    assert all(official[s]['main_agreement_count'] == 152 for s in SPLITS)
    assert len(set.intersection(*(set(official[s]['main_agreements']) for s in SPLITS))) == 152
    schema = build_schema(all_rows)
    write_json('manifests/questions.json', schema)
    agreements = sorted({r['contract_name'] for r in all_rows if r['data_type'] == 'main'},
                         key=lambda x: sha(config['split_seed'] + ':' + x))
    assert sum(config['agreement_counts']) == len(agreements) == 152
    assignments = {}
    pos = 0
    for split, count in zip(SPLITS, config['agreement_counts']):
        for agreement in agreements[pos:pos+count]:
            assignments[agreement] = split
        pos += count
    write_json('manifests/agreements.json', {'seed': config['split_seed'], 'algorithm': 'sha256(seed:agreement_id), ascending; 106/23/23', 'assignments': assignments})
    kept, excluded = [], []
    for row in all_rows:
        reason = None
        if row['data_type'] == 'rare_answers':
            reason = 'No reliable cell-level source agreement in released counterfactual file'
        elif row['contract_name'] not in assignments:
            reason = 'Unrecognized source agreement'
        if reason:
            excluded.append({'example_id': row['example_id'], 'reason': reason, 'data_type': row['data_type'], 'official_split': row['official_split']})
            continue
        row = {**row, 'task_id': row['id'], 'agreement_id': row['contract_name'],
               'split': assignments[row['contract_name']], 'label': int(row['label']),
               'target_answer':mapped_answer(row)}
        assert row['text'].strip()
        assert schema[row['task_id']]['answers'][row['label']] == row['target_answer']
        kept.append(row)
    write_jsonl('manifests/excluded.jsonl', excluded)
    manifest_rows = []
    coverage = {}
    for split in SPLITS:
        rows = sorted((r for r in kept if r['split'] == split), key=lambda r: r['example_id'])
        write_jsonl(f'data/heldout/{split}.jsonl', rows)
        manifest_rows.extend({k: r[k] for k in ('example_id', 'agreement_id', 'split', 'task_id', 'data_type', 'official_split', 'official_row')} for r in rows)
    write_jsonl('manifests/examples.jsonl', manifest_rows)
    for tid, task in schema.items():
        info = {'answers': task['answers']}
        for split in SPLITS:
            rs = [r for r in kept if r['task_id'] == tid and r['split'] == split]
            info[split] = {'examples': len(rs), 'agreements': len({r['agreement_id'] for r in rs}),
                          'label_examples': [sum(r['label'] == i for r in rs) for i in range(len(task['answers']))],
                          'label_agreements': [len({r['agreement_id'] for r in rs if r['label'] == i}) for i in range(len(task['answers']))]}
        train = info['train']
        info['eligible'] = (train['agreements'] >= config['eligible_min_train_agreements'] and
            sum(n >= config['eligible_min_class_train_agreements'] for n in train['label_agreements']) >= 2)
        info['unsupported_train_labels'] = [i for i, n in enumerate(train['label_examples']) if not n]
        info['rare_labels'] = [i for i, n in enumerate(train['label_examples']) if n / max(1, train['examples']) <= config['rare_label_train_frequency_max']]
        info['inadequate_support_labels'] = [i for i, n in enumerate(train['label_agreements']) if n < config['eligible_min_class_train_agreements']]
        coverage[tid] = info
    write_json('manifests/coverage.json', coverage)
    duplicates = duplicate_audit(kept, config)
    contract_hashes = {p.name: file_sha(p) for p in sorted((repo / 'data/contracts').glob('*.txt'))}
    write_json('manifests/contracts.json', contract_hashes)
    audit = {'official': official, 'official_main_agreement_intersection': 152,
             'task_count': len(schema), 'parent_question_count': len({s['question'] for s in schema.values()}),
             'eligible_tasks': sum(v['eligible'] for v in coverage.values()),
             'excluded': dict(Counter(r['data_type'] for r in excluded)),
             'heldout': {s: {'examples': sum(r['split'] == s for r in kept),
                              'agreements': sum(v == s for v in assignments.values()),
                              'types': dict(Counter(r['data_type'] for r in kept if r['split'] == s))} for s in SPLITS},
             'whole_contract_exact_duplicate_count': len(contract_hashes) - len(set(contract_hashes.values())),
             'duplicates': duplicates,
             'policy': 'All source-assigned rows retained, including shared boilerplate. Report overlap and separate novelty subset; no semantic deduplication or label edits.'}
    write_json('reports/data_audit.json', audit)
    for p in itertools.chain(Path('manifests').glob('*'), Path('data/heldout').glob('*'), Path('reports').glob('*overlap.jsonl'), [Path('configs/initial.json'), Path('docs/evaluation_protocol.md'), Path('src/maud_qwen/prompt.py')]):
        hashes[str(p)] = file_sha(p)
    write_json('manifests/freeze.json', {'dataset_commit': config['dataset_commit'], 'files': hashes})
    print(json.dumps({k: v for k, v in audit.items() if k != 'official'}, indent=2))


def measure_lengths(config):
    from transformers import AutoTokenizer
    from .common import read_jsonl
    from .prompt import prompt_ids
    verify_freeze()
    tokenizer = AutoTokenizer.from_pretrained(config['model_id'], revision=config['model_revision'])
    schema = read_json('manifests/questions.json')
    lengths = []
    for split in SPLITS:
        for row in read_jsonl(f'data/heldout/{split}.jsonl'):
            n = len(prompt_ids(tokenizer, row, schema))
            lengths.append({'example_id': row['example_id'], 'split': split, 'task_id': row['task_id'],
                            'prompt_tokens': n, 'supervised_tokens': len(tokenizer.encode(chr(65+row['label']), add_special_tokens=False))+1})
    maximum = max(r['prompt_tokens'] + max(r['supervised_tokens'], config['max_new_tokens']) for r in lengths)
    context = math.ceil(maximum / 256) * 256
    if context > 32768:
        raise ValueError(f'Full passages need {context} tokens; context redesign required before experiments')
    import numpy as np
    report = {'model_revision': config['model_revision'], 'context_length': context, 'truncated_examples': 0,
              'policy': 'Full instructions, choices and passage. Abort on overflow; no truncation.',
              'by_split': {s: dict(zip(['p50', 'p90', 'p95', 'p99', 'max'], map(float, np.percentile([r['prompt_tokens'] for r in lengths if r['split'] == s], [50, 90, 95, 99, 100])))) for s in SPLITS},
              'prompt_source_sha256': file_sha('src/maud_qwen/prompt.py'), 'freeze_sha256': file_sha('manifests/freeze.json')}
    if Path('manifests/context.json').exists():
        assert read_json('manifests/context.json') == report, 'Frozen context changed'
    write_json('manifests/context.json', report)
    write_jsonl('reports/token_lengths.jsonl', lengths)
    write_json('configs/resolved.json', {**config, 'context_length': context})
    print(json.dumps(report, indent=2))
