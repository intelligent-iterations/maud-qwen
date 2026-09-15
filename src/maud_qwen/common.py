import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess


def sha(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(p)


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    with tmp.open('w') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(p)


def metadata(config):
    def git(*args):
        return subprocess.check_output(['git', *args], text=True).strip()
    result = {'config': config, 'platform': platform.platform(),
              'code_commit': os.environ.get('MAUD_CODE_COMMIT') or git('rev-parse', 'HEAD'),
              'code_dirty': bool(git('status', '--porcelain')) if Path('.git').exists() else None,
              'freeze_sha256': file_sha('manifests/freeze.json'),
              'config_sha256': sha(json.dumps(config, sort_keys=True))}
    if Path('manifests/context.json').exists():
        result['context_manifest_sha256'] = file_sha('manifests/context.json')
    result['runtime_config'] = read_json('configs/runtime.json')
    result['runtime_config_sha256'] = file_sha('configs/runtime.json')
    return result


def verify_freeze():
    frozen = read_json('manifests/freeze.json')
    for path, expected in frozen['files'].items():
        if file_sha(path) != expected:
            raise ValueError(f'Frozen input changed: {path}')
    return frozen


def tracking(name, config):
    import mlflow
    uri = os.environ.get('MLFLOW_TRACKING_URI', 'sqlite:///' + str(Path('outputs/mlflow.db').resolve()))
    Path('outputs').mkdir(exist_ok=True)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment('maud-agreement-held-out-v1')
    run = mlflow.start_run(run_name=name)
    mlflow.log_dict(metadata(config), 'reproducibility.json')
    mlflow.log_artifacts('manifests', 'manifests')
    mlflow.log_params({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in config.items()})
    return mlflow, run
