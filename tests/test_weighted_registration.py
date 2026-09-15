import importlib.util
import json
from pathlib import Path

import pytest

from maud_qwen.common import file_sha
from maud_qwen.weighted_branch import registration


spec = importlib.util.spec_from_file_location('register_weighted', Path(__file__).parents[1] / 'scripts/register_weighted.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_registration_refuses_shared_workspace_and_existing_state(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match='separate checkout'):
        module.register(tmp_path)
    (tmp_path / 'outputs/smoke').mkdir(parents=True)
    (tmp_path / 'outputs/smoke/proof.json').write_text('{}')
    with pytest.raises(ValueError, match='fresh branch workspace'):
        module.register(tmp_path / 'parent')


def test_registration_pins_verified_new_parent_and_never_overwrites(tmp_path, monkeypatch):
    work = tmp_path / 'branch'; work.mkdir()
    parent = tmp_path / 'normal'; checkpoint = parent / 'outputs/initial/epoch-2'
    checkpoint.mkdir(parents=True)
    monkeypatch.chdir(work)
    config = {'learning_rate': 3e-5}
    plan = {'parent_checkpoint': 'outputs/initial/epoch-2', 'control_predictions': 'outputs/initial/validation-epoch-3.jsonl',
            'parent_code_commit': 'historical'}
    (work / 'configs').mkdir()
    (work / 'configs/resolved.json').write_text(json.dumps(config))
    (work / 'configs/rare3x-epoch3.json').write_text(json.dumps(plan))
    (checkpoint / 'trainer_state.json').write_text(json.dumps({'epoch':2,'row_offset':0,'global_step':3154,'planned_steps':9462,'steps_per_epoch':1577}))
    (checkpoint / 'experiment.json').write_text(json.dumps({'config':config,'code_commit':'new-commit'}))
    (checkpoint / 'checkpoint_hashes.json').write_text('{}')
    (parent / plan['control_predictions']).write_text('{}\n')
    verified = []
    monkeypatch.setattr(module, 'verify_freeze', lambda: None)
    monkeypatch.setattr(module, 'verify_checkpoint', lambda p, c: verified.append((p, c)))
    monkeypatch.setattr(module, 'metadata', lambda c: {'code_commit':'new-commit'})
    module.register(parent)
    assert verified == [(checkpoint, config)]
    recorded, receipt = registration(parent)
    assert recorded['parent_code_commit'] == 'new-commit'
    assert receipt['checkpoint_hashes_sha256'] == file_sha(checkpoint / 'checkpoint_hashes.json')
    with pytest.raises(ValueError, match='fresh branch workspace'):
        module.register(parent)
    with pytest.raises(ValueError, match='parent project differs'):
        registration(tmp_path / 'different')
