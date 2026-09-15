import json
from pathlib import Path

import pytest

from maud_qwen import cli, train


def test_new_run_selects_its_validation_winner_and_preserves_record(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    adapter = Path('outputs/initial/epoch-2'); adapter.mkdir(parents=True)
    (adapter / 'adapter_model.safetensors').write_bytes(b'fixture adapter identity')
    run = Path('outputs/initial')
    (run / 'completed.json').write_text(json.dumps({
        'best_checkpoint': str(adapter),
        'completed_epoch_validation': [
            {'epoch':1,'score':0.5,'checkpoint':'outputs/initial/epoch-1'},
            {'epoch':2,'score':0.7,'checkpoint':str(adapter)},
            {'epoch':3,'score':0.7,'checkpoint':'outputs/initial/epoch-3'}]}))
    checked = []
    monkeypatch.setattr(train, 'verify_checkpoint', lambda p, c: checked.append(p))
    monkeypatch.setattr(cli, 'metadata', lambda c: {'config':c})
    output = Path('manifests/final_selection.json')
    cli.select({'selection_metric':'question_macro_f1'}, run, output)
    before = output.read_bytes()
    assert json.loads(before)['selected_epoch'] == 2
    assert checked == [adapter]
    with pytest.raises(ValueError, match='already exists'):
        cli.select({'selection_metric':'question_macro_f1'}, run, output)
    assert output.read_bytes() == before


def test_test_inference_requires_a_new_selection_record():
    with pytest.raises(ValueError, match='Test is sealed'):
        cli.require_final_record({}, None)
