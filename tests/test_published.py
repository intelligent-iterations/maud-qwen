import gzip
import json

import pytest

from maud_qwen.common import file_sha, sha
from maud_qwen.published import read_archive, verify_prediction_rows, rare_stats


def test_compressed_predictions_preserve_original_bytes_and_reject_changes(tmp_path):
    raw = b'{"example_id":"one"}\n'
    path = tmp_path / 'predictions.jsonl.gz'
    path.write_bytes(gzip.compress(raw, mtime=0))
    entry = {'path': str(path), 'gzip_sha256': file_sha(path), 'sha256': sha(raw), 'rows': 1}
    assert read_archive(entry) == [json.loads(raw)]
    with pytest.raises(ValueError, match='Original JSONL'):
        read_archive({**entry, 'sha256': '0' * 64})
    path.write_bytes(gzip.compress(b'{}\n', mtime=0))
    with pytest.raises(ValueError, match='Compressed prediction'):
        read_archive(entry)


def test_correct_label_cannot_hide_changed_passage_or_raw_answer():
    row = {'example_id': 'one', 'agreement_id': 'a', 'task_id': 'q', 'label': 0,
           'split': 'test', 'data_type': 'main', 'text': 'Original passage.'}
    schema = {'q': {'answers': ['no', 'yes']}}
    prediction = {k: v for k, v in row.items() if k != 'text'} | {
        'prediction': 0, 'raw_output': 'A', 'valid_answers': ['no', 'yes'],
        'text_sha256': sha(row['text']), 'train_overlap': False}
    assert verify_prediction_rows([row], [prediction], schema, set()) == [prediction]
    with pytest.raises(ValueError, match='text_sha256'):
        verify_prediction_rows([{**row, 'text': 'Changed passage.'}], [prediction], schema, set())
    with pytest.raises(ValueError, match='Raw answer'):
        verify_prediction_rows([row], [{**prediction, 'raw_output': 'Answer: A'}], schema, set())


def test_rare_precision_does_not_use_gold_rare_denominator():
    rows = [{'task_id': 'q', 'label': y, 'prediction': p} for y, p in [(1, 1), (1, 0), (0, 1), (0, 1)]]
    result = rare_stats(rows, {'q': {'rare_labels': [1]}})
    assert result['recall'] == 0.5
    assert result['precision'] == 1 / 3
