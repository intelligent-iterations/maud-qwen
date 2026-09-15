#!/usr/bin/env python3
"""Pin a new reproduction's epoch-two parent in an isolated weighted workspace."""
import argparse
from pathlib import Path

from maud_qwen.common import file_sha, metadata, read_json, verify_freeze, write_json
from maud_qwen.train import verify_checkpoint


def register(parent_project):
    parent_project = Path(parent_project).resolve()
    if parent_project == Path.cwd().resolve():
        raise ValueError('Use a separate checkout for the weighted branch')
    destination = Path('outputs/weighted-registration.json')
    if destination.exists() or any(Path('outputs/initial').glob('*')) or any(Path('outputs/smoke').glob('*')):
        raise ValueError('Registration requires a fresh branch workspace')
    verify_freeze()
    base = read_json('configs/resolved.json')
    plan = read_json('configs/rare3x-epoch3.json')
    parent = parent_project / plan['parent_checkpoint']
    verify_checkpoint(parent, base)
    state = read_json(parent / 'trainer_state.json')
    expected = {'epoch': 2, 'row_offset': 0, 'global_step': 3154, 'planned_steps': 9462, 'steps_per_epoch': 1577}
    if any(state.get(k) != v for k, v in expected.items()):
        raise ValueError('Parent is not the complete epoch-two checkpoint for this protocol')
    if not (parent_project / plan['control_predictions']).exists():
        raise ValueError('Complete normal epoch-three validation before registering the branch')
    experiment = read_json(parent / 'experiment.json')
    if experiment['config'] != base or experiment['code_commit'] != metadata(base)['code_commit']:
        raise ValueError('Use the same committed code and base configuration as the normal parent')
    plan = {**plan, 'parent_code_commit': experiment['code_commit']}
    receipt = {'parent_checkpoint': plan['parent_checkpoint'],
               'checkpoint_hashes_sha256': file_sha(parent / 'checkpoint_hashes.json'),
               'trainer_state_sha256': file_sha(parent / 'trainer_state.json'),
               'experiment_sha256': file_sha(parent / 'experiment.json'),
               'parent_code_commit': experiment['code_commit'], 'global_step': 3154, 'epoch': 2}
    write_json(destination, {'parent_project': str(parent_project), 'plan': plan, 'parent_receipt': receipt})
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-project', required=True)
    args = parser.parse_args()
    print(register(args.parent_project))
