import base64
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

import pytest

SPEC = importlib.util.spec_from_file_location('security_check', Path(__file__).parents[1] / 'scripts/security_check.py')
security = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(security)


def audit_at(tmp_path):
    export = tmp_path / 'export'
    export.mkdir()
    return security.Audit(export)


def fake_secret():
    # Synthetic, offline-only detector fixture; never sent to a verification API.
    return 'gh' + 'p_' + hashlib.sha256(b'nonfunctional scanner fixture').hexdigest()[:36]


def test_nested_archives_and_embedded_downloads_are_inspected(tmp_path):
    audit = audit_at(tmp_path)
    path = '/'.join(['', 'home', 'sample-user', 'private-project'])
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        archive.writestr('payload.json.gz', gzip.compress(json.dumps({'path': path}).encode()))
    uri = 'data:application/zip;base64,' + base64.b64encode(buffer.getvalue()).decode()
    audit.inspect('article.html', uri.encode())
    assert any(x['rule'] == 'personal-absolute-path' and '!data-uri-' in x['path'] for x in audit.findings)


def test_zip_traversal_links_and_expansion_limits_fail_closed(tmp_path, monkeypatch):
    audit = audit_at(tmp_path)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        archive.writestr('../outside.txt', 'bad')
        link = zipfile.ZipInfo('link')
        link.external_attr = 0o120777 << 16
        archive.writestr(link, 'target')
    audit.inspect('bad.zip', buffer.getvalue())
    assert sum(x['rule'] == 'unsafe-archive-member' for x in audit.findings) == 2
    monkeypatch.setattr(security, 'MAX_BYTES', 1024)
    audit.inspect('large.gz', gzip.compress(b'x' * 1025))
    assert any(x['rule'] == 'inspection-limit' for x in audit.findings)
    assert not (tmp_path / 'outside.txt').exists()


def test_private_addresses_terms_and_controls_but_not_local_ui(tmp_path):
    audit = audit_at(tmp_path)
    address = str(security.PRIVATE_NETS[0].network_address + 123)
    audit.private_terms = [security.re.compile('sample-secret-host')]
    audit.inspect('config.txt', (address + '\n' + 'sample-secret-host' + '\n' + chr(0x202E)).encode())
    assert {x['rule'] for x in audit.findings} == {
        'private-network-address', 'private-operator-term', 'hidden-control-character'}
    other = security.Audit(tmp_path / 'export')
    other.inspect('README.md', b'mlflow ui --host 127.0.0.1\nAGENTS.local.md\n')
    assert not other.findings


def run_gitleaks(audit, tmp_path):
    if not shutil.which('gitleaks'):
        pytest.skip('Gitleaks is required by make security; not installed for unit tests')
    report = tmp_path / 'leaks.json'
    result = subprocess.run(['gitleaks', 'dir', str(audit.export), '--config',
        str(Path(__file__).parents[1] / '.gitleaks.toml'), '--redact', '--no-banner',
        '--ignore-gitleaks-allow', '--max-decode-depth', '5', '--report-format', 'json',
        '--report-path', str(report)], capture_output=True)
    assert result.returncode in (0, 1)
    return json.loads(report.read_text())


def test_digest_handling_keeps_secret_in_same_record_and_encoded_secret(tmp_path):
    audit = audit_at(tmp_path)
    digest = hashlib.sha256(b'prompt').hexdigest()
    row = {'prompt_tokens_sha256': digest, 'api_token': fake_secret()}
    raw = (json.dumps(row) + '\n').encode()
    audit.inspect('predictions.jsonl.gz', gzip.compress(raw))
    audit.inspect('encoded.txt', base64.b64encode(fake_secret().encode()))
    assert audit.normalized_digest_fields == 1
    findings = run_gitleaks(audit, tmp_path)
    assert any(x['RuleID'] == 'github-pat' for x in findings)
    assert len({x['File'] for x in findings}) == 2
    assert all(fake_secret() not in json.dumps(x) for x in findings)


def test_large_hash_only_predictions_have_no_false_secret_hits(tmp_path):
    audit = audit_at(tmp_path)
    row = json.dumps({'prompt_tokens_sha256': hashlib.sha256(b'prompt').hexdigest()}) + '\n'
    audit.inspect('predictions.jsonl.gz', gzip.compress((row * 6000).encode()))
    assert not run_gitleaks(audit, tmp_path)


def test_non_digest_token_value_is_not_suppressed(tmp_path):
    audit = audit_at(tmp_path)
    audit.inspect('bad.json', json.dumps({'prompt_tokens_sha256': fake_secret()}).encode())
    assert audit.normalized_digest_fields == 0
    assert run_gitleaks(audit, tmp_path)


def test_deleted_history_and_staged_content_are_scanned(tmp_path):
    repo = tmp_path / 'repo'
    repo.mkdir()
    def git(*args):
        return subprocess.run(['git', '-C', str(repo), *args], check=True, capture_output=True)
    git('init')
    git('config', 'user.name', 'Example reviewer')
    git('config', 'user.email', 'reviewer@example.com')
    p = repo / 'old.txt'
    p.write_text(fake_secret())
    git('add', '.')
    git('commit', '-m', 'Offline detector fixture')
    git('rm', 'old.txt')
    git('commit', '-m', 'Remove fixture')
    staged = repo / 'staged.txt'
    staged.write_text('/'.join(['', 'home', 'sample-user', 'private']))
    git('add', 'staged.txt')
    staged.write_text('clean working tree')
    audit = audit_at(tmp_path)
    counts = security.inspect_repository(repo, audit)
    assert counts['reachable_commits'] == 2
    assert any(x['path'] == 'index/staged.txt' for x in audit.findings)
    assert run_gitleaks(audit, tmp_path)


def test_unknown_binary_is_not_silently_skipped(tmp_path):
    audit = audit_at(tmp_path)
    audit.inspect('payload.bin', bytes([255, 254, 253]))
    assert audit.findings[0]['rule'] == 'uninspectable-payload'


def test_gpu_logging_does_not_request_process_table(monkeypatch):
    from maud_qwen.model import gpu_summary
    captured = {}
    def fake_output(command, **kwargs):
        captured['command'] = command
        return 'Example GPU, 24576 MiB, 000.00\n'
    monkeypatch.setattr('maud_qwen.model.subprocess.check_output', fake_output)
    assert 'Example GPU' in gpu_summary()
    assert captured['command'] == ['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader']


def test_agent_link_exception_requires_empty_regular_target(tmp_path):
    assert security.allowed_agent_link(tmp_path, 'AGENTS.md', '120000', b'CLAUDE.md', '100644', b'')
    assert not security.allowed_agent_link(tmp_path, 'AGENTS.md', '120000', b'CLAUDE.md', '100644', b'payload')
    assert not security.allowed_agent_link(tmp_path, 'AGENTS.md', '120000', b'../CLAUDE.md', '100644', b'')
    assert not security.allowed_agent_link(tmp_path, 'AGENTS.md', '120000', b'CLAUDE.md', '120000', b'')
    assert not security.allowed_agent_link(tmp_path, 'other-link', '120000', b'CLAUDE.md', '100644', b'')
