#!/usr/bin/env python3
"""Local publication audit of Git history, staged/current files and embedded artifacts."""
import argparse
import base64
import gzip
import hashlib
import io
import ipaddress
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import tarfile
import tempfile
import zipfile

PINNED_GITLEAKS = '8.30.1'
MAX_BYTES = 64 * 1024 * 1024
MAX_TOTAL_BYTES = 512 * 1024 * 1024
MAX_DEPTH = 6
PATTERNS = {
    'personal-absolute-path': re.compile(r'(?:/(?:Users|home)/[^\s/"\'<>]+|[A-Za-z]:\\Users\\[^\s\\]+)'),
    'private-dns-name': re.compile(r'\b[\w.-]+\.(?:ts\.net|internal|local)(?![\w.-])', re.I),
    'url-credentials': re.compile(r'https?://[^\s/@:]+:[^\s/@]+@', re.I),
    'hidden-control-character': re.compile('[\u200b-\u200f\u202a-\u202e\u2060-\u206f\U000e0000-\U000e007f]'),
}
IPV4 = re.compile(r'(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])')
PRIVATE_NETS = [ipaddress.ip_network(n) for n in (
    '10.0.0.0/8', '172.16.0.0/12', '192.168.0.0/16', '100.64.0.0/10', '169.254.0.0/16')]
IPV6 = re.compile(r'(?<![\w:])(?:f[cd][0-9a-f]{2}|fe[89ab][0-9a-f]):[0-9a-f:]+', re.I)
DATA_URI = re.compile(r'data:[\w.+/-]+;base64,([A-Za-z0-9+/]+={0,2})')
DIGEST_FIELD = re.compile(r'(?<![\w])prompt_tokens_sha256": "[a-f0-9]{64}"')
FORBIDDEN_PARTS = {'.git', '.ssh', '.aws', '.codex', '.claude', 'secrets', '.secrets', '.operator', '.security-reports'}
FORBIDDEN_SUFFIXES = {'.pem', '.key', '.p12', '.pfx', '.token', '.secret', '.pt', '.pth', '.safetensors'}


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args])


def unsafe_path(name):
    path = PurePosixPath(name)
    leaf = path.name.lower()
    return (path.is_absolute() or '..' in path.parts or '\\' in name
            or any(p in FORBIDDEN_PARTS for p in path.parts)
            or path.suffix.lower() in FORBIDDEN_SUFFIXES
            or leaf == '.env' or leaf.startswith('.env.')
            or leaf in {'agents.local.md', 'claude.local.md', '.security-private-terms'}
            or leaf.startswith('credentials') or leaf in {'id_rsa', 'id_ed25519'}
            or 'cookies' in leaf or 'storage-state' in leaf)


def allowed_agent_link(root, name, mode, blob, target_mode, target_blob):
    """Only the public convention AGENTS.md -> empty regular CLAUDE.md is allowed."""
    return (name == 'AGENTS.md' and mode == '120000' and blob == b'CLAUDE.md'
            and target_mode in {'100644', '100755'} and target_blob == b'')


class Audit:
    def __init__(self, export, private_terms=()):
        self.export = Path(export)
        self.private_terms = [re.compile(re.escape(t), re.I) for t in private_terms if t.strip()]
        self.findings = []
        self.seen = set()
        self.payloads = 0
        self.total_bytes = 0
        self.normalized_digest_fields = 0
        self.export_map = {}

    def flag(self, rule, name, line=None):
        self.findings.append({'rule': rule, 'path': name, 'line': line})

    def inspect(self, name, data, depth=0):
        if unsafe_path(name):
            self.flag('sensitive-or-unsafe-filename', name)
        if depth > MAX_DEPTH or len(data) > MAX_BYTES:
            self.flag('inspection-limit', name)
            return
        identity = hashlib.sha256(data).hexdigest()
        if identity in self.seen:
            return
        self.seen.add(identity)
        self.total_bytes += len(data)
        if self.total_bytes > MAX_TOTAL_BYTES:
            raise ValueError('Aggregate inspection limit exceeded')
        self.payloads += 1
        source = io.BytesIO(data)
        try:
            if data.startswith(b'\x1f\x8b'):
                with gzip.GzipFile(fileobj=source) as stream:
                    child = stream.read(MAX_BYTES + 1)
                self.inspect(name + '!gzip', child, depth + 1)
                return
            if zipfile.is_zipfile(source):
                source.seek(0)
                with zipfile.ZipFile(source) as archive:
                    if archive.comment:
                        self.inspect(name + '!zip-comment', archive.comment, depth + 1)
                    for member in archive.infolist():
                        if unsafe_path(member.filename) or stat.S_ISLNK(member.external_attr >> 16):
                            self.flag('unsafe-archive-member', name + '!' + member.filename)
                            continue
                        if member.is_dir():
                            continue
                        if member.comment:
                            self.inspect(name + '!' + member.filename + '!comment', member.comment, depth + 1)
                        if member.file_size > MAX_BYTES or member.flag_bits & 1:
                            self.flag('uninspectable-archive-member', name + '!' + member.filename)
                            continue
                        with archive.open(member) as stream:
                            self.inspect(name + '!' + member.filename, stream.read(MAX_BYTES + 1), depth + 1)
                return
            # POSIX tar, including the original source subset.
            if len(data) >= 262 and data[257:262] == b'ustar':
                with tarfile.open(fileobj=source, mode='r:') as archive:
                    for member in archive:
                        if member.uname not in {'', 'root'} or member.gname not in {'', 'root'}:
                            self.flag('archive-owner-identity', name + '!' + member.name)
                        if member.pax_headers:
                            self.inspect(name + '!' + member.name + '!pax-metadata',
                                         json.dumps(member.pax_headers).encode(), depth + 1)
                        if unsafe_path(member.name) or not (member.isfile() or member.isdir()):
                            self.flag('unsafe-archive-member', name + '!' + member.name)
                            continue
                        if member.isfile():
                            with archive.extractfile(member) as stream:
                                self.inspect(name + '!' + member.name, stream.read(MAX_BYTES + 1), depth + 1)
                return
            text = data.decode('utf-8')
        except (UnicodeDecodeError, OSError, ValueError, EOFError, zipfile.BadZipFile, tarfile.TarError):
            self.flag('uninspectable-payload', name)
            return
        if '\x00' in text:
            self.flag('binary-or-nul-payload', name)
            return
        for rule, pattern in PATTERNS.items():
            for match in pattern.finditer(text):
                self.flag(rule, name, text.count('\n', 0, match.start()) + 1)
        for match in IPV4.finditer(text):
            try:
                address = ipaddress.ip_address(match.group())
            except ValueError:
                continue
            standard_cidr = any(address == network.network_address and
                                text[match.end():].startswith('/' + str(network.prefixlen))
                                for network in PRIVATE_NETS)
            if not standard_cidr and any(address in network for network in PRIVATE_NETS):
                self.flag('private-network-address', name, text.count('\n', 0, match.start()) + 1)
        for match in IPV6.finditer(text):
            self.flag('private-ipv6-address', name, text.count('\n', 0, match.start()) + 1)
        for pattern in self.private_terms:
            for match in pattern.finditer(text):
                self.flag('private-operator-term', name, text.count('\n', 0, match.start()) + 1)
        for index, match in enumerate(DATA_URI.finditer(text)):
            try:
                decoded = base64.b64decode(match.group(1), validate=True)
            except ValueError:
                self.flag('uninspectable-data-uri', name)
                continue
            self.inspect(name + '!data-uri-' + str(index), decoded, depth + 1)
        # Only this exact digest field is normalized in temporary scanner input.
        # Gitleaks chunks large JSONL records mid-field, making regex allowlists
        # unreliable. The source artifact and infrastructure scan stay byte-exact.
        projected, count = DIGEST_FIELD.subn('prompt_tokens_sha256": "SHA256_DIGEST"', text)
        self.normalized_digest_fields += count
        output = self.export / (identity + '.txt')
        output.write_text(projected)
        self.export_map[output.name] = name


def inspect_repository(root, audit, revisions=()):
    if git(root, 'rev-parse', '--is-shallow-repository').strip() != b'false':
        raise ValueError('Full history is required; shallow clones are not accepted')
    audit.inspect('ref-names.txt', git(root, 'for-each-ref', '--format=%(refname)'))
    for revision in revisions:
        if not re.fullmatch(r'[0-9a-f]{40,64}', revision):
            raise ValueError('Push revisions must be full object IDs')
    objects = git(root, 'rev-list', '--objects', '--all', *revisions).decode().splitlines()
    object_ids = [line.split(' ', 1)[0] for line in objects]
    types = subprocess.check_output(['git', '-C', str(root), 'cat-file', '--batch-check=%(objectname) %(objecttype)'],
                                    input=('\n'.join(object_ids) + '\n').encode()).decode().splitlines()
    commits = 0
    for entry, description in zip(objects, types):
        oid, kind = description.split()
        name = entry.partition(' ')[2]
        if kind == 'blob':
            audit.inspect('history/' + name, git(root, 'cat-file', 'blob', oid))
        elif kind in {'commit', 'tag'}:
            # Includes authors, messages and annotated tag metadata.
            audit.inspect(kind + '/' + oid, git(root, 'cat-file', kind, oid))
            commits += kind == 'commit'
        elif kind == 'tree':
            entries = {}
            for row in git(root, 'ls-tree', '-z', oid).split(b'\0'):
                if row:
                    fields, entry_name = row.split(b'\t', 1)
                    entries[entry_name.decode()] = fields.decode().split()
            for entry_name, (mode, _, entry_oid) in entries.items():
                if unsafe_path(entry_name):
                    audit.flag('sensitive-history-filename', 'tree/' + oid)
                target = entries.get('CLAUDE.md', ['', '', ''])
                if mode == '120000' and target[0] in {'100644', '100755'} and allowed_agent_link(
                        root, entry_name, mode, git(root, 'cat-file', 'blob', entry_oid),
                        target[0], git(root, 'cat-file', 'blob', target[2])):
                    continue
                if mode not in {'100644', '100755', '040000'}:
                    audit.flag('symlink-or-submodule', 'tree/' + oid)
    current = set()
    staged = {}
    for row in git(root, 'ls-files', '--stage', '-z').split(b'\0'):
        if row:
            fields, raw_name = row.split(b'\t', 1)
            staged[raw_name.decode()] = fields.decode().split()
    for row in git(root, 'ls-files', '--stage', '-z').split(b'\0'):
        if not row:
            continue
        fields, raw_name = row.split(b'\t', 1)
        mode, oid, stage = fields.decode().split()
        name = raw_name.decode()
        current.add(name)
        target = staged.get('CLAUDE.md', ['', '', ''])
        if stage == '0' and target[2] == '0' and target[0] in {'100644', '100755'} and allowed_agent_link(
                root, name, mode, git(root, 'cat-file', 'blob', oid),
                target[0], git(root, 'cat-file', 'blob', target[1])):
            continue
        if mode not in {'100644', '100755'} or stage != '0':
            audit.flag('unsafe-index-entry', name)
            continue
        audit.inspect('index/' + name, git(root, 'cat-file', 'blob', oid))
    current.update(n.decode() for n in git(root, 'ls-files', '--others', '--exclude-standard', '-z').split(b'\0') if n)
    for name in sorted(current):
        path = root / name
        if path.is_symlink():
            target = root / 'CLAUDE.md'
            if not (name == 'AGENTS.md' and os.readlink(path) == 'CLAUDE.md'
                    and target.is_file() and not target.is_symlink() and target.read_bytes() == b''):
                audit.flag('symlink', name)
        elif path.is_file():
            with path.open('rb') as source:
                audit.inspect(name, source.read(MAX_BYTES + 1))
    return {'reachable_commits': commits, 'current_paths': len(current)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--revision', action='append', default=[])
    args = parser.parse_args()
    root = args.root.resolve()
    reports = root / '.security-reports'
    reports.mkdir(mode=0o700, exist_ok=True)
    executable = shutil.which('gitleaks')
    if not executable:
        parser.error('Install Gitleaks ' + PINNED_GITLEAKS + ' before running this check')
    version = subprocess.check_output([executable, 'version'], text=True).strip().lstrip('v')
    if version != PINNED_GITLEAKS:
        parser.error('Expected Gitleaks ' + PINNED_GITLEAKS + '; found ' + version)
    terms_path = root / '.security-private-terms'
    terms = terms_path.read_text().splitlines() if terms_path.exists() else []
    with tempfile.TemporaryDirectory(prefix='maud-security-') as temporary:
        export = Path(temporary) / 'payloads'
        export.mkdir()
        audit = Audit(export, terms)
        counts = inspect_repository(root, audit, args.revision)
        env = {k: v for k, v in os.environ.items() if not k.startswith('GITLEAKS_')}
        result = subprocess.run([executable, 'dir', str(export), '--config', str(root / '.gitleaks.toml'),
            '--redact', '--no-banner', '--ignore-gitleaks-allow', '--gitleaks-ignore-path', temporary,
            '--max-archive-depth', '0', '--max-decode-depth', '5', '--report-format', 'json',
            '--report-path', str(reports / 'gitleaks.json')], capture_output=True, text=True, env=env)
        (reports / 'gitleaks.log').write_text(result.stdout + result.stderr)
        if result.returncode not in (0, 1):
            raise RuntimeError('Gitleaks failed; inspect the local redacted log')
        leaks = json.loads((reports / 'gitleaks.json').read_text())
        if result.returncode and not leaks:
            raise RuntimeError('Gitleaks exited unsuccessfully without a findings report')
        for leak in leaks:
            name = audit.export_map.get(Path(leak['File']).name, 'scanner-input')
            audit.flag(leak['RuleID'], name, leak.get('StartLine'))
        report = {**counts, 'code_commit': git(root, 'rev-parse', 'HEAD').decode().strip(),
            'gitleaks_version': version, 'unique_payloads': audit.payloads,
            'inspected_bytes': audit.total_bytes, 'normalized_digest_fields': audit.normalized_digest_fields,
            'private_terms_configured': bool(terms), 'findings': audit.findings,
            'secret_verification_network_calls': False, 'passed': not audit.findings}
        (reports / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))
        return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
