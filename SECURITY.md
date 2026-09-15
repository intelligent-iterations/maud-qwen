# Security

Report suspected credentials, internal infrastructure disclosures or exploitable behavior privately to **security@intelligentiterations.com**. Include the affected commit, file and a description; do not put credentials or sensitive payloads in a public issue.

## Local leak check

Install [Gitleaks 8.30.1](https://github.com/gitleaks/gitleaks/releases/tag/v8.30.1) and Python 3.10 or later, then run:

```bash
make security
make security-hooks
```

The first command audits this checkout. The second runs the audit and installs a local pre-push hook. Every clone needs its own hook installation; the installer refuses to replace an unrelated hook. Missing tools, shallow history, unreadable artifacts and findings block the check. Reports remain in ignored `.security-reports/` and contain redacted findings. The scanner does not call credential-verification services.

The audit covers all locally reachable branches/tags, commit metadata, every historical blob, staged content, working copies and non-ignored new files. The hook also includes commits explicitly named by the push. It expands ZIP, gzip, tar and embedded base64 downloads, checks member paths without extracting to the filesystem, and rejects unsupported binary content or excessive expansion. Gitleaks also decodes common encodings up to five layers. These are bounded scans, not proof that arbitrary encoding or obfuscation cannot hide information.

Additional checks cover personal filesystem paths, private network addresses and DNS names, credential-shaped filenames, symlinks/submodules and hidden control characters. The sole symlink exception is `AGENTS.md` pointing to an empty regular `CLAUDE.md`. Standard private-network CIDR definitions and loopback addresses in local examples are allowed. Organization branding, public project URLs, research citations and reproducibility hashes are intentional public material.

For organization-specific names, maintain an **untracked** `.security-private-terms` file with one literal term per line. The audit checks those terms without including their values in its reports. Never publish a denylist of real hosts, customers or internal project names. Generic automated rules supplement a human review of infrastructure references and generated artifacts.

The exact `prompt_tokens_sha256` field containing a lowercase 64-character SHA-256 is normalized in temporary secret-scanner inputs. Gitleaks otherwise mistakes these input hashes for API tokens, including partial matches at its chunk boundaries. Original files and the infrastructure scan are unchanged. No prediction directory, whole line or other token field is excluded. Regression tests plant a synthetic token alongside the hash, in a malformed hash field, in archives and in encoded content to verify detection remains active.

## Execution boundaries

- This is a local research project, not a hosted inference service. The historical MLflow version must not be used to start a UI/server, including on loopback. Use the saved reports and CPU audit; any replacement service needs current security and database compatibility checks.
- Resume only trusted checkpoints. Historical optimizer and RNG state use pickle; loading an untrusted checkpoint can execute code. Hash matching establishes artifact identity, not trust in its producer.
- The optional Docker thermal guard uses the local Docker socket and can control containers. Run it only in a trusted operator context. It is not automatically started by the plain-shell quickstart.
- Experiment outputs can include local checkpoint paths and environment details. They are ignored by Git and must be curated and scanned before sharing. Current GPU telemetry records only model, memory and driver fields, excluding the process table.
- Historical training dependencies are pinned for reproduction. A clean secret scan is not a dependency-vulnerability certification; see the dated [audit](docs/security-audit.md).

## Release and automation

Public releases and mirrors require explicit project-owner approval after reviewing the exact candidate. Prepare a fresh, audited root for an initial public mirror rather than copying private branches, tags, issues, settings or operational history. Scan that candidate separately immediately before publication.

The local hook is a developer safeguard, not a server-enforced rule; a client or repository administrator can bypass it. The public repository uses pinned CI, CodeQL and OpenSSF Scorecard workflows on standard GitHub-hosted runners, guarded to run only while the repository is public and its automation variable is enabled. Weekly Dependabot updates cover Python and Actions. No Actions artifact uploads, dependency caches or self-hosted runners are configured. Do not add public pull-request execution on an internal runner. Any future CI or publishing integration needs its own reviewed execution and access boundaries.

The public release automation and branch protection are documented in [publication.md](docs/publication.md). Historical dependency advisories remain explicit; security badges report checks, not a blanket guarantee.
