# Security sweep — 2026-09-14

The publication surface was reviewed for credentials, internal infrastructure details and avoidable execution risks. This is a repository audit, not a penetration test or a certification of the historical training environment.

## Changes

- Removed the private-review banner from the README and retained release approval in contributor/security guidance.
- Added a pinned Gitleaks check, recursive artifact inspection, private-path/address checks and a local pre-push hook. No directories or historical findings are excluded from the secret detector.
- Expanded ignore rules for credentials, local operator material and scanner reports. Organization-specific search terms remain untracked.
- Restricted new GPU metadata to model, memory and driver fields; the original process table is no longer collected.
- Updated the test runner from pytest 8.4.2 to 9.0.3. Training pins remain the historical recipe.
- Removed the historical MLflow UI/server launch instruction after checking its DNS-rebinding advisory.

## Leak and artifact findings

Gitleaks 8.30.1 and a second offline TruffleHog 3.96.0 scan found no credential candidates after accounting for input-token SHA-256 fields. The default Gitleaks generic-token rule initially matched those provenance hashes. The new audit normalizes only the exact digest field in temporary inputs; original predictions, their hashes and all other fields remain unchanged. See [SECURITY.md](../SECURITY.md) for the scope and regression checks.

The sweep includes all local Git refs and history, commit metadata, current/staged files, all twelve compressed prediction sets, the original source tarball and the standalone HTML/ZIP's embedded downloads. Generic private-address/path rules and an operator-only list of internal terms found no disclosures. The source tarball uses generic root ownership metadata and the blog ZIP contains no archive/member comments. No history rewrite was needed.

All 41 tests passed after the pytest update, including nine security tests. Tests verify detection of synthetic secrets in archived/encoded data, deleted history and staged content, distinguish hashes from credentials in the same row, reject unsafe archive paths/links and enforce the reduced GPU telemetry query. No live credential verification was performed. No model training was started. Frozen manifests, saved predictions, result objects and blog artifacts remain byte-identical to the prior release.

## Dependency findings remain open

A direct-pin audit with pip-audit 2.10.0 against the PyPI advisory feed returned 60 records, representing **33 distinct package/advisory pairs across four packages** after the pytest update. Duplicate advisory records were combined by package and advisory ID. These are scanner matches, not 33 demonstrated exploits in this pipeline. The [structured report](dependency-audit.json) includes versions, identifiers, aliases and suggested fixes returned by the feed.

| Historical dependency | Distinct advisory IDs | Current disposition |
| --- | ---: | --- |
| MLflow 3.4.0 | 25 | Do not start its UI/server. Keep historical local artifacts; inspect the standalone reports. A patched tracking environment needs separate compatibility checks. |
| Transformers 4.57.1 | 6 | Historical reproduction pin; upgrade requires model-loading, answer-format and GPU resumption validation. |
| Accelerate 1.10.1 | 1 | Use only trusted model/checkpoint inputs; investigate the sharded-checkpoint advisory before deployment. |
| Torch 2.11.0 | 1 | Historical numerical stack; the scanner cites a JIT advisory. Applicability and a compatible upgrade still need review. |

The three direct CPU scoring dependencies and updated pytest had no returned advisories. **Transitive dependencies and the installed GPU environment were not audited.** Advisory feeds and version-range metadata can change; rerun the audit before installing a new environment. A clean leak check must not be presented as dependency approval.

The MLflow issue is documented in [GHSA-pgqp-8h46-6x4j](https://github.com/advisories/GHSA-pgqp-8h46-6x4j); the [upstream middleware change](https://github.com/mlflow/mlflow/pull/17910) adds protections against DNS rebinding and related browser-origin attacks. Loopback binding alone is insufficient for the historical server. The pytest update addresses the version range in [GHSA-6w46-j5rx-g56g](https://github.com/advisories/GHSA-6w46-j5rx-g56g).

To repeat the direct-pin advisory check, use a separate disposable environment with `pip-audit==2.10.0`, export the core/test/train pinned requirements from `pyproject.toml`, and run:

```bash
python -m pip_audit -r direct-requirements.txt --no-deps --disable-pip \
  --format json --output dependency-audit.json
```

This deliberately queries package metadata without installing the historical training environment. It is not a transitive dependency audit. Do not use `--fix` on the historical recipe and then imply that the updated stack produced the saved results.

## Operational limits

The local pre-push hook is installed in the reviewed checkout; fresh clones need `make security-hooks`. It can be bypassed and is not a server-side branch rule. GitHub Actions was disabled for the organization repository during this sweep, and no workflow or paid scanner was added. At the time of this September 14 audit, the repository remained private and no public release had been authorized.

The next security task is a separately versioned dependency-modernization candidate with a complete resolved-package audit and compatibility tests. Original experiment pins and evidence must remain identifiable. The subsequent September 15 approval and public automation are documented in [publication.md](publication.md); they do not resolve these dependency findings.
