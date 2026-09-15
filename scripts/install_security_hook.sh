#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
current=$(git config --get core.hooksPath || true)
if [[ -n "$current" && "$current" != .githooks ]]; then
  echo 'An existing hooksPath is configured. Integrate this pre-push check with it before replacing it.' >&2
  exit 1
fi
if [[ -z "$current" && -f "$(git rev-parse --git-path hooks/pre-push)" ]]; then
  echo 'An existing pre-push hook is installed. Integrate this check with it before replacing it.' >&2
  exit 1
fi
command -v gitleaks >/dev/null
python3 scripts/security_check.py
git config --local core.hooksPath .githooks
echo 'Local pre-push security check installed.'
