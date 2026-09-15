#!/usr/bin/env bash
# Linux x64 binary pinned by release version and publisher-provided SHA-256.
set -euo pipefail
: "${RUNNER_TEMP:?Expected a disposable GitHub runner}"
: "${GITHUB_PATH:?Expected a GitHub Actions environment}"
tool_dir=$(mktemp -d "$RUNNER_TEMP/maud-gitleaks.XXXXXX")
trap 'rm -rf "$tool_dir"' EXIT
curl --fail --silent --show-error --location --proto '=https' --tlsv1.2 \
  https://github.com/gitleaks/gitleaks/releases/download/v8.30.1/gitleaks_8.30.1_linux_x64.tar.gz \
  --output "$tool_dir/gitleaks.tar.gz"
(cd "$tool_dir" && printf '%s  %s\n' '551f6fc83ea457d62a0d98237cbad105af8d557003051f41f3e7ca7b3f2470eb' 'gitleaks.tar.gz' | sha256sum --check --strict)
tar -xzf "$tool_dir/gitleaks.tar.gz" -C "$tool_dir" gitleaks
install -d "$RUNNER_TEMP/maud-tools"
install -m 0755 "$tool_dir/gitleaks" "$RUNNER_TEMP/maud-tools/gitleaks"
printf '%s\n' "$RUNNER_TEMP/maud-tools" >> "$GITHUB_PATH"
"$RUNNER_TEMP/maud-tools/gitleaks" version
