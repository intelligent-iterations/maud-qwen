# Public release and repository controls

Version 0.2.0 is the September 15, 2026 publication approved by the project owner. Its public Git history begins with a single parentless commit. The original private history and old tags are retained separately and are not ancestors or refs of this repository. Experiment hashes inside results remain scientific provenance.

## Checks

| Control | Scope |
| --- | --- |
| CI | Full-history and archive-aware leak scan, CPU tests, frozen-data reconstruction, and saved-prediction rescoring |
| OpenSSF Scorecard | Published repository assessment and live README badge; push, protection-change and weekly triggers |
| CodeQL | Python and JavaScript static analysis; pull requests, pushes and weekly runs |
| Dependabot | Weekly Python and Actions update pull requests; historical training changes still require compatibility review |
| Code owners | Named maintainers cover the repository and security/workflow configuration |
| Main protection | Required checks and owner review, stale-review dismissal, linear history, no force-push or branch deletion |
| Secret scanning | GitHub's public-repository scanner plus the project's local/CI Gitleaks scan |

Actions are pinned to full commit IDs, checkout credentials are not persisted, and jobs have explicit limited permissions and timeouts. The Gitleaks executable is pinned by version and SHA-256. Scorecard publishes its public assessment with GitHub OIDC; CodeQL/Scorecard submit SARIF to code scanning. There are no Actions artifact uploads or dependency caches.

Every job checks that the repository is public and `PUBLIC_AUTOMATION_ENABLED` is `true`. The private source repository keeps Actions disabled. Standard GitHub-hosted runners are used only for the public project, where their execution is [free](https://docs.github.com/en/billing/concepts/product-billing/github-actions). Public code scanning is [available without a private-repository Code Security license](https://docs.github.com/en/code-security/concepts/code-scanning/code-scanning). No paid GitHub security product, larger runner, model API or training compute is provisioned.

## Evidence and limits

Use the live workflow runs and Scorecard results linked from the README to inspect the actual status. A configured badge does not establish a passing run or a particular score. Fresh repositories may need time for the Scorecard API to index their first result.

The original numerical experiment and all saved predictions remain unchanged. The [September 14 dependency audit](security-audit.md) still identifies unresolved advisories in historical training packages. Modernizing that stack is separate work requiring a resolved-package audit and model compatibility checks. Publishing the reproducible experiment does not certify that historical environment for deployment.

Private credentials, deployment instructions, operator term lists and model checkpoints are excluded. Public pull requests do not run on an internal machine. New public releases require approval of the exact candidate and passing release checks.
