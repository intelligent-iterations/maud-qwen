# Publication provenance

This is a curated snapshot of a completed local experiment by Intelligent Iterations. The original project history contained private machine deployment, network and operations records. That history was not pushed to the public repository. Results and scientific inputs retain their original identities.

## Preserved scientific material

- Official MAUD commit: `4640316078dcd370debb877f350c39a28b181ffe`.
- Qwen model/tokenizer revision: `b968826d9c46dd6066d109eabc6255188de91218`.
- Frozen input manifest SHA-256: `2102136c0be405200b0152951053f7678765711ed136b495eaae73d29d5ff655`.
- Original model/evaluator commit: `874c27a33e4f3982ed2241bbaf05aa612d5749b4`.
- Original selected adapter SHA-256: `a544b41c61871f08d86f07819531fd8dbf195a245c3b233c7adcfd56ada328b7`.
- Original selection SHA-256: `0b170b851921e905a6578b6b0a7e268e5633e419f1d47607be1807b20d6d328c`.

The recorded experiment commits identify the original archive; they are not commits in this curated Git history. `results/published/source-provenance.json` records source-module hashes and the archive snapshot identity. The pinned original model/evaluator source subset is also included as `results/published/model-source.tar.gz` for inspection; it excludes deployment code and private history.

Twelve prediction sets preserve the exact original JSONL bytes under lossless gzip compression, with both compressed and uncompressed hashes recorded. Predictions contain MAUD-derived labels and identifiers, plus raw model outputs and evaluation metadata, but no full passages. `prepare` reconstructs the original data from the official release and verifies its hashes.

Published result objects, selection, historical weighting plan and parent receipt remain under `results/published/`. Paths and original commit/MLflow IDs inside these historical records are provenance, not promises that the old deployment or every operational artifact is public. The included CPU audit validates the claims supported by the released predictions and frozen inputs.

## Public packaging changes

- Split installation into CPU scoring, test and training dependencies; preserve the recorded algorithm and relevant training pins.
- Keep historical results separate from new outputs and the new run's `manifests/final_selection.json`. The original selection does not require a private checkpoint to audit published scores.
- Add a CPU-only published-result audit and integrity tests.
- Add isolated weighted-parent registration. A new reproduction has new checkpoint hashes and a new code commit; registration captures those identities before branching. Loss weighting, ordering, schedule and GPU replay/resumption checks are retained.
- Adapt article builds to read archived public results. Replace internal deployment instructions with portable local commands.
- Exclude machine-specific SSH/Fleet launchers, network configuration, raw operational logs, model weights, checkpoint state files, caches and private Git history.

The preparation, prompt, metrics, training loop and evaluator modules retain the final archived implementation. The model loader now restricts GPU metadata collection to model, memory and driver fields; it no longer records the process table. This logging-only change does not alter weights, numerical operations or the recorded experiment. Compared with the earlier normal-run commit, `train.py` includes the subsequent optional weighted-loss extension: its default unweighted loss, optimizer and schedule are retained. Both versions are inspectable through the source archive and recorded hashes. The frozen prompt and evaluation protocol remain byte-identical. Public weighted-branch changes concern how a new reproduction's parent identity is registered and read. A new CUDA environment must pass its smoke gates; CPU publication verification is not a new full-model GPU run.

## Licensing

Original project code/documentation use Apache-2.0. MAUD-derived material retains CC BY 4.0 attribution. The official licensor's [license page](https://www.atticusprojectai.org/legal/) identifies MAUD's license; the pinned [Qwen model](https://huggingface.co/Qwen/Qwen3-8B/tree/b968826d9c46dd6066d109eabc6255188de91218) identifies its Apache-2.0 license. See [NOTICE](../NOTICE). No MAUD/Qwen endorsement is implied.

## Public history boundary

The project owner approved public release with a fresh Git history on September 15, 2026. Version `v0.2.0` starts from one audited, parentless commit. Private development branches, old release tags and repository operational records are not part of the public Git history. Historical code hashes inside scientific artifacts identify the original experiment; they do not import that history into this repository.

The original predictions, frozen manifests and result artifacts remain unchanged. Prior publication remains part of the scientific provenance, and the final test remains closed to tuning. The public release adds the documented security checks and repository automation without claiming a new training run or resolving every historical dependency advisory.
