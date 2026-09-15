# Contributing

Please include the problem, the proposed change and relevant validation in an issue or pull request. Changes to data processing, scoring or training should include a small test that could catch the original error.

## Local checks

Run `make security` before sharing a change and `make security-hooks` once per clone to install the pre-push check. See [SECURITY.md](SECURITY.md) for scanner setup and private vulnerability reporting.

Run commands from a repository checkout with Python 3.12. Install CPU Torch first on Linux to avoid pulling a CUDA build just for tests:

```bash
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[test]'
python -m maud_qwen prepare
python -m pytest -q
python scripts/reproduce_results.py
```

The result audit uses published predictions and never loads a model. Tests exercise a tiny randomly initialized Qwen model on CPU, not Qwen3-8B. GPU changes also require the documented full-model smoke and checkpoint-resumption checks on compatible hardware before a new training claim.

## Experiment boundaries

Keep `results/published/` and the frozen input manifests intact. Put new run outputs in an ignored `outputs/` directory. State your hypothesis, configuration, sampling, validation rule, outcome and next decision in a separate experiment record. Report unsuccessful candidates alongside successful ones.

This repository's final test is published and remains closed to tuning. Use validation for development and reserve previously unused agreements for a new confirmatory study. Do not present tuning against the published test as an independent test result.

Never commit credentials, private contract data, personal paths, model caches or checkpoint pickle files. Keep MAUD attribution when redistributing derived metadata. Run the local checks before pushing. Public CI repeats the CPU tests, result audit and leak checks; it does not train models, call model APIs, upload Actions artifacts or use dependency caches.

## Release approval

Public releases and mirrors require explicit project-owner approval and an audit of the exact candidate. See [SECURITY.md](SECURITY.md) for the release boundary.
