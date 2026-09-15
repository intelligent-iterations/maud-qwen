# Reproduction guide

There are two separate activities: recomputing the published scores from saved predictions, and training a new adapter with the recorded recipe. The first needs only CPU dependencies. The second needs a compatible CUDA GPU and fresh smoke/resumption checks.

Run all commands from the repository root. Python 3.12 is recommended; the original GPU experiment used Python 3.10. Both the public verification environment and original experiment identities are recorded separately.

## CPU result audit

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m maud_qwen prepare
python scripts/reproduce_results.py
```

Preparation fetches the official repository at commit `4640316078dcd370debb877f350c39a28b181ffe`. It reconstructs ignored data files and verifies every entry in `manifests/freeze.json`; the committed split assignments are reused, not regenerated. It does not download model weights.

The result audit verifies compressed and original JSONL hashes, labels, source passage hashes, answer mappings, raw-answer parsing, training-majority choices and paired prompt identities. It recomputes complete final-test score objects, the validation summary, the controlled weighting comparison and both 2,000-draw primary bootstrap intervals. It writes `outputs/reproduction/verification.json`. This is rescoring a published experiment, not a new blind test.

## Local tests

For CPU Torch on Linux, install its CPU wheel before the test extras. The same command is supported for macOS:

```bash
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[test]'
python -m pytest -q
```

Tests cover label mapping, fixed-label macro-F1, invalid answers, bootstrap, loss masks, a tiny random Qwen3 model's answer-position loss, weighting gradients, prediction integrity and recovery controls. They do not benchmark the 8B model.

## CUDA training environment

The recorded experiment used Torch `2.11.0+cu128`, CUDA 12.8, Transformers 4.57.1, PEFT 0.17.1, bitsandbytes 0.48.1 and MLflow 3.4.0. Training dependencies preserve the relevant historical pins. The [security audit](security-audit.md) identifies unresolved advisories in this historical stack; it is not a recommended deployment environment. Use trusted local inputs and keep historical tooling isolated. Dependency modernization requires fresh compatibility and GPU resumption checks. The private multipurpose container image and unrelated applications are not required or distributed.

```bash
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[train,test]'
python -m maud_qwen prepare
python -m maud_qwen lengths
python -m maud_qwen majority
bash scripts/run_first.sh
```

`lengths` downloads only the pinned tokenizer and checks the full-prompt context manifest. The 4,096-token limit includes instructions, choices, passage and answer reserve. Overflow is an error. Model inference/training then downloads Qwen3-8B at revision `b968826d9c46dd6066d109eabc6255188de91218`. Reuse the standard Hugging Face cache or set `HF_HOME` to a local cache directory. Budget disk space for the base weights, data, local MLflow artifacts and full resumable checkpoints.

`run_first.sh` runs, sequentially:

1. Frozen-input verification and local tests.
2. A 32-example untuned evaluation.
3. A backward pass on the longest training passage.
4. Eight-step uninterrupted/resumed training comparison, including adapter, optimizer, scheduler and RNG state.
5. Full untuned validation and a measured runtime estimate.
6. Natural-frequency training at 3e-5, validating after each epoch, up to six epochs with patience two.

Failed checks prevent substantive training. Smoke outputs must belong to the same code commit and frozen inputs. Use a fresh checkout/output directory for a new study; never overwrite a prior run's smoke evidence to bypass a failed check. Keep the code commit fixed during a run.

The public packaging was verified on CPU. It has not been re-run as a new 3090 training experiment. Historical GPU feasibility and resumption evidence describe the original experiment; a new machine must pass these gates independently. No performance claim is made for a different CUDA/driver/library stack.

## Thermal controls and measured runtime

The original machine used a 200 W GPU cap and a 3 GHz CPU cap. Training averaged about 26.6 seconds per optimizer step, with 1,577 updates per epoch and approximately 63 minutes of validation. This gives roughly 12–13 hours per epoch under those controls. Re-estimate from your own measured steps; the eight-step smoke is an initial estimate, not a substitute for observing sustained training.

On a supported NVIDIA Linux installation, the GPU cap can be set before starting:

```bash
sudo nvidia-smi --power-limit=200
nvidia-smi --query-gpu=name,memory.total,power.limit,temperature.gpu --format=csv
```

Confirm that the requested limit is supported by your device and monitor temperatures during the pilot. CPU frequency controls depend on the host; do not copy machine-specific frequency writes without adapting them.

`scripts/thermal_guard.py` and `configs/thermal-guard.json` retain the original Docker pause/resume guard. It expects a single GPU, Linux sysfs CPU frequency controls and an AMD `k10temp` sensor. Its threshold logic is tested, but it is not a universal hardware monitor. The plain-shell launcher does not start this Docker-specific guard automatically. Integrate and verify host-appropriate monitoring before unattended work. No script in the public quickstart changes power limits automatically.

## Resume a saved training run

Preserve the original output directory and all checkpoint files. Activate the same code revision and dependency environment. Resume into a new output directory to keep previous history and checkpoint names intact:

```bash
python -m maud_qwen train \
  --resume outputs/initial/checkpoint-100 \
  --output outputs/continuation
```

The checkpoint's frozen-data, configuration, context, runtime and code identities must match. Adapters, optimizer moments, scheduler, Python/NumPy/Torch/CUDA RNG and trainer position are restored. A partial `.writing` directory is not a checkpoint. Resume only trusted checkpoints produced by this project: optimizer and RNG files use PyTorch pickle serialization.

## Reproduce the weighted third-epoch branch

First complete normal epoch-three validation. Keep its epoch-two checkpoint, epoch-three control predictions and training history. Use an isolated checkout at the **same code commit**, with its own ignored outputs, and install the same environment. Reconstruct the same frozen data there.

In that branch checkout:

```bash
python -m maud_qwen prepare
python scripts/register_weighted.py --parent-project /path/to/normal-checkout
python -m maud_qwen.weighted_branch preflight --parent-project /path/to/normal-checkout
python -m maud_qwen.weighted_branch smoke --parent-project /path/to/normal-checkout
python -m maud_qwen.weighted_branch train --parent-project /path/to/normal-checkout
python -m maud_qwen.weighted_branch compare --parent-project /path/to/normal-checkout
```

Registration pins your new epoch-two checkpoint hashes and code identity in `outputs/weighted-registration.json`. It never changes the archived original plan. The smoke replays normal updates to verify parent/order/schedule, then checks weighted save/resume equivalence. The weighted loop, batch normalization, rare definition and third-epoch order are preserved from the original implementation. Do not run the two full-model processes concurrently on the same 3090.

The original plan and parent hashes remain under `results/published/`. Public registration is a portability change: the historical checkpoint hashes cannot identify a newly trained checkpoint. The weighting math is unchanged; a public-package GPU replay remains unmeasured until the smoke passes on your machine.

## MLflow

Training and majority evaluation use SQLite at `outputs/mlflow.db` and local artifacts by default. An existing server can be selected with `MLFLOW_TRACKING_URI`.

Do not start a tracking UI/server from the historical MLflow 3.4.0 environment, including on loopback: that version has a [DNS-rebinding advisory](https://github.com/advisories/GHSA-pgqp-8h46-6x4j). Use the standalone reports and CPU result audit to inspect this experiment. A replacement UI/server needs a separately audited patched environment and database compatibility check before use.

Preserve both `outputs/` and `mlruns/`. Recorded metadata includes code, model, input/config hashes, actual module names, precision, dependencies, hardware, throughput and metrics. The CPU published-result audit requires no MLflow installation. There are no paid API comparators in these commands.

## Final selection for a new reproduction

The published selection is stored at `results/published/final_selection.json`. It is never used as your new run's selection record. After validation-led iteration is complete:

```bash
python -m maud_qwen select --run-dir outputs/initial \
  --output manifests/final_selection.json
```

If training resumed into `outputs/continuation`, select from that run directory instead. The selector uses the highest validation macro-F1, with the earliest checkpoint on ties; it does not hard-code epoch five.

For a completed run in `outputs/initial`, `bash scripts/final_comparison.sh` performs selection and test comparison. If selection already exists, run the individual guarded commands instead:

```bash
python -m maud_qwen majority --split test --final-record manifests/final_selection.json
python -m maud_qwen evaluate --split test --final-record manifests/final_selection.json
# Use the adapter path recorded by your selection command:
python -m maud_qwen evaluate --split test --adapter outputs/initial/epoch-5 \
  --final-record manifests/final_selection.json
python -m maud_qwen compare --split test --final-record manifests/final_selection.json \
  --left outputs/untuned-qwen/test.jsonl --right outputs/tuned-qwen/test.jsonl
```

The epoch-five path above is an example; substitute the actual selected adapter. These commands require that adapter to exist and match its recorded hash. Published predictions cannot be substituted for a new inference run. The benchmark's test is now visible: repeating this fixed recipe is a reproduction, not a new independent confirmation. A study that iterates after reading these results needs previously unused agreements for its final claim.
