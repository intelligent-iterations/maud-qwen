# Experiment journal and error analysis

This is a condensed scientific record of the completed study. The full machine-readable validation and final comparisons are preserved in `results/published/`. Infrastructure logs and machine-specific operational history are outside the public package.

## 1. Establish the evaluation

**Hypothesis:** MAUD fine-tuning improves Qwen3-8B's predefined deal-point answers on agreements excluded from fine-tuning.

**Setup:** pin the official data and model revisions; verify answer mappings; freeze 106/23/23 whole-agreement partitions and the evaluation protocol before model experiments. Exclude ambiguous counterfactual provenance. Keep main/abridged relatives together. Estimate label support from training only. Preserve all official splits separately.

**Result:** 25,225 training, 5,043 validation and 5,283 test rows; 119 eligible tasks. No cross-split exact task/text duplicates. Near training overlap affects seven validation and 94 test rows; report a predeclared sensitivity subset. Full prompts fit 4,096 tokens with zero truncation.

**Decision:** use equal-task macro-F1 as primary, plus accuracy, rare recall/precision, invalid outputs and per-task/category/agreement diagnostics. The original MAUD AUPR requires a validated continuous option scorer and is not reported here.

## 2. Establish baselines and a recoverable run

**Change:** evaluate training-majority and untuned NF4 Qwen using the fixed non-thinking prompt and strict answer-letter grading. Test the longest-passage backward pass and checkpoint resumption before substantive training.

**Result:** validation majority macro-F1 0.4049, accuracy 81.96%, rare correctness 0/154. Untuned Qwen: 0.3531, 49.45%, 82/154. Initial full-model smoke showed divergent gradients; deterministic algorithms and runtime controls corrected the issue before substantive training. Later checkpoint replay reproduced 35 updates exactly.

**Decision:** retain the deterministic runtime, full-state checkpoints and measured throughput. A thermal monitor paused the original workload on telemetry failure; verified recovery resumed from retained state.

## 3. Normal six-epoch trajectory

**Change:** natural-frequency training at 3e-5 with rank-8 LoRA, alpha 16, dropout zero, batch 16 through accumulation, cosine scheduling and answer-only supervision. Evaluate after each epoch, stop after two consecutive nonimproving epochs, with six as the ceiling.

**Result:** validation macro-F1 progressed 0.6149 → 0.6294 → 0.6396 → 0.6525 → 0.6547 → 0.6514. All six epochs completed. Accuracy and rare recall had different trajectories; one metric did not determine the others.

**Interpretation:** epoch five won the predefined rule. E5-minus-E4 interval [−0.005140, 0.009257] and E6-minus-E5 interval [−0.008285, 0.001971] include zero. This does not establish that the fifth epoch is universally optimal or identify the cause of the final dip.

**Decision:** select epoch five using validation. No full learning-rate grid or three-seed training study was completed.

## 4. Controlled rare-answer weighting

**Hypothesis:** making rare-target examples contribute 3× loss weight would increase rare recall without reducing primary macro-F1.

**Change:** restore the same epoch-two checkpoint, optimizer/scheduler/RNG and original third-epoch order. Train one alternative third epoch, weighting the 716 rare training examples. Normalize weighted losses across the effective batch, not separately inside one-example microbatches.

| Validation metric | Normal E3 | Weighted E3 |
| --- | ---: | ---: |
| Macro-F1 | 0.639650 | 0.645621 |
| Accuracy | 88.578% | 89.034% |
| Rare correct | 59/154 | 56/154 |
| Rare precision | 62.105% | 59.574% |

**Result:** primary difference +0.005971, interval [−0.004001, 0.014810]. Rare cases had one correction and four regressions. The intervention did not meet the predefined rare-recall criterion. Neither invalid formatting nor runtime truncation explains the change.

**Interpretation:** this is an unsuccessful configuration, not evidence that all weighting strategies fail. Sparse agreement-level support may matter; a weighting multiplier cannot supply missing supervision or missing passage context.

**Decision:** preserve the branch and comparison; do not change its success rule after seeing results. Continue the normal trajectory under its existing policy. See [the original plan](../results/published/rare3x-epoch3.json) and [paired comparison](../results/published/weighted-vs-control-epoch3.json).

## 5. Freeze the candidate and evaluate test

**Change:** fix epoch-five adapter/selection hashes before any final-test predictions, then evaluate majority, untuned Qwen and the selected adapter. Use identical model quantization, prompt, context, decoding and scoring for both Qwen variants.

**Result:** primary macro-F1 0.3661 → 0.6788; accuracy 48.93% → 89.04%. Primary improvement +0.312774 with interval [0.261203, 0.332256]. Rare correctness 66/154 → 57/154; rare precision 4.83% → 54.81%. Incorrect rare predictions 1,300 → 47. Removing preflagged overlap rows retains the primary gain. Full results and per-category diagnostics are in [the test report](../results/published/final-test-report.md).

**Interpretation:** supports the narrow hypothesis for this fixed recipe and held-out population. A lower rare-recall point estimate and much higher precision describe different behavior. The exploratory rare-recall difference interval [−22.50, +11.91] percentage points includes zero. Agreement bootstrap does not measure training-seed variability or unknown pretraining exposure.

**Decision:** close this experiment. Test results do not change the chosen checkpoint, prompt or training settings.

## Error patterns and proposed studies

Rare targets are 2.84% of training examples. Some valid labels have little or no supporting training agreements. Validation review identified recurring majority-answer substitutions, fine distinctions between answer options, and abridged passages missing relevant source clauses. Inspected source rows matched the official CSV: missing source context is different from runtime truncation. These observations are not a dataset-wide annotation-quality estimate or a single causal diagnosis.

Unrun hypotheses:

1. Repeat the selected recipe with at least three independent seeds to measure training variability.
2. Audit training-passage sufficiency under predefined rules, then test a documented data variant on validation.
3. Test interactions between learning rate, duration and rare-example weight with a predefined small grid.
4. Reproduce an original LegalBERT/MAUD experiment on official splits and its scorer as a separate published baseline.

The original final test is now public. Future development uses validation; new confirmatory claims require previously unused agreements. A larger untuned API comparator and alternative adapter methods remain unrun.
