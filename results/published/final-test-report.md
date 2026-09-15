# Frozen final-test comparison

Epoch 5 was fixed from validation before any test predictions. Both Qwen evaluations use the same pinned NF4 base, tokenizer, non-thinking prompt, full-passage context policy, greedy decoding and deterministic scorer. This test contains **5,283 examples from 23 held-out agreements**.

| System | Primary macro-F1 | Accuracy | Rare correct | Rare recall | Rare precision | Invalid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Training majority | 0.4016 | 81.13% | 0/154 | 0.00% | n/a | 0 |
| Untuned Qwen3-8B | 0.3661 | 48.93% | 66/154 | 42.86% | 4.83% | 6 |
| QLoRA epoch 5 | 0.6788 | 89.04% | 57/154 | 37.01% | 54.81% | 0 |

The tuned-minus-base primary difference is **+0.312774**, with paired 95% agreement-bootstrap interval **[0.261203, 0.332256]**. The selected fine-tune improves the primary metric on this frozen test set, with a positive paired agreement-bootstrap interval.

There are 2000 valid draws from 2000 requested (seed 42); 722 omit eligible tasks absent from their sampled agreements, under the frozen rule. These intervals do not include training-seed variability or uncertainty from unknown pretraining exposure.

Eligible tasks scored: 119; missing eligible tasks: []. F1 includes every fixed answer label; undefined class F1 is zero. Invalid answers count as wrong. Rare labels are defined only from training frequencies (at most 10%, including absent labels). MAUD minority-class AUPR remains deferred: generated letters are not meaningful probability scores.

## Category results

| Category | Base macro-F1 | Tuned macro-F1 | Test rows |
| --- | ---: | ---: | ---: |
| Conditions to Closing | 0.3615 | 0.5688 | 1053 |
| Deal Protection and Related Provisions | 0.2748 | 0.5660 | 1852 |
| General Information | 0.4129 | 0.7500 | 47 |
| Knowledge | 0.7208 | 0.9407 | 92 |
| Material Adverse Effect | 0.4213 | 0.7711 | 1855 |
| Operating and Efforts Covenant | 0.4472 | 0.9270 | 336 |
| Remedies | 0.0204 | 0.4783 | 48 |

## Runtime and supporting diagnostics

| System | Inference minutes | Examples/second | Peak allocated GiB | Peak reserved GiB |
| --- | ---: | ---: | ---: | ---: |
| base | 52.27 | 1.685 | 11.522 | 11.887 |
| epoch5 | 62.73 | 1.404 | 11.604 | 11.965 |

Excluding preflagged training lexical overlap leaves 5189 rows: base macro-F1 0.3668, tuned 0.6791. Runtime-truncated model rows: base 0, tuned 0. Latency covers recorded predictions; peak memory covers the current process segment if evaluation resumed.

The machine-readable verification includes per-question/class/category/agreement results, main/abridged and length strata, rare training-support strata, paired errors and secondary agreement-bootstrap intervals. Example-level raw predictions, hashes, labels, token IDs and timings remain in the local evaluation outputs and MLflow artifacts. These are results for predefined answers from supplied merger-agreement passages, not evidence of general legal expertise. Agreement-held-out scores are not directly comparable with published results on MAUD’s official row splits. No test result changes the selected checkpoint or training configuration.

[Full verification](final-test-verification.json) · [Registered comparison](comparison-test.json) · [Selection ledger](final-selection-ledger.json)
