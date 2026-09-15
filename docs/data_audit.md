# Executed data audit

Source commit: `4640316078dcd370debb877f350c39a28b181ffe`. ZIP SHA-256: `75af5a33d038e9254864f043da38072490ffe11e8488d58d0a2dd39c8f554519`. Frozen manifest SHA-256: `2102136c0be405200b0152951053f7678765711ed136b495eaae73d29d5ff655`.

The executed audit independently confirms 25,827 / 6,753 / 6,651 official rows. **Every official split's main-data rows contain all 152 agreement IDs**, with a 152-ID three-way intersection. Official files remain byte-identical and separate.

| Agreement-held-out split | Agreements | Main rows | Abridged rows | Total |
| --- | ---: | ---: | ---: | ---: |
| Train | 106 | 14,366 | 10,859 | 25,225 |
| Validation | 23 | 3,145 | 1,898 | 5,043 |
| Test | 23 | 3,112 | 2,171 | 5,283 |

All 35,551 retained rows follow their named source. The 3,680 counterfactual rows are excluded because reliable cell-level source provenance is absent. No source is guessed from textual similarity. No partition was chosen using model performance.

There are 144 question/subquestion tasks and 92 parent questions. Binary labels were verified against membership in the retained parent answer list, using the original 0=`<OTHER>` / 1=named-option mapping. Multiple-choice integer/string mappings were checked consistently. The raw `answer` is separate from `target_answer`.

119 tasks meet the fixed training-support criterion. The other 25 remain in accuracy and diagnostic results. All 119 eligible tasks have validation examples. Eleven tasks have at least one valid answer absent from training; 36 have a label supported by fewer than three training agreements. Counts and flags are in `manifests/coverage.json`.

## Overlap and provenance

Normalized exact matching found two cross-agreement task/text groups, both within a split, and **zero cross-split exact task/text groups**. Exhaustive 5-word-shingle Jaccard >=0.90 matching found 364 cross-agreement pairs, including 165 crossing splits. Near training overlap affects 101 held-out rows: 7 validation and 94 test. They remain in the primary score and are excluded from a separate novelty subset. Edge records are in `reports/*_overlap.jsonl`. Numbers are retained; no semantic or pretraining leakage guarantee follows.

`contract_148.txt` and `contract_149.txt` are byte-identical Kimco/Weingarten source files. Their official passages, however, match different raw annotation records exactly: Weingarten/Kimco and Welbilt/Ali Group. Their consideration passages and labels differ. This appears to be an upstream source-file packaging duplication. Both IDs landed in **training** under the frozen hash assignment, so no split change was made. Duplicated full-contract files are not model inputs. See `reports/source_file_duplicate_review.json` for evidence.

## Complete prompt lengths

The pinned tokenizer measured instructions, question/subquestion, choices and full passage:

| Split | Median | P90 | P99 | Maximum |
| --- | ---: | ---: | ---: | ---: |
| Train | 617 | 1,292 | 2,402 | 3,922 |
| Validation | 674 | 1,346 | 2,326.7 | 2,899 |
| Test | 676 | 1,304 | 2,207 | 2,960 |

Context is fixed at **4,096 tokens**, including an eight-token generation reserve. **Zero rows are truncated or discarded for length.** Overflow remains a hard error. Per-example lengths are in `reports/token_lengths.jsonl`.

The original CPU audit passed **16 tests**: mapping integrity, invalid-output parsing, equal-question/fixed-label macro-F1 against scikit-learn, paired agreement bootstrap, prompt/padding masks, disjoint agreement partitions, frozen checksums, and answer-position loss versus full masked causal loss in a tiny Qwen3 model. This does not verify 8B GPU feasibility or checkpoint resumption.
