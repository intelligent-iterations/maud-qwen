# Methods and source record

Reviewed on 2026-09-06 before implementation. This project adapts published methods; it does not reproduce MAUD's original models or splits.

| Source | Inspected method/code | Adaptation |
| --- | --- | --- |
| [MAUD paper, v3](https://arxiv.org/html/2301.00876v3) | Sections 3–4, Appendix A.3–A.5 and A.12–A.13 | Predefined labels and conditional/binary subquestions; agreement-held-out evaluation replaces its example/task-stratified splits. |
| [Official MAUD code at 4640316](https://github.com/The-Atticus-Project/maud/tree/4640316078dcd370debb877f350c39a28b181ffe) | `specs.py`, `data.py`, `csv_builder.py`, `mega_tune.py`, `multi_head_model.py`, `pr_curves.py`, `eval_utils.py`, training scripts | Preserve official answer IDs and question/subquestion schema. Keep official files independently. |
| [QLoRA](https://arxiv.org/abs/2305.14314) | Frozen 4-bit base, NF4, double quantization, adapters across linear layers | NF4/double quantization and BF16. Ordinary PyTorch AdamW for the small adapter state; no claim to reproduce QLoRA's paged-optimizer experiments. |
| [Domaino1s](https://aclanthology.org/2025.findings-acl.171.pdf) | Appendix A training setup and B data construction | LoRA rank 8, alpha 16, dropout 0 are a Qwen-family precedent. Its model was Qwen2.5-7B-Instruct, using 4×48 GB GPUs and generated reasoning data. Neither its 120 epochs nor reasoning traces are copied. |
| [Qwen3-8B model card](https://huggingface.co/Qwen/Qwen3-8B/tree/b968826d9c46dd6066d109eabc6255188de91218) | Model architecture, requirements and non-thinking chat template | Pin model and tokenizer to `b968826d9c46dd6066d109eabc6255188de91218`; use `enable_thinking=False`. Greedy letter classification is our engineering choice. |
| [PEFT quantization guide](https://huggingface.co/docs/peft/developer_guides/quantization) | `prepare_model_for_kbit_training`, QLoRA targets | Use explicit Qwen attention/MLP projections, record every matched module, and apply identical nonquantized casting to baseline and tuned models. |
| [Transformers Trainer documentation, 4.57.1](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/trainer) | Checkpoint state, gradient accumulation, scheduler/optimizer settings | An explicit single-GPU loop makes answer-only loss, partial accumulation and resume position inspectable. Checkpoint components are saved and compared in a GPU smoke test. |
| [MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking/quickstart/) | Parameters, metrics, runs and artifact logging | SQLite tracking plus local artifacts; all storage stays on premises. |

## Published sampling versus this experiment

MAUD's `data.oversample` repeats each label to the largest label count within its task. `mega_tune.subroutine` applies it unconditionally for multitask training. The multitask loader cycles shuffled task queues; the source fixes 100 inner batches per epoch and groups four task batches. This differs from a natural pass over the dataset. We use natural-frequency sampling without oversampling, one seeded permutation of all retained training rows per epoch. Effective batch size is 16 except the final remainder, which is averaged over its actual examples. The journal must identify a future balanced-sampling experiment as a new candidate.

MAUD searched multitask encoder learning rates `{1e-5, 3e-5, 1e-4}` and epochs `{1,2,3,4,5,6}`. Single-task experiments instead searched update counts, and BigBird used a different grid. Our first Qwen candidate uses 3e-5 for up to six epochs; the other learning rates remain candidates, not completed experiments or demonstrated optima.

## Additional engineering choices

The config records cosine scheduling, 3% warmup rounded up to optimizer steps, AdamW beta=(0.9,0.999), epsilon=1e-8, weight decay=0.01, nonfused/nonforeach PyTorch implementation, gradient clipping=1.0, seed=42, TF32 disabled, SDPA, and non-reentrant gradient checkpointing. Microbatch=1 and accumulation=16 were verified on the recorded RTX 3090. A fresh runtime must repeat the GPU smoke checks.

Only the answer letter and EOS are supervised. An explicit label mask excludes the prompt and padding. The model materializes logits only for the final answer positions; a tiny Qwen test compares this loss against full masked causal cross-entropy. No reasoning text, generated legal interpretations, confidence targets or new labels are created.

MAUD's binary CSV rows retain the full parent answer list in `answer`; it is not a scalar binary target. `MultiBinaryDatasetSubQuestionSpec.answer_to_label` checks membership after splitting on comma-space, with 0=`<OTHER>` and 1=the named option. The preparation code checks that exact rule on every binary row, preserves the raw answer, and records a separate `target_answer`. Initial preparation correctly stopped on an inconsistent scalar mapping assumption; this was corrected before any scoring or model experiment.

Checkpoint selection uses validation question macro-F1, earliest epoch on ties, and patience 2 for consecutive epochs without strict improvement. Intermediate saves occur every 100 optimizer steps and at step 10 to establish an early durable checkpoint. Epoch checkpoints include the updated validation/selection state. Adapters, optimizer moments, scheduler, Python/NumPy/Torch/CUDA RNG, trainer position, tokenizer, config, code identity, and checksums are saved atomically. An incomplete `.writing` directory is never treated as a checkpoint.

No semantic deduplication is inferred from lexical similarity. Shared wording across distinct agreements is retained and explicitly reported, with an additional evaluation subset that excludes exact/near training overlap. The experiment answers whether this training procedure improves held-out MAUD performance; it does not establish that Qwen has never seen the agreements during pretraining.

## Runtime determinism correction

The first full-model smoke showed divergent gradients before its save/resume boundary. `configs/runtime.json` therefore enables deterministic Torch algorithms, cuDNN determinism and the CUDA BLAS workspace setting `:4096:8`; TF32 and cuDNN benchmarking stay disabled. Runtime configuration bytes are hashed into metadata and checked on resumption/comparison. This is an engineering correction, not a setting attributed to MAUD or Domaino1s. The original `1e-6` smoke tolerance is unchanged. See [PyTorch 2.11 reproducibility](https://docs.pytorch.org/docs/2.11/notes/randomness.html). Throughput is measured again after this correction.
