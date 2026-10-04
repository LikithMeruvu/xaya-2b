# Model and release status

The trained, frozen XAYA-2B checkpoint exists. The repository is presently a documentation and diagnostic repository, not a complete installable model release. There are no GitHub Release weight assets at this point.

## What exists

The original checkpoint is distributed in the project archive named `xaya_2b_final_release.zip`: trained adapter, original decision head, processor, release metadata, historical evaluation summaries, and reference inference code. The pinned Qwen base is a separate dependency. A model consists of the base plus this adapter and head; a diagram or README is not the model.

The original adapter/head/config hashes were checked against the historical log. No v1 training or weight modification has been performed. A separate safetensors head serialization was checked for exact tensor equality; its hash is separate from the original lock. End-to-end canonical runtime validation is still pending.

The supplied Kaggle run found its checkpoint at `/kaggle/input/datasets/kartxlegendyt/xaya-2b-final`. This records the notebook's model input location; it is not a statement that a public download or canonical final export is available.

## Latest supplied canonicalization output

| Profile | Process result | Overall | Original | Easy | Hard | Brier |
| --- | --- | --- | --- | --- | --- | --- |
| `nf4_gp1` | Failed, rc 2 | — | — | — | — | — |
| `nf4_gp0` | Completed, no match | 142/231 | 50/72 | 48/48 | 44/111 | 0.5946317165 |
| `fp16_gp1` | Failed, rc 2 | — | — | — | — | — |
| `fp16_gp0` | Failed, rc 2 | — | — | — | — | — |
| Required historical target | Single-order fingerprint | 148/231 | 50/72 | 48/48 | 50/111 | 0.5611851962 ± 0.0015 |

The successful process retained the Original and Easy counts and missed the historical Hard count by six. That aggregate difference does not identify six specific question IDs: more items can change in opposite directions. Raw probabilities are needed for a task-by-task comparison.

The notebook printed `NO PROFILE reproduced the locked XAYA v1 fingerprint`. Its export cell depends on a passing profile and was not demonstrated as executed. The attached paste contains summary output, not the 231 prediction records or the error tracebacks for the three failed profiles. The two longest embedded helper payloads are also truncated in the paste, so it is not a complete executable notebook source.

The pasted notebook's embedded contract uses a Brier tolerance of 0.005. The authoritative recovery contract is stricter, ±0.0015. This profile fails both tolerances and the accuracy/tier counts.

## What a complete release still needs

1. Actual diagnostic JSON and worker logs, plus the complete recovery notebook, to establish failures and per-question differences.
2. One frozen, single-order runtime that passes the historical counts and Brier gate, with saved predictions and exact dependency/runtime settings.
3. One shared runtime used by the SDK, server, benchmark, and demo; GPU validation of the head serialization.
4. A completed data/license audit and a deliberate distribution license.
5. Published weight assets, checksums, a working loading example, and the validated FINAL package.

Until those steps are complete, the checkpoint should be identified as frozen and its runtime as unvalidated. Its archive filename must not substitute for release evidence.
