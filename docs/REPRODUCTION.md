# Canonical reproduction protocol

The target is a genuine single-order run: **148/231**, Original **50/72**, Easy **48/48**, Hard **50/111**, and Brier **0.5611851962 ± 0.0015**. Use JevBench commit `9ec6f15a8773aaffbbb41e7d7e65a20599ab0cc3` and the frozen model identity in `release/model_lock.json`.

The available archive contains the historical summary but no raw historical predictions. The original continuation log records a two-worker invocation of `eval_jevbench_continue.py` with max length 8192 and 256 options. Its implementation is missing. The earlier cached-embedding notebook must not substitute for that evaluator.

The latest recovery notebook described in the handoff compares broad-v2 NF4 normal/reversed averaging against the same runtime with one option order. That single-order match remains a hypothesis. Obtain its real diagnostics before adding more runtime variants.

## Intake real diagnostics

```bash
python tools/assess_diagnostics.py /path/to/xaya_canonical_diagnostics.zip
```

Read `diagnostic-assessment/summary.json`, the profile assessments, and pairwise task differences. The assessor recomputes probabilities-based accuracy and Brier instead of accepting summary claims. It rejects incomplete counts, duplicate/unknown tasks, invalid probabilities, stored prediction conflicts, and inference failures. A two-order or unknown-order profile cannot become a one-order match.

Profile JSON may contain `predictions`, `results`, `records`, or `rows`; JSONL records are also accepted. Record IDs may use pinned task IDs or broad `jev-{tier}-{index}` IDs. Each prediction needs `id` or `source_id`, `probabilities` (or `probs`), and an integer `gold` (or `answer`). For one-order evidence, include `order_count: 1` in profile metadata, or a one-element `latencies_s` list in every record. Conflicting order evidence cannot pass. Summaries alone are never prediction evidence.

The supplied gold indices are not semantic proof. Compare normalized options and original expected answers. Reorder probabilities by candidate label before comparing different option mappings. Check NOUL criteria and Yes/No order, SCORE levels, prompt separator/clipping behavior, processor/chat template, padding, generation prompt, loader dtype/quantization, and benchmark normalization. Per-task differences do not establish their cause without controlled inputs and runtime settings.

The parser recognizes the known literal-backslash-n JSONL separator bug without editing input files or replacing sequences inside string values. It reports that observation.

## Release gate

After a one-order fingerprint match, preserve the full runtime/configuration and verify all available historical prediction evidence. Among matching single-order implementations, measure and choose the fastest T4 p50. Keep SDK, server, benchmark, and demo on one shared implementation. Then validate GPU parity when switching from the preserved PT head to safetensors. Record new serialization hashes separately.

Do not add image-token caps, use the 61.47% speed wrapper, or describe the 64.50% two-order run as canonical. No optimization is accepted without fresh prediction-level validation. No additional training is part of v1 recovery.
