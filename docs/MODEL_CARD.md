# XAYA-2B — probabilities, not prose.

**Draft for publication after canonical runtime validation and the training-data license audit.**

A 2B multimodal decision model for bounded choices, ordinal scores, and yes/no probabilities.

XAYA accepts state or context, a question, and a finite candidate set, optionally with an image. It returns candidate probabilities through CHOICE, SCORE, and NOUL primitives. The maximum candidate count is 256.

Potential applications include agent/tool routing, RAG routing, support triage, moderation, model routing, multimodal classification, rubric scoring, bounded recommendation, and confidence gates. Probability outputs require application-specific calibration and evaluation; they do not by themselves establish reliable confidence thresholds.

## Historical benchmark evidence

| Evaluation | Result | Tasks | Protocol |
|---|---:|---:|---|
| JevBench public development | 64.07% | 231 | Historical frozen result; Original 50/72, Easy 48/48, Hard 50/111; Brier 0.5611851962 |
| Local text test | 79.8768% | 13,800 | Local held-out split; Brier 0.3093 |
| Financial sentiment | 62.37% | 11,000 | Fully held-out source; Brier approximately 0.5928 |
| AI2D | 68.33% | 3,088 | Full external evaluation; Brier 0.4939; ECE15 0.1839 |
| A-OKVQA MC | 74.50% | 1,145 | Full external evaluation; Brier 0.4172; ECE15 0.1653 |
| RM-Bench | 58.69% | 11,943 | Pairwise adaptation; not official scalar RM-Bench protocol |
| TruthfulQA | 68.10% | 790 | Binary adaptation |

Local text primitive accuracies: CHOICE 84.15%, SCORE 68.58%, NOUL 91.37%. Internal image dev/test reached 100% and is internal/in-distribution evidence only. Independent results above are retained from the authoritative project handoff and have not been newly rerun in this release work.

The later 64.50% JevBench run averaged two option orders. It is a robustness ensemble rerun. The 61.47% speed-wrapper run altered runtime behavior and is not the v1 definition. No sealed Benchmark Heaven rank is claimed. Broken MMMU-Pro outputs, unrun MMLU-Pro/RewardBench2, and the nonofficial JF100 run are excluded from performance claims.

## Comparable historical public results

On the documented public-231 protocol: XAYA-2B 64.07%, Open-Jev-2B 64.94%, Open-Jev-9B 77.49%, Open-Jev-27B v1.1 85.28%. On Hard-111, XAYA-2B scored 45.05% versus Open-Jev-2B 41.44%. These are documented historical comparisons, not a current leaderboard or a claim of statistical significance.

## Model identity

- Base: `Qwen/Qwen3.5-2B`, revision `15852e8c16360a2fea060d615a32b45270f8a8fc`.
- Adapter layout: `qwen35-language-projections-v2`; LoRA r=32, alpha=64, with RSLoRA enabled in the exported adapter configuration.
- Recorded model lock: `d1fc1e1d3ddd1795dcf207263424b320bb4692ccec7b092c9977e6e27c6a591a`.
- Dataset manifest: `c6a0802778083e6664f54efae2fe927c3dae104c3b040ef5956d33cbb8a7ae57`.
- Seed 13; trained with native NF4/PEFT on two Tesla T4 GPUs; locked before benchmarking.

## Publication placeholders

The shared canonical runtime, tested SDK examples, serving instructions, final manifest, public URLs, and distribution license must be added after verification. Previously measured T4 timing is experimental runtime evidence; final canonical latency is pending. The original PT head remains available for historical reproducibility. Safetensors conversion has a separate hash and still needs end-to-end GPU parity verification.
