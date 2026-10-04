![XAYA-2B — probabilities, not prose.](assets/xaya-2b.svg)

A 2B multimodal decision model for **choices, ordinal scores, and yes/no probabilities**. Give it context, a question, and up to **256 candidates**, optionally with an image. Get a probability distribution over those candidates.

Use it for tool routing, RAG routing, support triage, multimodal classification, and rubric scoring.

## Install

Requires Python 3.11+, an NVIDIA GPU, and CUDA-enabled PyTorch.

```bash
git clone https://github.com/LikithMeruvu/xaya-2b.git
cd xaya-2b
python -m pip install -e .
```

You also need the existing checkpoint archive, `xaya_2b_final_release.zip`. Place it in this directory and extract it:

```bash
python -m zipfile -e xaya_2b_final_release.zip weights/xaya-2b
```

The weights are not bundled in this GitHub repository; a public weight download is still pending. The SDK loads your local adapter, head, and processor, and downloads the pinned Qwen base on first use.

## Mini SDK

```python
from xaya import XAYA

model = XAYA.from_pretrained("weights/xaya-2b")  # 4-bit NF4 by default

result = model.choice(
    state="A customer says their invoice was charged twice.",
    question="Which support queue should handle this?",
    options=["Billing", "Technical", "Account"],
)

print(result["option"])
print(result["probabilities"])  # same order as your candidates
print(result["latency_ms"])
```

Load the model once and reuse it. Each call performs one forward pass, returning candidate probabilities. It does not generate a prose answer.

```python
# Image + text choice
result = model.choice(
    state="", question="What is shown?",
    options=["Cat", "Dog", "Other"], image="photo.jpg",
)

# Ordinal scoring: probabilities over levels and their expected value
result = model.score(
    state="The reply resolves the customer's issue clearly.",
    question="Rate the reply's usefulness.",
    levels=[1, 2, 3, 4, 5],
    descriptions={1: "Not useful", 5: "Fully useful"},
)
print(result["expected_score"])

# Yes / no probabilities
result = model.yes_no(
    state="The request is to change an account password.",
    question="Does this request concern account access?",
)
print(result["yes"], result["no"])
```

The SDK is a reference wrapper around the original prompt and head. Reproduction of the historical JevBench fingerprint is pending; the benchmark scores below are not newly established scores for this wrapper.

## Speed

Earlier NF4 experiments on a **Tesla T4** measured:

| Input | Median request latency |
| --- | ---: |
| Text | 155–162 ms |
| Image + text | 326–356 ms |

Peak GPU memory in those experiments was about **2.5 GB**.

These are historical runtime measurements, not a guarantee for this SDK. Request length, image resolution, GPU, and runtime configuration affect latency. Model loading is separate from request time.

This SDK was also checked on a **Windows RTX 4050 Laptop GPU** using fallback kernels: short-text p50 **1,003 ms**, small-image p50 **1,558 ms**, and about **2.1 GB peak allocated GPU memory**. Both probes used 20 warmed requests. Different inputs and kernels make these separate measurements, not a controlled GPU comparison. See [SDK validation](docs/SDK_VALIDATION.md).

Measure the warmed SDK on your own GPU:

```bash
python examples/speed.py --model weights/xaya-2b --runs 20
python examples/speed.py --model weights/xaya-2b --image photo.jpg --runs 20
```

The script reports p50/p95 latency and peak allocated GPU memory. Its timing includes preprocessing, device transfer, the forward pass, and the returned probabilities.

## Historical results

| Evaluation | Accuracy | Examples | Scope |
| --- | ---: | ---: | --- |
| JevBench public development | 64.07% | 231 | Historical frozen result |
| Local text test | 79.8768% | 13,800 | Local held-out split |
| Financial sentiment | 62.37% | 11,000 | Fully held-out source |
| AI2D | 68.33% | 3,088 | External evaluation |
| A-OKVQA MC | 74.50% | 1,145 | External evaluation |
| RM-Bench | 58.69% | 11,943 | Pairwise adaptation |
| TruthfulQA | 68.10% | 790 | Binary adaptation |

On the documented JevBench public-231 comparison, Open-Jev-2B scored 64.94%, Open-Jev-9B 77.49%, and Open-Jev-27B v1.1 85.28%. On Hard-111, XAYA scored **45.05%**, compared with Open-Jev-2B at **41.44%**. These are historical public development results, not an official sealed leaderboard rank.

[Full inference tutorial](docs/INFERENCE.md) · [SDK source](xaya/) · [Model card](docs/MODEL_CARD.md) · [Release status](docs/RELEASE_STATUS.md) · [Reproducibility](docs/REPRODUCTION.md)

The training-data license audit and model distribution license are pending. See [license status](docs/LICENSE_AUDIT.md).
