# Run XAYA-2B

## 1. Prepare the environment

Use Python 3.11 or newer, an NVIDIA GPU, and CUDA-enabled PyTorch. If PyTorch is not installed, use the command for your platform from [PyTorch's installation page](https://pytorch.org/get-started/locally/). Then install the mini SDK from the repository:

```bash
git clone https://github.com/LikithMeruvu/xaya-2b.git
cd xaya-2b
python -m pip install -e .
python -c "import torch; print(torch.cuda.is_available())"
```

The last command must print `True`. This reference loader supports CUDA inference; it has no CPU or Apple Silicon implementation. Optional acceleration kernels and compilation are not required by the SDK.

## 2. Extract your existing checkpoint

Use the original `xaya_2b_final_release.zip` archive. The repository contains code and documentation; it does not include the trained binaries or a public weight download yet.

```bash
python -m zipfile -e xaya_2b_final_release.zip weights/xaya-2b
```

Pass the directory containing these files to the SDK:

```text
weights/xaya-2b/
  release_config.json
  decision_head.pt
  adapter/
    adapter_config.json
    adapter_model.safetensors
  processor/
    tokenizer.json
    ...
```

If your archive has an enclosing folder, point the SDK to that inner folder. You can also pass the Kaggle checkpoint directory directly, or any existing extracted directory. The base model is downloaded from Hugging Face on first load and cached; the adapter, head, and processor come from your checkpoint.

## 3. Load once, decide repeatedly

```python
from xaya import XAYA

model = XAYA.from_pretrained("weights/xaya-2b")

result = model.choice(
    "A customer says their invoice was charged twice.",
    "Which support queue should handle this?",
    ["Billing", "Technical", "Account"],
)

for candidate, probability in zip(["Billing", "Technical", "Account"], result["probabilities"]):
    print(f"{candidate}: {probability:.1%}")
print("Selected:", result["option"])
```

The result contains a zero-based `index`, the selected `option`, probabilities in the supplied option order, and `latency_ms`. Probabilities sum to approximately one. The examples print the actual computed values; no expected predictions are supplied.

Keep the instance alive across requests. Loading it inside every request adds model download/loading time and repeated GPU allocation.

Candidates may also carry descriptions:

```python
result = model.choice(
    state="A customer cannot log in after resetting their password.",
    question="Which team should handle this?",
    options=[
        {"label": "Billing", "description": "Invoices, payments, and refunds"},
        {"label": "Account", "description": "Sign-in and account access"},
    ],
)
```

## 4. Include an image

```python
result = model.choice(
    state="",
    question="What is shown in the image?",
    options=["Cat", "Dog", "Other"],
    image="photo.jpg",
)
```

`image` is a local file path. Images are processed using the checkpoint's processor without an added image-token cap. Choose candidates suited to your task and evaluate accuracy on representative images.

## 5. Score or ask a yes/no question

```python
rating = model.score(
    state="The response is clear, accurate, and resolves the issue.",
    question="Rate the response quality.",
    levels=[1, 2, 3, 4, 5],
    descriptions={1: "Poor", 3: "Adequate", 5: "Excellent"},
)
print(rating["probabilities"])
print(rating["expected_score"])

answer = model.yes_no(
    state="The request concerns a failed payment.",
    question="Should the billing team handle this request?",
)
print(answer["yes"], answer["no"])
```

`score()` uses the SCORE primitive, ordered numeric candidate levels, and optional descriptions keyed by level. `expected_score` is the probability-weighted mean of those levels, which can be fractional. `yes_no()` uses NOUL with the explicit candidate order `Yes`, `No`. These are SDK conventions; no official benchmark adaptation is implied.

## 6. Choose loading settings

The default is native 4-bit NF4, double quantization, and FP16 computation. This keeps the adapter and decision head unchanged and quantizes the base for loading. Right padding, generation prompt ON, the original decision prompt, and one option order are used. The head uses the original PT serialization with restricted `weights_only=True` loading.

```python
model = XAYA.from_pretrained(
    "weights/xaya-2b",
    load_in_4bit=True,
    device="cuda:0",
    max_length=8192,
)
```

FP16 base loading is also exposed for reproduction work and needs more GPU memory:

```python
model = XAYA.from_pretrained("weights/xaya-2b", load_in_4bit=False)
```

An over-limit input raises an error so you can shorten it explicitly. No automatic component clipping, option rotation, averaging, image-token cap, or training is performed. The loader follows the [Qwen multimodal processor interface](https://huggingface.co/docs/transformers/en/model_doc/qwen3_5), with a decision head instead of text generation.

This reference wrapper does not establish historical JevBench parity. Preserve [reproduction evidence](REPRODUCTION.md) separately from simple inference smoke tests.

## 7. Measure latency

```bash
python examples/speed.py --model weights/xaya-2b --runs 20
python examples/speed.py --model weights/xaya-2b --image photo.jpg --runs 20
```

The script performs three warmup calls, synchronizes CUDA, then reports the median, p95, and peak allocated GPU memory. Timing covers processor work, device transfer, model forward computation, and returned probabilities. It excludes model loading and warmup. Its small built-in prompt is a speed probe, not an accuracy benchmark. Peak allocation differs from total process VRAM reported by system tools.

Earlier T4 NF4 experiments measured 155–162 ms text p50, 326–356 ms vision p50, and around 2.5 GB peak VRAM. Those figures belong to the previous experimental runtime; this SDK's speed must be measured on the actual device and inputs.

The current SDK ran on a Windows RTX 4050 Laptop GPU with PyTorch fallback kernels: short-text p50 1,002.8215 ms and p95 1,007.34025 ms; a synthetic 224×224 image p50 1,558.15 ms and p95 1,570.7945 ms. Each probe used 20 warmed requests. Peak allocated GPU memory was approximately 2.1 GB. See [validation and limitations](SDK_VALIDATION.md), including the observed NOUL example error.

## Troubleshooting

- **Missing checkpoint files:** point to the extracted directory containing `release_config.json`, not the ZIP or its parent.
- **CUDA unavailable:** install CUDA-enabled PyTorch for your environment before installing this SDK.
- **Out of memory:** close other GPU-heavy applications, keep 4-bit loading enabled, and shorten the input or use a smaller image explicitly. The historical 2.5 GB figure is not an upper bound for all inputs.
- **Context limit:** shorten the request explicitly. The mini SDK accepts one image and one decision per call; it does not implement batching.
- **Different benchmark results:** an inference smoke test proves execution, not canonical parity. See [release status](RELEASE_STATUS.md) for the known reproduction failure.
