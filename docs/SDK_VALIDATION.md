# Mini SDK validation

The SDK installed locally and completed real GPU requests against the original frozen adapter and PT head. No training or checkpoint updates were performed. The pinned base revision was downloaded and loaded with NF4 double quantization.

## Checks completed

- 15 offline tests passed, including the diagnostic tests and new SDK checks.
- The prompt was compared byte for byte against the formatting functions in the preserved original `legacy/inference.py` for CHOICE, SCORE, and NOUL.
- Real GPU CHOICE, SCORE, NOUL, and image requests returned finite probabilities of the expected length that summed to approximately one.
- The checked-in quickstart and speed CLI both executed on the actual GPU.
- The reference file and benchmark integrity checks passed.

These checks establish installation, request execution, formatting, and output structure. They do not establish benchmark accuracy, calibration, or canonical JevBench parity.

## Observed example results

| Request | Observed output |
| --- | --- |
| Duplicate invoice → support queue | Billing |
| Response usefulness → levels 1–5 | Selected level 5; expected score about 3.69 |
| Synthetic solid-red image → dominant color | Red |
| Cannot sign in → does this concern account access? | No |

The last result is inconsistent with the ordinary meaning of the request. It remains recorded as an observed error. The cause is not established: do not infer a systematic label reversal, change NOUL ordering, or claim the primitive is validated from this single example. The reference SDK explicitly uses the candidate order `Yes`, `No` from the documented broad harness; historical normalization semantics still need investigation. Check task accuracy before using the output in a workflow.

## Measured local speed

Hardware: NVIDIA GeForce RTX 4050 Laptop GPU, Windows. Optional `causal_conv1d` and flash-linear-attention kernels were absent, so Transformers used reference PyTorch implementations.

| Probe | Runs after warmup | p50 | p95 | Peak allocated GPU memory |
| --- | ---: | ---: | ---: | ---: |
| Short text, built-in `examples/speed.py` request | 20 | 1,002.8215 ms | 1,007.34025 ms | 2.0896 GB |
| Synthetic 224×224 image, three color candidates | 20 | 1,558.15 ms | 1,570.7945 ms | 2.1023 GB |

Three warmups preceded each probe. Timing includes preprocessing, transfer, forward execution, and the probability result, with CUDA synchronization. It excludes loading and warmup. These are small-input execution probes, not benchmark evaluations or estimates of concurrent throughput. Allocated GPU memory is a PyTorch measurement, not total process VRAM.

The text probe was rerun in isolation using the committed speed script. The earlier local text timing from the all-interface test was not used because another loader was briefly active. The image probe followed cancellation of that loader. The historical T4 measurements belong to a different runtime and inputs and must remain labeled separately.

Environment used: torch `2.5.1+cu121`, transformers `5.17.0`, peft `0.21.1`, bitsandbytes `0.49.0`, accelerate `1.15.0`, safetensors `0.8.0`.
