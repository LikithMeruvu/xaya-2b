# Preserved runtime references

`inference.py` is the exact source from the original frozen release archive. `broad_benchmark_worker.py` was recovered as a literal source string from the supplied broad benchmark notebook without executing notebook cells.

The original loader uses FP16, generation prompt ON, and no explicit component clipping. The broad worker uses double-quantized NF4, right padding, head/tail token clipping, and two option orders by default. Both use literal backslash-n prompt separators. Do not normalize these separators while attempting reproduction.

These references are not the verified canonical runtime. The historical benchmark used `eval_jevbench_continue.py`; its source and historical per-task predictions are still needed. The original minimal loader has not been validated with the safetensors head.

`requirements.txt` preserves the original dependency declarations. The repository's offline diagnostic tools do not require these GPU dependencies.
