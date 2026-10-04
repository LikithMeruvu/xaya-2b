# Training-data license audit: pending

No distribution license is selected in this working bundle. The handoff identifies the base as Apache-2.0; that does not settle adapter/head distribution rights. The embedded adapter README's base metadata is not a completed training-data audit.

The verified training manifest lists CLINC150, Banking77, MASSIVE, Civil Comments, AEGIS, HH-RLHF, OASST2, NVD, CISA KEV and generated negatives, synthetic fraud, DocLayNet, CORD, rendered images, and generated hard-decision families. Financial sentiment appears as a held-out evaluation source; verify split membership from the actual data before treating it as training data.

For each actual training source, collect the exact dataset/version and originating terms, provenance, transformations, usage and redistribution conditions, attribution requirements, and any restrictions applicable to model artifacts. Inspect rendered images and generated hard examples for embedded third-party content. Record findings and unresolved items before choosing the license. No source-specific legal conclusion has been made in this work.

The copied pinned JevBench task files carry the upstream license separately in `evidence/JevBench-LICENSE`; this does not license the model weights.
