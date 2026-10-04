# Contributing

The immediate goal is to reproduce the historical frozen v1 runtime. Include exact dependency versions, pinned dataset commit, model component hashes, normalized inputs, candidate labels/order, and all raw probabilities with any inference change.

Do not select or tune weights using public benchmark results. Keep research into a future v2 separate from the frozen v1 release.

Run `python -m unittest discover -s tests -v` and `python tools/check_integrity.py`. Add tests for meaningful evidence-integrity failures when changing the diagnostic assessor. Keep weights and raw diagnostic bundles outside Git.

Historical reference source and pinned data are byte-preserved. Runtime changes should be reviewed with prediction-level differences and documented protocol changes.
