# Release roadmap

- [x] Preserve locked release identity and component hashes.
- [x] Verify training dataset manifest hash.
- [x] Convert decision head to safetensors and verify exact tensor equality locally.
- [x] Pin the 231 JevBench public tasks and preserve upstream license.
- [x] Add diagnostic integrity checks and per-task comparison tooling.
- [x] Prepare model documentation and historical comparison charts.
- [ ] Obtain recovery diagnostics and original continuation evaluator/source.
- [ ] Reproduce the single-order historical fingerprint and inspect prediction differences.
- [ ] Freeze the shared canonical runtime and dependency configuration.
- [ ] Validate safetensors end-to-end inference parity and measure T4 latency.
- [ ] Build tested Python SDK, server, benchmark adapter, and probability-bar demo on that shared runtime.
- [ ] Complete training-data license audit and select a distribution license.
- [ ] Generate a separately hashed XAYA-2B-v1.0.0 FINAL package.
- [ ] Publish Hugging Face model, GitHub release, and Space together.
- [ ] Request maintainer-run sealed evaluation; report rank only after results arrive.

This source repository can be prepared now while FINAL remains pending. Further training is reserved for an explicitly authorized v2.
