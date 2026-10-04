# XAYA-2B repository instructions

- Preserve locked v1 weights. No training, tuning, calibration, or weight modification without explicit authorization for v2.
- No FINAL tag, release, canonical SDK/server claim, or sealed benchmark rank before validated runtime evidence exists.
- Treat files under `legacy/` and pinned JevBench data as reference material. Preserve bytes; proposed runtime changes belong in a separate implementation after evidence review.
- Never commit credentials, model binaries, raw personal logs, or local machine paths.
- Keep public development, adaptations, internal results, ensemble reruns, and invalid runs clearly labeled.
- Preserve the historical 148/231 tier/Brier target. A 149/231 ensemble or 61.47% speed-wrapper result does not redefine v1.
- Do not assume a model distribution license before completing the training-data audit.
- Run `python -m unittest discover -s tests -v` and `python tools/check_integrity.py` for diagnostic changes. The workflow template under `docs/ci/` runs offline checks with no model inference; it is not active CI yet.
