# Diagnostic workflow template

`checks.yml` runs the evidence integrity checker and offline tests on Python 3.11 and 3.12. Its GitHub action references are pinned to verified release commit hashes.

The credential used for initial publication has repository access but lacks the `workflow` scope. GitHub rejected the initial upload of `.github/workflows/checks.yml`. This template is preserved here and is not active CI.

Activate it by moving `docs/ci/checks.yml` to `.github/workflows/checks.yml` using GitHub's editor or a credential authorized to write workflows. The local test commands in the main README work now.
