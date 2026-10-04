"""Verify byte-preserved reference files and pinned benchmark identities."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check(root=ROOT):
    manifest = json.loads((root / "repo_integrity.json").read_text("utf-8"))
    errors = []
    protected = manifest["protected_reference_files_sha256"]
    for name, expected in protected.items():
        p = root / name
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != expected:
            errors.append(f"Reference bytes changed or missing: {name}")
    ref = json.loads((root / "benchmarks/jevbench/reference.json").read_text("utf-8"))
    if ref["commit"] != "9ec6f15a8773aaffbbb41e7d7e65a20599ab0cc3":
        errors.append("Wrong benchmark commit")
    ids = []
    for tier, count in [("original", 72), ("easy", 48), ("hard", 111)]:
        p = root / f"benchmarks/jevbench/{tier}.jsonl"
        data = p.read_bytes()
        tasks = [json.loads(line) for line in data.splitlines() if line.strip()]
        if len(tasks) != count or hashlib.sha256(data).hexdigest() != ref["files"][tier]:
            errors.append(f"Wrong {tier} data or count")
        expected_tasks = [t["task"] for t in ref["tasks"] if t["tier"] == tier]
        if tasks != expected_tasks:
            errors.append(f"Reference mappings do not match {tier} task bytes")
        ids.extend(t["id"] for t in tasks)
    if len(ids) != 231 or len(set(ids)) != 231:
        errors.append("Missing or duplicate benchmark IDs")
    dataset = (root / "release/training_dataset_manifest.json").read_bytes()
    lock = json.loads((root / "release/model_lock.json").read_text("utf-8"))
    if hashlib.sha256(dataset).hexdigest() != lock["dataset_manifest_sha256"]:
        errors.append("Training manifest does not match model lock")
    if lock["sha256"] != "d1fc1e1d3ddd1795dcf207263424b320bb4692ccec7b092c9977e6e27c6a591a":
        errors.append("Wrong frozen model lock")
    return errors


if __name__ == "__main__":
    failures = check()
    print(json.dumps({"ok": not failures, "errors": failures, "benchmark_tasks": 231}))
    raise SystemExit(1 if failures else 0)
