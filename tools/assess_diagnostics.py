"""Recompute profile fingerprints and per-task differences without model inference."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import zipfile
from pathlib import Path

TARGET = {"n": 231, "correct": 148, "brier": 0.5611851962,
          "tiers": {"original": (72, 50), "easy": (48, 48), "hard": (111, 50)}}


def records_in(obj):
    if isinstance(obj, list):
        return obj
    if not isinstance(obj, dict):
        return None
    for key in ("predictions", "results", "records", "rows"):
        if isinstance(obj.get(key), list):
            return obj[key]
    if "probabilities" in obj or "probs" in obj or "error" in obj:
        return [obj]
    return None


def parse_records(text):
    """Accept JSON, JSONL, and the observed literal-backslash-n JSONL bug."""
    try:
        obj = json.loads(text)
        rows = records_in(obj)
        return obj, rows, False
    except json.JSONDecodeError:
        decoder, pos, rows, repaired = json.JSONDecoder(), 0, [], False
        while pos < len(text):
            if text[pos].isspace():
                pos += 1
                continue
            if text.startswith("\\n", pos):
                repaired = True
                pos += 2
                continue
            obj, pos = decoder.raw_decode(text, pos)
            chunk = records_in(obj)
            if chunk is None:
                raise ValueError("JSON stream contains a non-prediction record")
            rows.extend(chunk)
        return {}, rows, repaired


def order_count(obj, rows):
    cfg = obj if isinstance(obj, dict) else {}
    declared = []
    for scope in [cfg, cfg.get("config", {}), cfg.get("profile", {})]:
        if not isinstance(scope, dict):
            continue
        for key in ("order_count", "orders", "rotations", "n_orders"):
            if key in scope:
                value = scope[key]
                if isinstance(value, list):
                    declared.append(len(value))
                elif isinstance(value, int) and not isinstance(value, bool):
                    declared.append(value)
    counts = []
    for r in rows:
        if not isinstance(r, dict):
            return None
        if "order_count" in r:
            counts.append(r["order_count"])
        elif isinstance(r.get("latencies_s"), list):
            counts.append(len(r["latencies_s"]))
    evidence = declared + counts
    if not evidence or len(set(evidence)) != 1:
        return None
    if not declared and len(counts) != len(rows):
        return None
    return evidence[0]


def assess(obj, rows, name, reference=None):
    errors, checked, seen = [], [], set()
    aliases = {}
    if reference:
        for task in reference["tasks"]:
            aliases[task["id"]] = task
            aliases[task["broad_id"]] = task
    for r in rows:
        if not isinstance(r, dict):
            errors.append("prediction record is not an object")
            continue
        rid = str(r.get("source_id", r.get("id", "")))
        task = aliases.get(rid)
        rid = task["id"] if task else rid
        if not rid or rid in seen:
            errors.append(f"missing or duplicate task id: {rid!r}")
            continue
        seen.add(rid)
        if reference and task is None:
            errors.append(f"unknown task: {rid}")
            continue
        if r.get("error"):
            errors.append(f"{rid}: inference error {r['error']}")
            continue
        meta = r.get("meta") or {}
        tier = r.get("tier", meta.get("tier") if isinstance(meta, dict) else None)
        if task:
            if tier is not None and tier != task["tier"]:
                errors.append(f"{rid}: wrong tier")
                continue
            tier = task["tier"]
        p = r.get("probabilities", r.get("probs"))
        gold = r.get("gold", r.get("answer"))
        if (not isinstance(p, list) or not 1 <= len(p) <= 256
                or any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or not 0 <= v <= 1 for v in p)
                or abs(sum(p) - 1) > 1e-5
                or isinstance(gold, bool) or not isinstance(gold, int) or not 0 <= gold < len(p)):
            errors.append(f"{rid}: invalid probability vector or gold index")
            continue
        if tier not in TARGET["tiers"]:
            errors.append(f"{rid}: missing or unknown tier")
            continue
        pred = max(range(len(p)), key=p.__getitem__)
        if "pred" in r and r["pred"] != pred:
            errors.append(f"{rid}: stored pred disagrees with probabilities")
            continue
        brier = sum((v - int(i == gold)) ** 2 for i, v in enumerate(p))
        checked.append({"id": rid, "tier": tier, "gold": gold, "pred": pred,
                        "probabilities": p, "correct": int(pred == gold), "brier": brier})
    tiers = {k: {"n": sum(r["tier"] == k for r in checked),
                 "correct": sum(r["correct"] for r in checked if r["tier"] == k)} for k in TARGET["tiers"]}
    metrics = {"n": len(checked), "correct": sum(r["correct"] for r in checked),
               "brier": statistics.mean(r["brier"] for r in checked) if checked else None, "tiers": tiers}
    match = (not errors and metrics["n"] == TARGET["n"] and metrics["correct"] == TARGET["correct"]
             and all(tiers[k] == {"n": n, "correct": c} for k, (n, c) in TARGET["tiers"].items())
             and abs(metrics["brier"] - TARGET["brier"]) <= 0.0015)
    orders = order_count(obj, rows)
    report = {"profile": name, "metrics": metrics, "order_count": orders,
              "fingerprint_match": match, "one_order_fingerprint_match": match and orders == 1,
              "errors": errors, "historical_per_task_parity_proven": False,
              "gold_semantics_verified": False,
              "note": "Aggregate fingerprint assessment using supplied gold indices. Confirm option/gold semantics against the original evaluator before release. No FINAL export or claim of identical historical predictions."}
    return report, checked


def compare(left, right):
    a, b = {r["id"]: r for r in left}, {r["id"]: r for r in right}
    output = []
    for rid in sorted(a.keys() | b.keys()):
        x, y = a.get(rid), b.get(rid)
        if x is None or y is None:
            output.append({"id": rid, "change": "missing_task", "left_present": x is not None, "right_present": y is not None})
            continue
        if len(x["probabilities"]) != len(y["probabilities"]):
            output.append({"id": rid, "change": "option_count", "left_n": len(x["probabilities"]), "right_n": len(y["probabilities"])})
            continue
        delta = max(abs(v - w) for v, w in zip(x["probabilities"], y["probabilities"]))
        output.append({"id": rid, "tier": x["tier"], "left_pred": x["pred"], "right_pred": y["pred"],
                       "argmax_changed": x["pred"] != y["pred"], "gold_mapping_changed": x["gold"] != y["gold"],
                       "left_gold": x["gold"], "right_gold": y["gold"], "max_probability_delta": delta,
                       "change": "gold_mapping" if x["gold"] != y["gold"] else "prediction" if x["pred"] != y["pred"] else "probabilities"})
    return output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path, help="diagnostics ZIP, JSON, JSONL, or directory")
    ap.add_argument("--reference", type=Path, default=Path(__file__).resolve().parents[1] / "benchmarks/jevbench/reference.json")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "diagnostic-assessment")
    args = ap.parse_args()
    reference = json.loads(args.reference.read_text("utf-8"))
    if args.input.suffix.lower() == ".zip":
        with zipfile.ZipFile(args.input) as z:
            sources = [(i.filename, z.read(i).decode("utf-8")) for i in z.infolist()
                       if not i.is_dir() and Path(i.filename).suffix.lower() in (".json", ".jsonl")]
    else:
        files = sorted(args.input.rglob("*.json")) + sorted(args.input.rglob("*.jsonl")) if args.input.is_dir() else [args.input]
        sources = [(str(p), p.read_text("utf-8")) for p in files]
    args.out.mkdir(parents=True, exist_ok=True)
    profiles, skipped, parse_errors = {}, [], []
    for i, (name, text) in enumerate(sources):
        try:
            obj, rows, separators = parse_records(text)
            if rows is None:
                skipped.append(name)
                continue
            label = f"{i:03d}-{Path(name).stem}"
            report, checked = assess(obj, rows, name, reference)
            report["literal_newline_separators_observed"] = separators
            (args.out / (label + ".assessment.json")).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            profiles[label] = (report, checked)
        except (ValueError, TypeError, KeyError) as e:
            parse_errors.append({"file": name, "error": str(e)})
    keys = list(profiles)
    for i, left in enumerate(keys):
        for right in keys[i + 1:]:
            diff = compare(profiles[left][1], profiles[right][1])
            (args.out / f"{left}__vs__{right}.json").write_text(json.dumps(diff, indent=2) + "\n", encoding="utf-8")
    summary = {"profiles": [v[0] for v in profiles.values()], "skipped_summary_files": skipped,
               "parse_errors": parse_errors, "final_exported": False}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"profiles_assessed": len(profiles), "one_order_matches": sum(v[0]["one_order_fingerprint_match"] for v in profiles.values()),
                      "parse_errors": len(parse_errors), "report": str(args.out / "summary.json")}))


if __name__ == "__main__":
    main()
