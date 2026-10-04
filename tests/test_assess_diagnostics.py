import copy
import math
import unittest

from tools.assess_diagnostics import TARGET, assess, compare, parse_records


class DiagnosticIntegrityTests(unittest.TestCase):
    def fixture(self):
        # Synthetic, explicitly test-only. Same fingerprint, different underlying tasks.
        miss_probability = math.sqrt(TARGET["brier"] * 231 / (2 * 83))
        rows = []
        for tier, (n, correct) in TARGET["tiers"].items():
            for i in range(n):
                p = [1., 0.] if i < correct else [1 - miss_probability, miss_probability]
                rows.append({"id": f"TEST-{tier}-{i}", "tier": tier, "gold": 0, "probabilities": p})
        return rows

    def test_recomputes_metrics_and_accepts_exact_one_order_fingerprint(self):
        report, _ = assess({"order_count": 1}, self.fixture(), "TEST-ONLY")
        self.assertTrue(report["one_order_fingerprint_match"])
        self.assertAlmostEqual(report["metrics"]["brier"], TARGET["brier"])
        self.assertFalse(report["historical_per_task_parity_proven"])

    def test_ensemble_or_unknown_order_cannot_be_one_order_match(self):
        for cfg in ({"order_count": 2}, {}):
            report, _ = assess(cfg, self.fixture(), "TEST-ONLY")
            self.assertTrue(report["fingerprint_match"])
            self.assertFalse(report["one_order_fingerprint_match"])

    def test_rejects_duplicate_missing_invalid_probability_or_stored_prediction(self):
        for change in ("duplicate", "missing", "nan", "sum", "negative", "pred"):
            rows = self.fixture()
            if change == "duplicate": rows[1]["id"] = rows[0]["id"]
            elif change == "missing": rows.pop()
            elif change == "nan": rows[0]["probabilities"] = [float("nan"), 0.]
            elif change == "sum": rows[0]["probabilities"] = [0.3, 0.3]
            elif change == "negative": rows[0]["probabilities"] = [1.1, -0.1]
            else: rows[0]["pred"] = 1
            self.assertFalse(assess({"order_count": 1}, rows, "TEST-ONLY")[0]["fingerprint_match"], change)

    def test_different_tier_fingerprint_cannot_pass(self):
        rows = self.fixture()
        rows[0]["tier"] = "hard"
        self.assertFalse(assess({"order_count": 1}, rows, "TEST-ONLY")[0]["fingerprint_match"])

    def test_reports_gold_mapping_separately_from_prediction_change(self):
        _, left = assess({}, self.fixture(), "TEST-ONLY")
        right = copy.deepcopy(left)
        right[0]["gold"] = 1
        changed = next(r for r in compare(left, right) if r["id"] == left[0]["id"])
        self.assertTrue(changed["gold_mapping_changed"])

    def test_conflicting_order_evidence_and_malformed_records_cannot_pass(self):
        rows = self.fixture()
        for r in rows:
            r["latencies_s"] = [0.1, 0.1]
        report, _ = assess({"order_count": 1}, rows, "TEST-ONLY")
        self.assertFalse(report["one_order_fingerprint_match"])
        rows[0] = None
        self.assertFalse(assess({}, rows, "TEST-ONLY")[0]["fingerprint_match"])

    def test_recovers_literal_jsonl_separators_without_changing_string_contents(self):
        text = '{"id":"a","probabilities":[1,0],"question":"literal\\\\n"}\\n{"id":"b","probabilities":[0,1]}\\n'
        _, rows, repaired = parse_records(text)
        self.assertTrue(repaired)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["question"], "literal\\n")

    def test_summary_is_not_prediction_evidence(self):
        self.assertIsNone(parse_records('{"accuracy":0.64,"n":231}')[1])


if __name__ == "__main__":
    unittest.main()
