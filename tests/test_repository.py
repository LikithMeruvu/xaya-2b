import tempfile
import unittest
from pathlib import Path
from tools.check_integrity import ROOT, check
from tools.assess_diagnostics import assess
import json


class RepositoryEvidenceTests(unittest.TestCase):
    def test_protected_evidence_and_task_identities(self):
        self.assertEqual(check(), [])

    def test_unknown_synthetic_ids_cannot_pass_with_pinned_reference(self):
        ref = json.loads((ROOT / "benchmarks/jevbench/reference.json").read_text("utf-8"))
        report, _ = assess({"order_count": 1}, [{"id": "TEST-ONLY", "gold": 0, "tier": "easy", "probabilities": [1., 0.]}], "TEST-ONLY", ref)
        self.assertFalse(report["fingerprint_match"])
        self.assertIn("unknown task", report["errors"][0])
