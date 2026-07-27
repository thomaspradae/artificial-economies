import unittest

from run_claim_audit_suite import _classify, _summary


class ClaimAuditSuiteTests(unittest.TestCase):
    def test_classifies_robust_positive_seed_effect(self):
        stats = _summary([1.0, 1.2, 0.9, 1.1, 1.3, 0.8])
        self.assertEqual(_classify(stats), "robust_positive")

    def test_classifies_robust_negative_seed_effect(self):
        stats = _summary([-1.0, -1.2, -0.9, -1.1, -1.3, -0.8])
        self.assertEqual(_classify(stats), "robust_negative")

    def test_classifies_mixed_seed_sign(self):
        stats = _summary([-1.0, -0.8, -0.4, 0.2, 0.5, 1.1])
        self.assertEqual(_classify(stats), "mixed_seed_sign")

    def test_summary_reports_positive_share(self):
        stats = _summary([-1.0, 0.0, 2.0, 3.0])
        self.assertEqual(stats["n"], 4)
        self.assertEqual(stats["positive_share"], 0.5)
        self.assertEqual(stats["negative_share"], 0.25)


if __name__ == "__main__":
    unittest.main()
