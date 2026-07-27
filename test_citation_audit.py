import tempfile
import unittest
from pathlib import Path

from tools.theory_scout.citation_audit import (
    build_citation_audit,
    extract_citations,
    parse_bib_keys,
    suggest_citations,
)


class CitationAuditTests(unittest.TestCase):
    def test_bib_key_extraction(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bib = Path(tmpdir) / "references.bib"
            bib.write_text(
                "@article{vickrey1961,\n  title={Auctions}\n}\n"
                "@book{sutton2018,\n  title={Reinforcement Learning}\n}\n",
                encoding="utf-8",
            )

            self.assertEqual(parse_bib_keys(bib), {"vickrey1961", "sutton2018"})

    def test_citation_extraction_supports_multiple_keys(self):
        sentence = "DQN and PPO are standard baselines \\citep{mnih2015,schulman2017}."

        self.assertEqual(extract_citations(sentence), ["mnih2015", "schulman2017"])

    def test_truthful_second_price_suggests_vickrey_when_available(self):
        sentence = "Second-price auctions have truthful bidding as the benchmark."

        self.assertEqual(suggest_citations(sentence, {"vickrey1961"}), ["vickrey1961"])

    def test_uncited_result_claim_gets_artifact_support_when_output_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "paper").mkdir()
            (root / "outputs/phase3_full").mkdir(parents=True)
            (root / "outputs/phase3_full/mind_comparison.csv").write_text(
                "world,mind,metric\npricing_arena,dqn,1.0\n",
                encoding="utf-8",
            )
            (root / "paper/main.tex").write_text(
                "\\section{Pricing Arena}\n"
                "The full Pricing Arena table shows DQN profit rises under the cap.\n",
                encoding="utf-8",
            )
            (root / "paper/references.bib").write_text("", encoding="utf-8")

            rows = build_citation_audit(
                repo_root=root,
                tex_path=root / "paper/main.tex",
                bib_path=root / "paper/references.bib",
            )

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].status, "supported_artifact")
            self.assertIn("outputs/phase3_full/mind_comparison.csv", rows[0].artifact_support)


if __name__ == "__main__":
    unittest.main()
