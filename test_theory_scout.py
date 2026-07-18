import csv
import json
import os
import tempfile
import unittest
from pathlib import Path

from tools.theory_scout.build_gap_table import build_gap_rows, write_gap_table
from tools.theory_scout.make_paper_cards import CARD_SECTIONS, make_blank_card
from tools.theory_scout.models import PaperRecord
from tools.theory_scout.query_config import load_queries
from tools.theory_scout.secrets import load_env_file
from tools.theory_scout.cli import build_parser, dedupe, rank_records, _is_rate_limit_error


class TheoryScoutTests(unittest.TestCase):
    def test_queries_yaml_loads_without_external_dependency(self):
        config = load_queries(Path("literature/queries.yaml"))
        self.assertIn("pricing_arena", config["worlds"])
        self.assertIn("auction_house", config["worlds"])
        self.assertIn("classical_terms", config["worlds"]["labor_market"])

    def test_dedupe_prefers_doi_title_year_key(self):
        first = PaperRecord(
            source="openalex",
            source_id="1",
            title="A Paper",
            year=2020,
            authors=["A"],
            abstract=None,
            doi="10.1/x",
            url=None,
            pdf_url=None,
            citation_count=10,
            world="pricing_arena",
            query="q",
            query_group="classical_terms",
        )
        second = PaperRecord(**{**first.to_dict(), "source": "semantic_scholar", "source_id": "2"})
        self.assertEqual(len(dedupe([first, second])), 1)

    def test_rank_records_rewards_citations_and_pdf(self):
        low = PaperRecord(
            "openalex", "1", "low", 2020, [], None, None, None, None, 1,
            "auction_house", "q", "learning_terms"
        )
        high = PaperRecord(
            "openalex", "2", "high", 2020, [], None, None, None, "x.pdf", 1,
            "auction_house", "q", "learning_terms"
        )
        rows = rank_records([low, high])
        self.assertEqual(rows[0]["title"], "high")

    def test_card_contains_strict_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_blank_card(
                {
                    "title": "Strict Test Paper",
                    "year": 2024,
                    "authors": ["A"],
                    "world": "public_goods",
                },
                Path(tmp),
            )
            text = path.read_text(encoding="utf-8")
        for section in CARD_SECTIONS:
            self.assertIn(f"## {section}", text)

    def test_gap_table_crosses_world_institutions_and_minds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            query_path = root / "queries.yaml"
            raw_path = root / "papers_raw.jsonl"
            query_path.write_text(
                """
worlds:
  pricing_arena:
    institutions:
      - none
      - price_cap
    minds:
      - q_learning
      - dqn
    classical_terms:
      - "pricing"
    learning_terms:
      - "rl pricing"
""".strip(),
                encoding="utf-8",
            )
            raw_path.write_text(
                json.dumps(
                    {
                        "world": "pricing_arena",
                        "title": "Best Pricing Paper",
                        "citation_count": 99,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            rows = build_gap_rows(query_path, raw_path)
            self.assertEqual(len(rows), 4)
            out = root / "gap.csv"
            write_gap_table(rows, out)
            with out.open(newline="", encoding="utf-8") as handle:
                loaded = list(csv.DictReader(handle))
        self.assertEqual(loaded[0]["closest_paper"], "Best Pricing Paper")

    def test_search_parser_defaults_merge_cache_and_rate_limits_semantic_scholar(self):
        parser = build_parser()
        args = parser.parse_args(["search", "--sources", "semantic_scholar"])
        self.assertTrue(args.merge_existing)
        self.assertGreaterEqual(args.semantic_delay_seconds, 1.0)

    def test_search_parser_can_replace_cache_explicitly(self):
        parser = build_parser()
        args = parser.parse_args(["search", "--replace-cache"])
        self.assertFalse(args.merge_existing)

    def test_full_parser_defaults_to_ignored_env_file(self):
        parser = build_parser()
        args = parser.parse_args(["full"])
        self.assertEqual(args.env_file, "literature/secrets.env")
        self.assertGreaterEqual(args.semantic_delay_seconds, 1.0)
        self.assertFalse(args.include_arxiv)

    def test_env_file_loader_handles_export_lines_without_printing_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / "secrets.env"
            env_path.write_text(
                'export THEORY_SCOUT_TEST_SECRET="abc123"\nTHEORY_SCOUT_OTHER_SECRET=xyz\n',
                encoding="utf-8",
            )
            try:
                loaded = load_env_file(env_path, overwrite=True)
                self.assertTrue(loaded["THEORY_SCOUT_TEST_SECRET"])
                self.assertEqual(os.environ["THEORY_SCOUT_TEST_SECRET"], "abc123")
                self.assertEqual(os.environ["THEORY_SCOUT_OTHER_SECRET"], "xyz")
            finally:
                os.environ.pop("THEORY_SCOUT_TEST_SECRET", None)
                os.environ.pop("THEORY_SCOUT_OTHER_SECRET", None)

    def test_rate_limit_error_detection(self):
        self.assertTrue(_is_rate_limit_error(RuntimeError("HTTP 429: rate limit exceeded")))
        self.assertFalse(_is_rate_limit_error(RuntimeError("connection reset")))


if __name__ == "__main__":
    unittest.main()
