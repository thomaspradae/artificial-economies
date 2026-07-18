import csv
import json
import os
import tempfile
import unittest
from pathlib import Path

from tools.theory_scout.build_gap_table import build_gap_rows, write_gap_table
from tools.theory_scout.audit_obligations import (
    audit_obligations,
    write_audit_csv,
    write_audit_markdown,
)
from tools.theory_scout.fill_paper_cards import fill_cards, parse_markdown_sections
from tools.theory_scout.hydrate_texts import hydrate_texts
from tools.theory_scout.make_paper_cards import CARD_SECTIONS, make_blank_card
from tools.theory_scout.models import PaperRecord
from tools.theory_scout.ollama_client import OllamaResult, extract_json_object
from tools.theory_scout.query_config import load_queries
from tools.theory_scout.secrets import load_env_file
from tools.theory_scout.cli import build_parser, dedupe, rank_records, _is_rate_limit_error


class FakeOllamaClient:
    def chat(self, **kwargs):
        content = json.dumps(
            {
                "paper": "Strict Test Paper",
                "world": "public_goods",
                "institution": "contribution matching",
                "agent_type": "Q-learning agents",
                "theoretical_benchmark": "free-rider and social optimum brackets",
                "learning_setup": "multi-agent public goods game",
                "metrics": ["contribution", "welfare", "sustainability"],
                "main_result": "matching changes contribution incentives",
                "what_they_prove": "Not stated in supplied text.",
                "what_they_only_simulate": "learning behavior",
                "what_they_do_not_test": "cross-world capability ladder",
                "what_we_need_to_reproduce": "free-rider bracket and contribution metrics",
                "how_our_project_differs": "shared world/mind/institution interface",
                "source_evidence": "text:/tmp/not-real.txt",
                "confidence": "medium",
            }
        )
        return OllamaResult(
            model=kwargs.get("model", "fake"),
            content=content,
            raw={"eval_count": 20, "eval_duration": 1_000_000_000},
        )


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

    def test_extract_json_object_accepts_fenced_or_prose_wrapped_json(self):
        parsed = extract_json_object('Here:\n```json\n{"a": 1}\n```')
        self.assertEqual(parsed["a"], 1)
        parsed = extract_json_object('prefix {"b": 2} suffix')
        self.assertEqual(parsed["b"], 2)

    def test_fill_cards_rewrites_todo_sections_with_validated_model_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "papers_raw.jsonl"
            raw.write_text(
                json.dumps(
                    {
                        "source": "test",
                        "source_id": "paper-1",
                        "title": "Strict Test Paper",
                        "year": 2024,
                        "authors": ["A"],
                        "abstract": "A public goods paper about contribution matching.",
                        "doi": "10.1/test",
                        "url": "https://example.test",
                        "pdf_url": None,
                        "citation_count": 1,
                        "world": "public_goods",
                        "query": "public goods",
                        "query_group": "learning_terms",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            results = fill_cards(
                raw_path=raw,
                cards_dir=root / "paper_cards",
                text_dir=root / "text",
                client=FakeOllamaClient(),  # type: ignore[arg-type]
                limit=1,
            )
            self.assertTrue(results[0].changed)
            text = results[0].card_path.read_text(encoding="utf-8")
            sections = parse_markdown_sections(text)
        self.assertEqual(sections["Theoretical benchmark"], "free-rider and social optimum brackets")
        self.assertIn("A public goods paper about contribution matching.", sections["Extraction evidence"])

    def test_hydrate_texts_downloads_pdf_and_writes_canonical_text_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = root / "papers_ranked.csv"
            records.write_text(
                "\n".join(
                    [
                        "world,query_group,query,source,source_id,title,year,authors,doi,url,pdf_url,citation_count,has_pdf,relevance_score,rank_score,abstract",
                        "auction_house,learning_terms,q,test,p1,Strict PDF Paper,2024,A,10.1/test,https://example.test,https://example.test/p.pdf,1,True,1,1,abstract",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            def fake_download(pdf_url, title, year, out_dir):
                out_dir.mkdir(parents=True)
                path = out_dir / "downloaded.pdf"
                path.write_bytes(b"%PDF fake")
                return path

            def fake_extract(pdf_path, out_dir, out_path=None):
                target = out_path or out_dir / "downloaded.txt"
                target.parent.mkdir(parents=True)
                target.write_text("full paper text " * 200, encoding="utf-8")
                return target

            rows = hydrate_texts(
                records_path=records,
                pdf_dir=root / "pdfs",
                text_dir=root / "text",
                report_path=root / "pdf_text_report.csv",
                limit=1,
                download_func=fake_download,
                extract_func=fake_extract,
            )
            self.assertEqual(rows[0].pdf_status, "downloaded")
            self.assertEqual(rows[0].text_status, "extracted")
            self.assertTrue((root / "text/2024_strict_pdf_paper.txt").exists())
            self.assertTrue((root / "pdf_text_report.csv").exists())

    def test_audit_obligations_reports_required_missing_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "literature/paper_cards").mkdir(parents=True)
            (root / "worlds/pricing_arena").mkdir(parents=True)
            (root / "worlds/pricing_arena/benchmarks.py").write_text("nash joint", encoding="utf-8")
            (root / "outputs/full_v0_multiseed").mkdir(parents=True)
            (root / "outputs/full_v0_multiseed/summary_aggregate.csv").write_text(
                "nash_price\n1\n",
                encoding="utf-8",
            )
            rows = audit_obligations(
                repo_root=root,
                literature_dir=root / "literature",
                include_card_obligations=False,
            )
            out_csv = root / "literature/obligation_audit.csv"
            out_md = root / "literature/obligation_audit.md"
            write_audit_csv(rows, out_csv)
            write_audit_markdown(rows, out_md)
            pricing_benchmark = [
                row
                for row in rows
                if row.world == "pricing_arena" and row.category == "benchmark"
            ][0]
            self.assertEqual(pricing_benchmark.status, "partial")
            self.assertIn("monopoly_price", pricing_benchmark.missing)
            self.assertTrue(out_csv.exists())
            self.assertTrue(out_md.exists())

    def test_new_cli_subcommands_are_registered(self):
        parser = build_parser()
        fill_args = parser.parse_args(["fill-cards", "--limit", "2", "--model", "llama3.2:3b"])
        self.assertEqual(fill_args.limit, 2)
        self.assertEqual(fill_args.model, "llama3.2:3b")
        hydrate_args = parser.parse_args(["hydrate-text", "--limit", "2", "--resolve-pdfs"])
        self.assertEqual(hydrate_args.limit, 2)
        self.assertTrue(hydrate_args.resolve_pdfs)
        full_args = parser.parse_args(["full", "--download", "--fill-cards", "--fill-limit", "4"])
        self.assertTrue(full_args.download)
        self.assertTrue(full_args.fill_cards)
        self.assertEqual(full_args.fill_limit, 4)
        audit_args = parser.parse_args(["audit-obligations", "--no-card-obligations"])
        self.assertTrue(audit_args.no_card_obligations)


if __name__ == "__main__":
    unittest.main()
