import csv
import json
import os
import tempfile
import unittest
from pathlib import Path

from tools.theory_scout.build_gap_table import build_gap_rows, write_gap_table
from tools.theory_scout.audit_obligations import (
    audit_obligations,
    card_obligation_rows,
    write_audit_csv,
    write_audit_markdown,
)
from tools.theory_scout.fill_paper_cards import (
    fill_cards,
    parse_markdown_sections,
    select_records_for_fill,
    source_context_for_record,
)
from tools.theory_scout.foundation_papers import (
    FOUNDATION_PAPERS,
    FoundationPaper,
    build_foundation_matches,
    write_foundation_csv,
    write_foundation_markdown,
    write_foundation_pdf_queue,
)
from tools.theory_scout.hydrate_texts import hydrate_texts
from tools.theory_scout.make_paper_cards import CARD_SECTIONS, make_blank_card
from tools.theory_scout.models import PaperRecord
from tools.theory_scout.obligation_templates import ROLE_TEMPLATES, apply_obligation_template
from tools.theory_scout.ollama_client import OllamaResult, extract_json_object
from tools.theory_scout.query_config import load_queries
from tools.theory_scout.review_outputs import (
    build_manual_pdf_queue_rows,
    build_scholar_comparison_rows,
    build_theory_coverage_rows,
    google_scholar_url,
    write_coverage_markdown,
)
from tools.theory_scout.scholar_compare import compare_worksheet_rows, split_titles, title_similarity
from tools.theory_scout.secrets import load_env_file
from tools.theory_scout.theory_code_audit import (
    build_theory_code_audit,
    write_theory_code_audit_csv,
    write_theory_code_audit_markdown,
)
from tools.theory_scout.world_obligation_bridge import build_world_obligation_bridge
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
                "implementation_obligations": "track contributions; track extraction; report sustainability",
                "metrics_to_compare_in_our_repo": "contribution_total; welfare; sustainability",
                "failure_modes_to_audit": "reward-only change without state change",
                "project_code_comparison": "Compare against worlds/public_goods/training.py and outputs/public_goods_full.",
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
        self.assertIn("sustainability", sections["Metrics to compare in our repo"])
        self.assertIn("A public goods paper about contribution matching.", sections["Extraction evidence"])

    def test_source_context_finds_manual_text_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            text_dir = root / "text"
            manual_dir = text_dir / "manual"
            manual_dir.mkdir(parents=True)
            manual_dir.joinpath("cross_world_methods__2017__proximal_policy_optimization_algorithms.txt").write_text(
                "manual PPO source text " * 100,
                encoding="utf-8",
            )
            text, basis = source_context_for_record(
                {
                    "world": "cross_world_methods",
                    "year": 2017,
                    "title": "Proximal Policy Optimization Algorithms",
                    "abstract": "abstract fallback",
                },
                text_dir=text_dir,
            )

        self.assertIn("manual PPO source text", text)
        self.assertIn("text/manual", basis)

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

    def test_per_world_limit_balances_record_selection(self):
        records = [
            {"world": "auction_house", "title": "Auction A", "year": 2024, "doi": "10/a"},
            {"world": "auction_house", "title": "Auction B", "year": 2024, "doi": "10/b"},
            {"world": "public_goods", "title": "Goods A", "year": 2024, "doi": "10/c"},
            {"world": "public_goods", "title": "Goods B", "year": 2024, "doi": "10/d"},
        ]
        selected = select_records_for_fill(records, per_world_limit=1)
        self.assertEqual([row["title"] for row in selected], ["Auction A", "Goods A"])

    def test_review_outputs_make_manual_pdf_and_scholar_worklists(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queries = root / "queries.yaml"
            queries.write_text(
                """
worlds:
  public_goods:
    institutions:
      - none
    minds:
      - q_learning
    classical_terms:
      - "public goods game"
    learning_terms:
      - "multi agent reinforcement learning public goods"
""".strip(),
                encoding="utf-8",
            )
            records = root / "papers_ranked.csv"
            records.write_text(
                "\n".join(
                    [
                        "world,query_group,query,source,source_id,title,year,authors,doi,url,pdf_url,citation_count,has_pdf,relevance_score,rank_score,abstract",
                        "public_goods,classical_terms,public goods game,test,p1,Public Goods Paper,2024,A,10.1/test,https://example.test,https://example.test/p.pdf,5,True,1,1,abstract",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            card_path = make_blank_card(
                {
                    "title": "Public Goods Paper",
                    "year": 2024,
                    "authors": ["A"],
                    "source": "test",
                    "source_id": "p1",
                    "world": "public_goods",
                    "query": "public goods game",
                    "pdf_url": "https://example.test/p.pdf",
                },
                root / "paper_cards",
            )
            text = card_path.read_text(encoding="utf-8").replace("TODO", "filled")
            card_path.write_text(text, encoding="utf-8")
            (root / "text").mkdir()
            (root / "text/2024_public_goods_paper.txt").write_text("paper text " * 200, encoding="utf-8")

            manual_rows = build_manual_pdf_queue_rows(
                records_path=records,
                text_dir=root / "text",
                per_world_limit=1,
            )
            scholar_rows = build_scholar_comparison_rows(
                queries_path=queries,
                records_path=records,
            )
            coverage_rows = build_theory_coverage_rows(
                queries_path=queries,
                records_path=records,
                cards_dir=root / "paper_cards",
                text_dir=root / "text",
            )
            write_coverage_markdown(coverage_rows, root / "theory_coverage.md")
            coverage_markdown_exists = (root / "theory_coverage.md").exists()

        self.assertIn("scholar.google.com", google_scholar_url("public goods game"))
        self.assertEqual(manual_rows[0]["has_extracted_text"], True)
        self.assertIn("Public Goods Paper", scholar_rows[0]["api_top_titles"])
        self.assertEqual(coverage_rows[0].filled_cards, 1)
        self.assertEqual(coverage_rows[0].extracted_text_records, 1)
        self.assertTrue(coverage_markdown_exists)

    def test_scholar_compare_matches_manual_titles_to_api_titles(self):
        rows = [
            {
                "world": "auction_house",
                "query_group": "learning_terms",
                "query": "RegretNet auction design",
                "google_scholar_url": "https://scholar.google.com/scholar?q=RegretNet",
                "api_top_titles": "Optimal Auctions through Deep Learning | Unrelated Edge AI Survey",
                "scholar_top_titles_manual": "Optimal Auctions Through Deep Learning | RegretNet: Learning to Design Auctions",
            }
        ]
        compared = compare_worksheet_rows(rows, threshold=0.7)
        self.assertEqual(compared[0]["comparison_status"], "partial_overlap")
        self.assertIn("RegretNet: Learning to Design Auctions", compared[0]["scholar_missing_from_api"])
        self.assertIn("Unrelated Edge AI Survey", compared[0]["api_false_positives"])
        self.assertGreater(
            title_similarity("Optimal Auctions Through Deep Learning", "Optimal Auctions through Deep Learning"),
            0.9,
        )
        self.assertEqual(split_titles("A | B\nC;;D"), ["A", "B", "C", "D"])

    def test_foundation_papers_match_cache_and_write_pdf_queue(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = root / "papers_ranked.csv"
            records.write_text(
                "\n".join(
                    [
                        "world,query_group,query,source,source_id,title,year,authors,doi,url,pdf_url,citation_count,has_pdf,relevance_score,rank_score,abstract",
                        "pricing_arena,classical_terms,q,test,p1,Artificial Intelligence Algorithmic Pricing and Collusion,2020,A; B,10.1257/aer.20190623,https://example.test,https://example.test/p.pdf,10,True,1,1,abstract",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            text_dir = root / "text"
            text_dir.mkdir()
            text_dir.joinpath("2020_artificial_intelligence_algorithmic_pricing_and_collusion.txt").write_text(
                "paper text " * 100,
                encoding="utf-8",
            )
            rows = build_foundation_matches(
                records_path=records,
                text_dir=text_dir,
                foundations=[
                    FoundationPaper(
                        world="pricing_arena",
                        priority="must_read",
                        role="algorithmic_collusion",
                        title="Artificial Intelligence, Algorithmic Pricing, and Collusion",
                        authors="A; B",
                        year="2020",
                        institution="none",
                        mind="q_learning",
                        why_it_matters="anchor",
                        theory_obligation="collusion metric",
                        code_result_check="comparison table",
                        doi="10.1257/aer.20190623",
                    ),
                    FoundationPaper(
                        world="auction_house",
                        priority="must_read",
                        role="truthful_second_price",
                        title="Counterspeculation, Auctions, and Competitive Sealed Tenders",
                        authors="William Vickrey",
                        year="1961",
                        institution="second_price",
                        mind="all",
                        why_it_matters="anchor",
                        theory_obligation="truthful benchmark",
                        code_result_check="benchmarks",
                    ),
                ],
            )
            write_foundation_csv(rows, root / "foundation_papers.csv")
            write_foundation_markdown(rows, root / "foundation_papers.md")
            write_foundation_pdf_queue(rows, root / "foundation_pdf_queue.csv")
            with (root / "foundation_pdf_queue.csv").open(encoding="utf-8") as handle:
                queue = list(csv.DictReader(handle))

        self.assertTrue(rows[0].has_extracted_text)
        self.assertEqual(rows[0].manual_action, "ready_for_llm")
        self.assertEqual(rows[1].manual_action, "manual_search_needed")
        self.assertEqual(len(queue), 1)
        self.assertEqual(queue[0]["title"], "Counterspeculation, Auctions, and Competitive Sealed Tenders")

    def test_foundation_short_titles_do_not_match_mentions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = root / "papers_ranked.csv"
            records.write_text(
                "\n".join(
                    [
                        "world,query_group,query,source,source_id,title,year,authors,doi,url,pdf_url,citation_count,has_pdf,relevance_score,rank_score,abstract",
                        "pricing_arena,classical_terms,q,test,p1,Artificial Collusion: Examining Supracompetitive Pricing by Q-Learning Algorithms,2026,A,,https://example.test,https://example.test/p.pdf,10,True,1,1,abstract",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            rows = build_foundation_matches(
                records_path=records,
                text_dir=root / "text",
                foundations=[
                    FoundationPaper(
                        world="cross_world_methods",
                        priority="must_read",
                        role="tabular_rl",
                        title="Q-learning",
                        authors="Christopher J. C. H. Watkins; Peter Dayan",
                        year="1992",
                        institution="all",
                        mind="q_learning",
                        why_it_matters="anchor",
                        theory_obligation="tabular update",
                        code_result_check="q table",
                    )
                ],
            )

        self.assertEqual(rows[0].cache_status, "not_found_in_cache")
        self.assertEqual(rows[0].manual_action, "manual_search_needed")

    def test_foundation_obligation_templates_cover_every_role(self):
        foundation_roles = {paper.role for paper in FOUNDATION_PAPERS}
        self.assertEqual(foundation_roles - set(ROLE_TEMPLATES), set())

    def test_foundation_obligation_templates_apply_to_all_foundation_records(self):
        required = [
            "what_we_need_to_reproduce",
            "implementation_obligations",
            "metrics_to_compare_in_our_repo",
            "failure_modes_to_audit",
            "project_code_comparison",
        ]
        for paper in FOUNDATION_PAPERS:
            with self.subTest(role=paper.role, title=paper.title):
                fields = {}
                applied = apply_obligation_template(fields, paper.__dict__)
                self.assertTrue(applied)
                for key in required:
                    self.assertIn(key, fields)
                    self.assertGreater(len(fields[key]), 20)

    def test_policy_gradient_template_turns_ppo_into_repo_obligations(self):
        fields = {
            "main_result": "PPO improves benchmark returns.",
            "what_they_prove": "Not stated in supplied text.",
        }
        applied = apply_obligation_template(
            fields,
            {
                "role": "policy_gradient",
                "title": "Proximal Policy Optimization Algorithms",
                "world": "cross_world_methods",
                "institution": "all",
                "mind": "ppo",
                "theory_obligation": "Report PPO as an on-policy actor-critic learner.",
                "code_result_check": "minds/deep_rl/torch_ppo_mind.py",
            },
        )
        self.assertTrue(applied)
        self.assertIn("clipped surrogate", fields["implementation_obligations"])
        self.assertIn("torch_ppo_mind.py", fields["metrics_to_compare_in_our_repo"])
        self.assertIn("Do not reproduce Atari/MuJoCo", fields["what_we_need_to_reproduce"])
        self.assertEqual(fields["main_result"], "PPO improves benchmark returns.")

    def test_auction_template_requires_truthfulness_efficiency_and_regret(self):
        fields = {}
        apply_obligation_template(
            fields,
            {
                "role": "truthful_second_price",
                "title": "Counterspeculation, Auctions, and Competitive Sealed Tenders",
                "world": "auction_house",
                "institution": "second_price",
                "mind": "all",
                "theory_obligation": "Second-price benchmark must include truthful bidding.",
                "code_result_check": "worlds/auction_house/benchmarks.py",
            },
        )
        combined = " ".join(str(value) for value in fields.values()).lower()
        self.assertIn("truthful", combined)
        self.assertIn("allocative efficiency", combined)
        self.assertIn("regret", combined)

    def test_public_goods_template_requires_brackets_and_state_metrics(self):
        fields = {}
        apply_obligation_template(
            fields,
            {
                "role": "public_goods_theory",
                "title": "The Pure Theory of Public Expenditure",
                "world": "public_goods",
                "institution": "none",
                "mind": "all",
                "theory_obligation": "Report free-rider versus social-optimum brackets.",
                "code_result_check": "worlds/public_goods/benchmarks.py",
            },
        )
        combined = " ".join(str(value) for value in fields.values()).lower()
        self.assertIn("free-rider", combined)
        self.assertIn("social-optimum", combined)
        self.assertIn("sustainability", combined)

    def test_resource_template_requires_activation_diagnostics(self):
        fields = {}
        apply_obligation_template(
            fields,
            {
                "role": "common_pool_governance",
                "title": "Governing the Commons",
                "world": "resource_island",
                "institution": "property_rights,reputation_system",
                "mind": "all",
                "theory_obligation": "Property/reputation mechanisms must have activation diagnostics.",
                "code_result_check": "Resource Island v1 property opportunities.",
            },
        )
        combined = " ".join(str(value) for value in fields.values()).lower()
        self.assertIn("property opportunity", combined)
        self.assertIn("violation", combined)
        self.assertIn("activation", combined)

    def test_labor_template_requires_stability_and_side_specific_incentives(self):
        fields = {}
        apply_obligation_template(
            fields,
            {
                "role": "incentives",
                "title": "The Economics of Matching: Stability and Incentives",
                "world": "labor_market",
                "institution": "deferred_acceptance",
                "mind": "all",
                "theory_obligation": "Do not interpret worker-side misreport profits under worker-proposing DA as expected theory.",
                "code_result_check": "labor_market_benchmark_cases.json",
            },
        )
        combined = " ".join(str(value) for value in fields.values()).lower()
        self.assertIn("strategy-proof", combined)
        self.assertIn("proposing side", combined)
        self.assertIn("stability", combined)

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

    def test_audit_obligations_maps_cross_world_methods_to_minds_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cards_dir = root / "literature/foundation_paper_cards"
            cards_dir.mkdir(parents=True)
            (root / "minds/deep_rl").mkdir(parents=True)
            (root / "minds/deep_rl/torch_ppo_mind.py").write_text("clipped surrogate", encoding="utf-8")
            (root / "worlds").mkdir()
            (root / "worlds/mind_ladder.py").write_text("ppo", encoding="utf-8")
            (root / "outputs/phase3_full").mkdir(parents=True)
            (root / "outputs/phase3_full/mind_comparison.csv").write_text("mind,welfare_mean\nppo,1\n", encoding="utf-8")
            card = cards_dir / "2017_proximal_policy_optimization_algorithms.md"
            card.write_text(
                """
## World

cross_world_methods

## Theoretical benchmark

PPO method benchmark.

## Metrics

Policy-gradient learning curves.

## What we need to reproduce

Clipped surrogate PPO implementation in this repo.
""".strip(),
                encoding="utf-8",
            )
            rows = card_obligation_rows(cards_dir, root)

        self.assertEqual({row.status for row in rows}, {"partial"})
        self.assertTrue(all("minds" in row.code_evidence for row in rows))
        self.assertTrue(all(row.output_evidence != "none" for row in rows))

    def test_world_obligation_bridge_writes_six_question_scaffold(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cards_dir = root / "literature/foundation_paper_cards"
            cards_dir.mkdir(parents=True)
            (cards_dir / "1961_vickrey.md").write_text(
                """
# Counterspeculation, Auctions, and Competitive Sealed Tenders

Year: 1961
World: auction_house
Confidence: high

## Theoretical benchmark

Truthful second-price bidding.

## Main result

Second-price auction theory anchors truthfulness and efficiency checks.

## Metrics to compare in our repo

revenue; allocative_efficiency; regret

## Project code comparison

Compare against worlds/auction_house/benchmarks.py and outputs/auction_house_full.

## Failure modes to audit

Revenue without regret can hide incentive failures.

## What they do NOT test

Learning dynamics.
""".strip(),
                encoding="utf-8",
            )
            gap = root / "literature/theory_gap_report.csv"
            gap.write_text(
                "\n".join(
                    [
                        "world,classical_prediction,known_rl_marl_result,benchmark_to_reproduce,prior_metric,our_metric,remaining_gap",
                        "auction_house,Second-price truthfulness.,Learning auctions use regret.,truthful Vickrey,revenue/regret,revenue/efficiency/regret,cross-world ladder auction behavior",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            novelty = root / "literature/novelty_gap_table.csv"
            novelty.write_text(
                "\n".join(
                    [
                        "world,institution,mind,closest_paper,theory_benchmark,their_metric,our_metric,gap",
                        "auction_house,second_price,q_learning,Vickrey,truthful,revenue,revenue,known auction theory in ladder",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            text = build_world_obligation_bridge(
                cards_dir=cards_dir,
                theory_gap_report=gap,
                novelty_gap_table=novelty,
            )

        for heading in [
            "### Classical Benchmark",
            "### Known RL/MARL Result",
            "### Metric Obligation",
            "### What Our Code Reproduces Or Validates",
            "### What Our Result Adds",
            "### What This World Still Cannot Claim",
        ]:
            self.assertIn(heading, text)
        self.assertIn("truthful Vickrey", text)
        self.assertIn("Claim Sentence", text)

    def test_theory_code_audit_flags_rerun_gates_separately_from_passing_gates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "worlds/resource_island").mkdir(parents=True)
            (root / "worlds/resource_island/env.py").write_text(
                "property_opportunities trade_food_units trade_wood_units specialization trade_radius",
                encoding="utf-8",
            )
            (root / "worlds/resource_island/training.py").write_text("v1 resource_island", encoding="utf-8")
            (root / "run_resource_island_smoke.py").write_text("v1 resource_island", encoding="utf-8")
            (root / "outputs/resource_island_v1_full").mkdir(parents=True)
            (root / "outputs/resource_island_v1_full/summary_aggregate.csv").write_text(
                ",".join(
                    [
                        "trade_count_mean",
                        "trade_attempt_count_mean",
                        "trade_institution_blocked_count_mean",
                        "property_opportunities_mean",
                        "property_resource_opportunities_mean",
                        "specialization_index_mean",
                        "resource_sustainability_mean",
                    ]
                )
                + "\n1,1,1,1,1,1,1\n",
                encoding="utf-8",
            )
            rows = build_theory_code_audit(root)
            write_theory_code_audit_csv(rows, root / "literature/theory_code_audit.csv")
            write_theory_code_audit_markdown(rows, root / "literature/theory_code_audit.md")
            by_gate = {row.gate: row for row in rows}
            self.assertEqual(by_gate["resource_v1_activation_pressure"].status, "pass")
            self.assertNotEqual(by_gate["resource_v1_neural_ladder_apples_to_apples"].status, "pass")
            self.assertEqual(
                by_gate["resource_v1_neural_ladder_apples_to_apples"].decision,
                "finish_or_rerun_before_final_claim",
            )
            self.assertTrue((root / "literature/theory_code_audit.csv").exists())
            self.assertTrue((root / "literature/theory_code_audit.md").exists())

    def test_new_cli_subcommands_are_registered(self):
        parser = build_parser()
        fill_args = parser.parse_args(["fill-cards", "--limit", "2", "--model", "llama3.2:3b"])
        self.assertEqual(fill_args.limit, 2)
        self.assertEqual(fill_args.model, "llama3.2:3b")
        self.assertEqual(fill_args.num_ctx, 8192)
        hydrate_args = parser.parse_args(["hydrate-text", "--limit", "2", "--resolve-pdfs"])
        self.assertEqual(hydrate_args.limit, 2)
        self.assertTrue(hydrate_args.resolve_pdfs)
        review_args = parser.parse_args(["review", "--per-world-limit", "3"])
        self.assertEqual(review_args.per_world_limit, 3)
        scholar_args = parser.parse_args(["scholar-compare", "--threshold", "0.8"])
        self.assertEqual(scholar_args.threshold, 0.8)
        foundation_args = parser.parse_args(["foundation-papers"])
        self.assertEqual(foundation_args.out_csv, "literature/foundation_papers.csv")
        full_args = parser.parse_args(
            [
                "full",
                "--download",
                "--fill-cards",
                "--fill-limit",
                "4",
                "--fill-per-world-limit",
                "2",
            ]
        )
        self.assertTrue(full_args.download)
        self.assertTrue(full_args.fill_cards)
        self.assertEqual(full_args.fill_limit, 4)
        self.assertEqual(full_args.fill_per_world_limit, 2)
        audit_args = parser.parse_args(["audit-obligations", "--no-card-obligations"])
        self.assertTrue(audit_args.no_card_obligations)
        bridge_args = parser.parse_args(["world-obligation-bridge"])
        self.assertEqual(bridge_args.out_md, "paper/theory_obligations_by_world.generated.md")
        theory_code_args = parser.parse_args(["theory-code-audit", "--fail-on-action"])
        self.assertTrue(theory_code_args.fail_on_action)
        self.assertEqual(theory_code_args.out_md, "literature/theory_code_audit.md")


if __name__ == "__main__":
    unittest.main()
