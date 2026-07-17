#!/usr/bin/env python3
"""Query SerpAPI Google Scholar for the paper's literature map.

Usage:
    SERPAPI_KEY=... python scripts/serpapi_scholar_lit.py \
      --output paper/serpapi_scholar_results.csv

The script intentionally writes a CSV review queue rather than modifying the
BibTeX file automatically. Google Scholar metadata is noisy; papers should be
checked before they become thesis citations.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path


QUERIES = [
    ("core_ace", "agent-based computational economics reinforcement learning economic simulation"),
    ("core_learning_games", "reinforcement learning experimental games Erev Roth Camerer Ho EWA"),
    ("pricing_collusion", "algorithmic collusion Q-learning repeated pricing Calvano"),
    ("pricing_regulation", "algorithmic pricing regulation price caps collusion reinforcement learning"),
    ("auction_ai", "artificial intelligence auction design Q-learning first price second price"),
    ("auction_rl", "reinforcement learning auction equilibrium first price second price bidding"),
    ("resource_ssd", "multi-agent reinforcement learning sequential social dilemmas resource gathering"),
    ("resource_commons", "common pool resource multi-agent reinforcement learning institutions property rights"),
    ("public_goods", "public goods game reinforcement learning punishment reward institutions"),
    ("tax_planner", "AI Economist reinforcement learning tax policy social planner"),
    ("matching_rl", "reinforcement learning two-sided matching market stable matching"),
    ("llm_agents_games", "large language models game theory economic agents rationality"),
    ("llm_market_sim", "large language model agents market simulation economics"),
]


def fetch(api_key: str, query: str, *, num: int, start: int, sleep_s: float) -> dict:
    params = {
        "engine": "google_scholar",
        "q": query,
        "api_key": api_key,
        "num": str(num),
        "start": str(start),
    }
    url = "https://serpapi.com/search?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=30) as response:
        data = json.loads(response.read().decode("utf-8"))
    time.sleep(sleep_s)
    return data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="paper/serpapi_scholar_results.csv")
    parser.add_argument("--num", type=int, default=10)
    parser.add_argument("--pages", type=int, default=2)
    parser.add_argument("--sleep", type=float, default=1.0)
    args = parser.parse_args()

    api_key = os.environ.get("SERPAPI_KEY")
    if not api_key:
        raise SystemExit("SERPAPI_KEY is not set")

    rows = []
    for topic, query in QUERIES:
        for page in range(args.pages):
            start = page * args.num
            data = fetch(api_key, query, num=args.num, start=start, sleep_s=args.sleep)
            for result in data.get("organic_results", []):
                pub = result.get("publication_info", {}) or {}
                rows.append(
                    {
                        "topic": topic,
                        "query": query,
                        "rank": result.get("position", ""),
                        "title": result.get("title", ""),
                        "link": result.get("link", ""),
                        "result_id": result.get("result_id", ""),
                        "snippet": result.get("snippet", ""),
                        "publication_summary": pub.get("summary", ""),
                        "cited_by": ((result.get("inline_links", {}) or {}).get("cited_by", {}) or {}).get("total", ""),
                    }
                )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "topic",
                "query",
                "rank",
                "title",
                "link",
                "result_id",
                "snippet",
                "publication_summary",
                "cited_by",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
