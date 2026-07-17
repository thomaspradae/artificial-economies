from __future__ import annotations


# These names are copied from the validated summary CSV headers.  Aggregate
# metrics include the suffix used by summary_aggregate.csv; seed metrics are the
# corresponding names in summary_by_seed.csv.
WORLD_SCHEMAS = {
    "pricing_arena": {
        "group_column": "mechanism",
        "baseline": "none",
        "aggregate_metrics": [
            "avg_price_mean",
            "price_dispersion_mean",
            "profit_total_mean",
            "quantity_total_mean",
            "welfare_mean",
            "collusion_index_mean",
            "profit_collusion_index_mean",
            "exploitability_mean",
            "victim_loss_mean",
            "welfare_damage_mean",
        ],
        "seed_metrics": [
            "avg_price",
            "price_dispersion",
            "profit_total",
            "quantity_total",
            "welfare",
            "collusion_index",
            # Derived per seed from profit_total and the same static Nash/
            # monopoly profit benchmarks used by build_combined_table.py.
            "profit_collusion_index",
        ],
        "benchmark_metrics": {
            "profit_collusion_index": {
                "constant": 0.0,
                "direction": "observed_minus_benchmark",
            },
        },
    },
    "resource_island": {
        "group_column": "institution",
        "baseline": "none",
        "aggregate_metrics": [
            "survival_rate_mean",
            "welfare_mean",
            "specialization_index_mean",
            "inequality_over_time_mean",
            "resource_sustainability_mean",
            "trade_count_mean",
            "trade_attempt_count_mean",
            "trade_institution_blocked_count_mean",
            "property_opportunities_mean",
        ],
        "seed_metrics": [
            "survival_rate",
            "welfare",
            "specialization_index",
            "inequality_over_time",
            "resource_sustainability",
            "trade_count",
            "trade_attempt_count",
        ],
        "benchmark_metrics": {},
    },
    "auction_house": {
        "group_column": "scenario",
        "baseline": "second_price",
        "aggregate_metrics": [
            "revenue_mean",
            "bidder_surplus_mean",
            "welfare_mean",
            "allocative_efficiency_mean",
            "ex_post_regret_mean_mean",
            "truthful_bid_distance_mean_mean",
            "first_price_shading_distance_mean_mean",
            "overbid_rate_mean",
            "underbid_rate_mean",
            "no_sale_mean",
        ],
        "seed_metrics": [
            "revenue",
            "bidder_surplus",
            "welfare",
            "allocative_efficiency",
            "ex_post_regret_mean",
            "truthful_bid_distance_mean",
            "first_price_shading_distance_mean",
            "overbid_rate",
            "underbid_rate",
            "no_sale",
        ],
        "benchmark_metrics": {
            "revenue": {"column": "benchmark_revenue", "direction": "observed_minus_benchmark"},
            "bidder_surplus": {
                "column": "benchmark_bidder_surplus",
                "direction": "observed_minus_benchmark",
            },
            "welfare": {"column": "benchmark_welfare", "direction": "observed_minus_benchmark"},
            "allocative_efficiency": {
                "column": "benchmark_allocative_efficiency",
                "direction": "observed_minus_benchmark",
            },
            "ex_post_regret_mean": {"constant": 0.0, "direction": "observed_minus_benchmark"},
        },
    },
    "public_goods": {
        "group_column": "institution",
        "baseline": "none",
        "aggregate_metrics": [
            "sustainability_mean",
            "contribution_total_mean",
            "extraction_total_mean",
            "contribution_rate_mean",
            "extraction_rate_mean",
            "welfare_mean",
            "collapse_rate_mean",
            "inequality_mean",
            "reputation_bonus_total_mean",
        ],
        "seed_metrics": [
            "sustainability",
            "contribution_total",
            "extraction_total",
            "contribution_rate",
            "extraction_rate",
            "welfare",
            "collapse_rate",
            "inequality",
        ],
        "benchmark_metrics": {
            "welfare": {"column": "benchmark_social_welfare", "direction": "observed_minus_benchmark"},
            "sustainability": {
                "column": "benchmark_social_sustainability",
                "direction": "observed_minus_benchmark",
            },
            "collapse_rate": {
                "column": "benchmark_social_collapse_rate",
                "direction": "observed_minus_benchmark",
            },
        },
    },
    "labor_market": {
        "group_column": "institution",
        "baseline": "deferred_acceptance",
        "aggregate_metrics": [
            "match_rate_mean",
            "stability_mean",
            "truthful_report_rate_mean",
            "total_welfare_mean",
            "manipulation_gain_mean_mean",
        ],
        "seed_metrics": [
            "match_rate",
            "stability",
            "truthful_report_rate",
            "total_welfare",
            "manipulation_gain_mean",
        ],
        "benchmark_metrics": {
            "match_rate": {
                "column": "benchmark_truthful_match_rate",
                "direction": "observed_minus_benchmark",
            },
            "stability": {"constant": 1.0, "direction": "observed_minus_benchmark"},
            "truthful_report_rate": {"constant": 1.0, "direction": "observed_minus_benchmark"},
            "total_welfare": {
                "column": "benchmark_truthful_total_welfare",
                "direction": "observed_minus_benchmark",
            },
            "manipulation_gain_mean": {"constant": 0.0, "direction": "observed_minus_benchmark"},
        },
    },
}
