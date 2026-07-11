"""Institution rules available to economic worlds."""

from institutions.anti_collusion import AntiCollusion
from institutions.demand_shock import DemandShock
from institutions.none import NoInstitution
from institutions.price_cap import PriceCap
from institutions.random_audit import RandomAudit
from institutions.resource_island import PropertyRights, Redistribution, ReputationSystem, TradePriceControls
from institutions.tax_high_price import TaxHighPrice

__all__ = [
    "AntiCollusion",
    "DemandShock",
    "NoInstitution",
    "PriceCap",
    "PropertyRights",
    "RandomAudit",
    "Redistribution",
    "ReputationSystem",
    "TaxHighPrice",
    "TradePriceControls",
]
