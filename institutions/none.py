from __future__ import annotations

from typing import Any

from core.institution import Institution
from core.registry import register_institution


@register_institution("none")
class NoInstitution(Institution):
    """No-op institution."""

    name = "none"

    def apply(self, state: dict[str, Any]) -> dict[str, Any]:
        return dict(state)

