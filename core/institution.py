from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Institution(ABC):
    """Base class for a rule that modifies state, rewards, or observations."""

    @abstractmethod
    def apply(self, state: dict[str, Any]) -> dict[str, Any]:
        """Return a modified state dict after applying this institution."""

    def reset(self) -> None:
        """Reset institution state between episodes. Default no-op."""

