from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict


class LoaderBase(ABC):
    """
    Base class for all sensor loaders. Subclasses handle loading and optional
    preprocessing, returning a structured payload (dict/dataclass).
    """

    def __init__(self, source: Path):
        self.source = Path(source)

    @abstractmethod
    def load(self) -> Any:
        """Load raw data from disk."""

    def preprocess(self, data: Any) -> Any:
        """Hook for subclasses to clean/normalize data before consumption."""
        return data

    def run(self) -> Any:
        payload = self.load()
        return self.preprocess(payload)
