from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Annotated, Union, Literal

from pydantic import BaseModel


class Processor(BaseModel, ABC):

    name: str

    @abstractmethod
    def compute(self) -> float:
        """Compute area (must be implemented by subclasses)."""
        raise NotImplementedError
