from dataclasses import dataclass
from typing import Any, Mapping


@dataclass
class ReasonScore:
    reason: str
    score: float

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ReasonScore":
        return cls(
            reason=str(payload["reason"]),
            score=float(payload["score"]),
        )
