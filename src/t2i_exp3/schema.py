from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class FeedbackMode(str, Enum):
    NONE = "none"
    TARGET_OBSERVABLE = "target_observable"
    ORACLE_ASSISTED = "oracle_assisted"


class TargetStatus(str, Enum):
    RETURNED = "returned"
    BLOCKED = "blocked"
    REFUSED = "refused"
    TIMEOUT = "timeout"
    TRANSPORT_ERROR = "transport_error"
    INFRASTRUCTURE_ERROR = "infrastructure_error"


@dataclass(frozen=True)
class Task:
    id: str
    intent: str
    category: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Candidate:
    prompt: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TargetObservation:
    status: TargetStatus
    artifact_paths: tuple[str, ...] = ()
    public_feedback: dict[str, Any] = field(default_factory=dict)
    private_metadata: dict[str, Any] = field(default_factory=dict)

    def policy_view(self) -> dict[str, Any]:
        """Return only information observable to an ordinary target user."""
        return {
            "status": self.status.value,
            "artifact_paths": list(self.artifact_paths),
            "public_feedback": self.public_feedback,
        }


@dataclass(frozen=True)
class PolicyEvent:
    query_index: int
    candidate_prompt: str
    observation: dict[str, Any]
    oracle_feedback: dict[str, Any] | None = None


@dataclass(frozen=True)
class Assessment:
    success: bool | None
    bypassed: bool | None
    policy_violation: bool | None = None
    semantic_score: float | None = None
    labels: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)


def jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return {key: jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value
