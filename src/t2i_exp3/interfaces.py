from __future__ import annotations

from pathlib import Path
from random import Random
from typing import Any, Protocol, Sequence

from .schema import Assessment, Candidate, PolicyEvent, TargetObservation, Task


class AttackMethod(Protocol):
    def propose(
        self,
        task: Task,
        history: Sequence[PolicyEvent],
        query_index: int,
        rng: Random,
    ) -> Candidate: ...

    def should_stop(self, task: Task, history: Sequence[PolicyEvent]) -> bool:
        """Optional early-stop hook evaluated before dispatching the next query."""
        ...

    def replay_proposal(
        self,
        task: Task,
        history: Sequence[PolicyEvent],
        query_index: int,
        candidate: Candidate,
        rng: Random,
    ) -> None:
        """Optional hook used by resume to replay proposal-side state without target queries."""
        ...

    def complete_task(self, task: Task, history: Sequence[PolicyEvent]) -> None:
        """Optional hook used to consume the final event and persist cross-task state."""
        ...


class Target(Protocol):
    def generate(
        self,
        task: Task,
        candidate: Candidate,
        generation_seed: int,
        artifact_dir: Path,
    ) -> TargetObservation: ...


class OnlineOracle(Protocol):
    def feedback(
        self,
        task: Task,
        candidate: Candidate,
        observation: TargetObservation,
    ) -> dict[str, Any]: ...


class PosthocJudge(Protocol):
    def assess(
        self,
        task: Task,
        candidate: Candidate,
        observation: TargetObservation,
    ) -> Assessment: ...


class UsageReportingPlugin(Protocol):
    def usage_snapshot(self) -> dict[str, Any]:
        """返回累计调用数、token、费用等插件内部成本；字段必须可 JSON 序列化。"""
        ...
