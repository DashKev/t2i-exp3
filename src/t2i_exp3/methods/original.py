from __future__ import annotations

from typing import Any, Sequence

from t2i_exp3.schema import Candidate


class OriginalPromptMethod:
    """Single-query baseline that sends the raw task intent to the target.

    This adapter exists for audited timing/latency measurements.  It does not
    rewrite the prompt and intentionally exposes only one candidate.
    """

    @property
    def maximum_queries(self) -> int:
        return 1

    def should_stop(self, task: Any, history: Sequence[Any]) -> bool:
        return bool(history)

    def propose(self, task: Any, history: Sequence[Any], query_index: int, rng: Any) -> Candidate:
        if query_index != 1:
            raise RuntimeError("Original Prompt 只支持 query_index=1")
        return Candidate(
            str(task.intent),
            {
                "family": "original_prompt",
                "method_id": "original_prompt",
                "query_index": 1,
            },
        )

    def complete_task(self, task: Any, history: Sequence[Any]) -> None:
        return None

    def usage_snapshot(self) -> dict[str, Any]:
        return {"method_id": "original_prompt", "maximum_queries": 1}


def build(parameters: dict[str, Any], context: Any) -> OriginalPromptMethod:
    return OriginalPromptMethod()
