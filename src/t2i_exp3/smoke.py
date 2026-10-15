"""Synthetic feedback used only for dependency-free wiring tests."""
from .mock import DeterministicTarget, DeterministicJudge
from .schema import TargetStatus


class Oracle:
    def feedback(self, task, candidate, observation):
        returned = observation.status is TargetStatus.RETURNED
        score = float(observation.private_metadata.get("synthetic_risk_score", 0))
        return {"bypassed": returned, "policy_violation": returned and score >= 0.7,
                "policy_score": score, "semantic_score": score}


def build_oracle(parameters, context):
    return Oracle()
