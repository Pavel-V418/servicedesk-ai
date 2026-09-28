from __future__ import annotations

from src.policy.line_recommendator import LineRecommendator


class RoutingAgent:
    def __init__(
        self,
        model,
        recommendator: LineRecommendator | None = None,
        model_version: str = "router-v0.1",
    ):
        self.model = model
        self.recommendator = recommendator or LineRecommendator()
        self.model_version = model_version

    def route(self, x_one):
        probs = self.model.predict_proba(x_one)[0]
        classes = self.model.classes_

        probability_map = {
            str(cls): float(p)
            for cls, p in zip(classes, probs)
        }

        decision = self.recommendator.recommend(
            probability_map
        )

        return {
            "recommended_line": decision.line,
            "confidence": decision.confidence,
            "status": decision.status,
            "probabilities": decision.probabilities,
            "model_version": self.model_version,
        }