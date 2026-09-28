from __future__ import annotations
from dataclasses import dataclass


@dataclass
class RoutingDecision:
    line: str | None
    confidence: float
    status: str
    probabilities: dict[str, float]


class LineRecommendator:
    def __init__(self, high_threshold: float = 0.80, review_threshold: float = 0.60, min_margin: float = 0.15):
        self.high_threshold = high_threshold
        self.review_threshold = review_threshold
        self.min_margin = min_margin

    def recommend(self, probabilities: dict[str, float]) -> RoutingDecision:
        ranked = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        top_line, top_prob = ranked[0]
        second_prob = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = top_prob - second_prob

        if top_prob < self.review_threshold or margin < self.min_margin:
            return RoutingDecision(None, top_prob, "manual_review", probabilities)
        if top_prob >= self.high_threshold:
            return RoutingDecision(top_line, top_prob, "high_confidence", probabilities)
        return RoutingDecision(top_line, top_prob, "medium_confidence", probabilities)
