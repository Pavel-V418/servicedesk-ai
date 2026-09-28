from __future__ import annotations

from sklearn.pipeline import Pipeline

from src.models.baseline import build_baseline
from src.models.weighted_ovr import WeightedOneVsRestLogReg


CLASS_WEIGHT = {
    "L1": 1.0,
    "L2": 1.3,
    "L3": 1.5,
}


def build_route_predictor() -> Pipeline:
    """
    Основная ML-модель Routing Agent.

    Использует:
    - TF-IDF по тексту;
    - OneHot по метаданным;
    - Logistic Regression;
    - мягкую компенсацию дисбаланса классов.
    """

    baseline = build_baseline()

    features = baseline.named_steps["features"]

    # Было: LogisticRegression(solver="liblinear", class_weight=CLASS_WEIGHT).
    # В sklearn >= 1.8 liblinear не принимает 3 класса, поэтому та же схема
    # One-vs-Rest с весами классов реализована явно (см. weighted_ovr.py).
    model = WeightedOneVsRestLogReg(
        class_weight=CLASS_WEIGHT,
        max_iter=1200,
    )

    return Pipeline([
        ("features", features),
        ("model", model),
    ])