from __future__ import annotations

import numpy as np


def combine_probabilities(
    baseline_proba: np.ndarray,
    catboost_proba: np.ndarray,
    baseline_weight: float = 0.6,
) -> np.ndarray:
    """
    Объединяет вероятности двух моделей.

    baseline_weight = вес TF-IDF + metadata + Logistic Regression.
    Остальной вес получает CatBoost.
    """

    catboost_weight = 1.0 - baseline_weight

    combined = (
        baseline_weight * baseline_proba
        + catboost_weight * catboost_proba
    )

    return combined