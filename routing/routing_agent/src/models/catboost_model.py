from __future__ import annotations

from catboost import CatBoostClassifier


def build_catboost() -> CatBoostClassifier:
    """
    CatBoost-модель для определения линии поддержки
    по структурированным признакам обращения.
    """

    return CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        loss_function="MultiClass",
        eval_metric="TotalF1:average=Macro",
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )