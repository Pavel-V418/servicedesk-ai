from __future__ import annotations

from src.paths import DEFAULT_XLSX

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold

from src.features.feature_builder import prepare_features
from src.models.baseline import build_baseline
from src.models.catboost_model import build_catboost
from src.models.ensemble_model import combine_probabilities


TARGET = "Кем решен (группа)"
DATE_COL = "Дата регистрации"

EXCEL_PATH = Path(
    str(DEFAULT_XLSX)
)

CLASSES = ["L1", "L2", "L3"]

CATEGORICAL_COLS = [
    "Пользователь",
    "Услуга",
    "Компонент услуги 1 уровня",
    "Вид запроса",
    "Тип запроса",
    "Критичность",
    "Срочность",
    "Приоритет",
    "Класс обслуживания",
    "Часовой пояс запроса",
]


def normalize_target(series: pd.Series) -> pd.Series:
    mapping = {
        "(1 линия)": "L1",
        "(2 линия)": "L2",
        "(3 линия)": "L3",
        "(4 линия)": "L4",
    }

    series = (
        series
        .fillna("")
        .astype(str)
        .str.strip()
    )

    return series.map(mapping).fillna(series)


def prepare_catboost_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    X = df[CATEGORICAL_COLS].copy()

    for col in CATEGORICAL_COLS:
        X[col] = (
            X[col]
            .fillna("__MISSING__")
            .astype(str)
        )

    return X


def align_probabilities(
    probabilities: np.ndarray,
    model_classes,
) -> np.ndarray:
    """
    Приводит столбцы predict_proba к единому порядку:
    L1, L2, L3.
    """

    class_to_index = {
        class_name: index
        for index, class_name in enumerate(model_classes)
    }

    return probabilities[
        :,
        [class_to_index[class_name] for class_name in CLASSES],
    ]


def main() -> None:

    # ========================================================
    # 1. DATA
    # ========================================================

    df = pd.read_excel(EXCEL_PATH)

    df[TARGET] = normalize_target(df[TARGET])

    df = df[
        df[TARGET].isin(CLASSES)
    ].copy()

    df[DATE_COL] = pd.to_datetime(
        df[DATE_COL],
        errors="coerce",
    )

    df = df.dropna(subset=[DATE_COL])

    df = (
        df
        .sort_values(DATE_COL)
        .reset_index(drop=True)
    )

    # ========================================================
    # 2. TEMPORAL HOLDOUT
    # ========================================================

    split_index = int(len(df) * 0.8)

    development_df = df.iloc[:split_index].copy()
    final_test_df = df.iloc[split_index:].copy()

    print("\nTemporal split:")
    print(f"Development: {len(development_df)}")
    print(f"Final test:  {len(final_test_df)}")

    print("\nDevelopment classes:")
    print(development_df[TARGET].value_counts())

    print("\nFinal test classes:")
    print(final_test_df[TARGET].value_counts())

    # Две модели получают разные представления
    # ОДНИХ И ТЕХ ЖЕ обращений.

    X_baseline = prepare_features(development_df)

    X_catboost = prepare_catboost_features(
        development_df
    )

    y = development_df[TARGET]

    # ========================================================
    # 3. CROSS-VALIDATION
    # ========================================================

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    cat_features = list(
        range(len(CATEGORICAL_COLS))
    )

    results = []

    for fold, (train_idx, val_idx) in enumerate(
        cv.split(X_baseline, y),
        start=1,
    ):

        print("\n" + "=" * 60)
        print(f"FOLD {fold}")
        print("=" * 60)

        # ----------------------------------------------------
        # SAME TRAIN / VALIDATION INDICES FOR BOTH MODELS
        # ----------------------------------------------------

        Xb_train = X_baseline.iloc[train_idx]
        Xb_val = X_baseline.iloc[val_idx]

        Xc_train = X_catboost.iloc[train_idx]
        Xc_val = X_catboost.iloc[val_idx]

        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        print(f"Train: {len(train_idx)}")
        print(f"Validation: {len(val_idx)}")

        # ====================================================
        # 4. BASELINE
        # ====================================================

        baseline = build_baseline()

        baseline.fit(
            Xb_train,
            y_train,
        )

        baseline_proba = baseline.predict_proba(
            Xb_val
        )

        baseline_proba = align_probabilities(
            baseline_proba,
            baseline.classes_,
        )

        # ====================================================
        # 5. CATBOOST
        # ====================================================

        catboost = build_catboost()

        catboost.fit(
            Xc_train,
            y_train,
            cat_features=cat_features,
        )

        catboost_proba = catboost.predict_proba(
            Xc_val
        )

        catboost_proba = align_probabilities(
            catboost_proba,
            catboost.classes_,
        )

        # ====================================================
        # 6. ENSEMBLE
        # ====================================================

        combined_proba = combine_probabilities(
            baseline_proba,
            catboost_proba,
            baseline_weight=0.6,
        )

        pred_indices = np.argmax(
            combined_proba,
            axis=1,
        )

        predictions = np.array(CLASSES)[
            pred_indices
        ]

        # ====================================================
        # 7. METRICS
        # ====================================================

        accuracy = accuracy_score(
            y_val,
            predictions,
        )

        macro_f1 = f1_score(
            y_val,
            predictions,
            average="macro",
            zero_division=0,
        )

        weighted_f1 = f1_score(
            y_val,
            predictions,
            average="weighted",
            zero_division=0,
        )

        recalls = recall_score(
            y_val,
            predictions,
            labels=CLASSES,
            average=None,
            zero_division=0,
        )

        results.append({
            "fold": fold,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "L1_recall": recalls[0],
            "L2_recall": recalls[1],
            "L3_recall": recalls[2],
        })

        print(f"Accuracy:    {accuracy:.4f}")
        print(f"Macro-F1:    {macro_f1:.4f}")
        print(f"Weighted-F1: {weighted_f1:.4f}")

        print("\nRecall:")
        print(f"L1: {recalls[0]:.4f}")
        print(f"L2: {recalls[1]:.4f}")
        print(f"L3: {recalls[2]:.4f}")

    # ========================================================
    # 8. RESULTS
    # ========================================================

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 60)
    print("ENSEMBLE — ИТОГИ CROSS-VALIDATION")
    print("=" * 60)

    print(
        results_df.round(4).to_string(
            index=False
        )
    )

    print("\n" + "=" * 60)
    print("СРЕДНИЕ МЕТРИКИ")
    print("=" * 60)

    metrics = [
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "L1_recall",
        "L2_recall",
        "L3_recall",
    ]

    for metric in metrics:

        mean = results_df[metric].mean()
        std = results_df[metric].std()

        print(
            f"{metric:15s}: "
            f"{mean:.4f} ± {std:.4f}"
        )

    # ========================================================
    # 9. SAVE
    # ========================================================

    reports_dir = Path("reports")

    reports_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        reports_dir
        / "ensemble_cross_validation.csv"
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nРезультаты сохранены: {output_path}"
    )


if __name__ == "__main__":
    main()