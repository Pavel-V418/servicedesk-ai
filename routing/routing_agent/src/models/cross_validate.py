from __future__ import annotations

from src.paths import DEFAULT_XLSX

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold

from src.features.feature_builder import prepare_features
from src.models.baseline import build_baseline


TARGET = "Кем решен (группа)"

EXCEL_PATH = Path(
    str(DEFAULT_XLSX)
)

CLASSES = ["L1", "L2", "L3"]


def normalize_target(series: pd.Series) -> pd.Series:
    """
    Приводим названия линий к L1/L2/L3/L4.
    """

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


def load_data() -> pd.DataFrame:
    """
    Загружаем Excel и подготавливаем target.
    """

    if not EXCEL_PATH.exists():
        raise FileNotFoundError(
            f"Excel-файл не найден: {EXCEL_PATH}"
        )

    print(f"Загружаем данные из:")
    print(EXCEL_PATH)

    df = pd.read_excel(EXCEL_PATH)

    if TARGET not in df.columns:
        raise ValueError(
            f"В датасете отсутствует колонка '{TARGET}'"
        )

    df[TARGET] = normalize_target(df[TARGET])

    # L4 пока исключаем,
    # потому что у нас только один пример.
    df = df[
        df[TARGET].isin(CLASSES)
    ].copy()

    return df


def main() -> None:

    # ========================================================
    # 1. ЗАГРУЗКА
    # ========================================================

    df = load_data()

    print(f"\nКоличество обращений: {len(df)}")

    print("\nРаспределение классов:")
    print(df[TARGET].value_counts())

    # ========================================================
    # 2. ПРИЗНАКИ
    # ========================================================
    DATE_COL = "Дата регистрации"

    df[DATE_COL] = pd.to_datetime(
        df[DATE_COL],
        errors="coerce",
    )

    df = df.dropna(subset=[DATE_COL])

    # Сортируем обращения от старых к новым
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    # Последние 20% оставляем как финальный temporal test
    split_index = int(len(df) * 0.8)

    development_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    print("\nTemporal split:")
    print(f"Development: {len(development_df)}")
    print(f"Final test:  {len(test_df)}")

    print("\nDevelopment classes:")
    print(development_df[TARGET].value_counts())

    print("\nFinal test classes:")
    print(test_df[TARGET].value_counts())

    # Cross-validation работает ТОЛЬКО на development
    X = prepare_features(development_df)
    y = development_df[TARGET]



    # ========================================================
    # 3. STRATIFIED K-FOLD
    # ========================================================

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    # Здесь будем хранить результаты каждого fold.
    results = []

    # ========================================================
    # 4. ОБУЧЕНИЕ
    # ========================================================

    for fold, (train_index, val_index) in enumerate(
        cv.split(X, y),
        start=1,
    ):

        print("\n" + "=" * 60)
        print(f"FOLD {fold}")
        print("=" * 60)

        X_train = X.iloc[train_index]
        X_val = X.iloc[val_index]

        y_train = y.iloc[train_index]
        y_val = y.iloc[val_index]

        print(f"Train: {len(X_train)}")
        print(f"Validation: {len(X_val)}")

        print("\nValidation classes:")
        print(y_val.value_counts().to_dict())

        # Каждый fold получает совершенно новую модель.
        model = clone(build_baseline())

        model.fit(
            X_train,
            y_train,
        )

        pred = model.predict(X_val)

        # ====================================================
        # ОБЩИЕ МЕТРИКИ
        # ====================================================

        accuracy = accuracy_score(
            y_val,
            pred,
        )

        macro_f1 = f1_score(
            y_val,
            pred,
            average="macro",
            zero_division=0,
        )

        weighted_f1 = f1_score(
            y_val,
            pred,
            average="weighted",
            zero_division=0,
        )

        # ====================================================
        # МЕТРИКИ ПО КАЖДОЙ ЛИНИИ
        # ====================================================

        recalls = recall_score(
            y_val,
            pred,
            labels=CLASSES,
            average=None,
            zero_division=0,
        )

        precisions = precision_score(
            y_val,
            pred,
            labels=CLASSES,
            average=None,
            zero_division=0,
        )

        f1_per_class = f1_score(
            y_val,
            pred,
            labels=CLASSES,
            average=None,
            zero_division=0,
        )

        fold_result = {
            "fold": fold,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,

            "L1_precision": precisions[0],
            "L1_recall": recalls[0],
            "L1_f1": f1_per_class[0],

            "L2_precision": precisions[1],
            "L2_recall": recalls[1],
            "L2_f1": f1_per_class[1],

            "L3_precision": precisions[2],
            "L3_recall": recalls[2],
            "L3_f1": f1_per_class[2],
        }

        results.append(fold_result)

        print(f"\nAccuracy:    {accuracy:.4f}")
        print(f"Macro-F1:    {macro_f1:.4f}")
        print(f"Weighted-F1: {weighted_f1:.4f}")

        print("\nRecall:")

        for class_name, recall in zip(
            CLASSES,
            recalls,
        ):
            print(
                f"{class_name}: {recall:.4f}"
            )

    # ========================================================
    # 5. ИТОГОВАЯ ТАБЛИЦА
    # ========================================================

    results_df = pd.DataFrame(results)

    print("\n")
    print("=" * 60)
    print("ИТОГИ CROSS-VALIDATION")
    print("=" * 60)

    print(
        results_df[
            [
                "fold",
                "accuracy",
                "macro_f1",
                "weighted_f1",
                "L1_recall",
                "L2_recall",
                "L3_recall",
            ]
        ].round(4).to_string(index=False)
    )

    # ========================================================
    # 6. СРЕДНИЕ ЗНАЧЕНИЯ
    # ========================================================

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
    # 7. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
    # ========================================================

    reports_dir = Path("reports")

    reports_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        reports_dir
        / "cross_validation_results.csv"
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nРезультаты сохранены: {output_path}"
    )

    print("\nCross-validation завершена!")


if __name__ == "__main__":
    main()