from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from src.features.feature_builder import prepare_features
from src.models.route_predictor import build_route_predictor


TARGET = "Кем решен (группа)"
DATE_COL = "Дата регистрации"

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

EXCEL_PATH = Path(
    "/Users/armancho/Downloads/Обращения_1931.xlsx"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "confidence_analysis.csv"
)

CLASSES = ["L1", "L2", "L3"]


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


def main():

    # ========================================================
    # 1. Загружаем данные
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
    # 2. Final test опять откладываем
    # ========================================================

    split_index = int(len(df) * 0.8)

    development_df = df.iloc[:split_index].copy()
    final_test_df = df.iloc[split_index:].copy()

    print("Temporal split:")
    print(f"Development: {len(development_df)}")
    print(f"Final test:  {len(final_test_df)}")

    X = prepare_features(development_df)
    y = development_df[TARGET]

    # ========================================================
    # 3. Cross-validation
    # ========================================================

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    all_rows = []

    for fold, (train_idx, val_idx) in enumerate(
        cv.split(X, y),
        start=1,
    ):

        print(f"\nFold {fold}")

        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]

        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        # ВАЖНО:
        # для каждого fold создаём и обучаем
        # совершенно новую модель.

        model = build_route_predictor()

        model.fit(
            X_train,
            y_train,
        )

        probabilities = model.predict_proba(
            X_val
        )

        classes = model.classes_

        for position, row_index in enumerate(val_idx):

            row_probabilities = probabilities[position]

            # Сортируем вероятности от большей к меньшей
            order = np.argsort(
                row_probabilities
            )[::-1]

            top_index = order[0]
            second_index = order[1]

            predicted = classes[top_index]

            confidence = float(
                row_probabilities[top_index]
            )

            second_probability = float(
                row_probabilities[second_index]
            )

            margin = (
                confidence
                - second_probability
            )

            actual = y.iloc[row_index]

            correct = predicted == actual

            probability_dict = {
                class_name: float(probability)
                for class_name, probability
                in zip(classes, row_probabilities)
            }

            original_row = development_df.iloc[row_index]

            all_rows.append({
                "fold": fold,
                "row_index": row_index,

                "actual": actual,
                "predicted": predicted,
                "correct": correct,

                "confidence": confidence,
                "margin": margin,

                "L1_probability":
                    probability_dict.get("L1", 0.0),

                "L2_probability":
                    probability_dict.get("L2", 0.0),

                "L3_probability":
                    probability_dict.get("L3", 0.0),

                # Исходные признаки обращения
                "Описание 2":
                    original_row.get("Описание 2", ""),

                "Услуга":
                    original_row.get("Услуга", ""),

                "Компонент услуги 1 уровня":
                    original_row.get(
                        "Компонент услуги 1 уровня",
                        "",
                    ),

                "Вид запроса":
                    original_row.get("Вид запроса", ""),

                "Тип запроса":
                    original_row.get("Тип запроса", ""),

                "Критичность":
                    original_row.get("Критичность", ""),

                "Срочность":
                    original_row.get("Срочность", ""),

                "Приоритет":
                    original_row.get("Приоритет", ""),

                "Класс обслуживания":
                    original_row.get(
                        "Класс обслуживания",
                        "",
                    ),

                "Пользователь":
                    original_row.get("Пользователь", ""),
            })
    # ========================================================
    # 4. Собираем OOF predictions
    # ========================================================

    results = pd.DataFrame(
        all_rows
    )

    print("\n" + "=" * 70)
    print("OUT-OF-FOLD RESULTS")
    print("=" * 70)

    print(
        f"Количество обращений: {len(results)}"
    )

    print(
        f"Accuracy: {accuracy_score(results['actual'], results['predicted']):.4f}"
    )

    # ========================================================
    # 5. Анализ confidence
    # ========================================================

    print("\n" + "=" * 70)
    print("CONFIDENCE ANALYSIS")
    print("=" * 70)

    confidence_groups = [
        ("< 0.60", 0.0, 0.60),
        ("0.60 - 0.80", 0.60, 0.80),
        (">= 0.80", 0.80, 1.01),
    ]

    for name, lower, upper in confidence_groups:

        group = results[
            (results["confidence"] >= lower)
            & (results["confidence"] < upper)
        ]

        if len(group) == 0:
            continue

        accuracy = group[
            "correct"
        ].mean()

        share = (
            len(group)
            / len(results)
        )

        print(f"\n{name}")
        print(f"Количество: {len(group)}")
        print(f"Доля:       {share:.2%}")
        print(f"Accuracy:   {accuracy:.2%}")

    # ========================================================
    # 6. Проверяем текущий LineRecommendator
    # ========================================================

    manual_review = (
        (results["confidence"] < 0.60)
        | (results["margin"] < 0.15)
    )

    automatic = ~manual_review

    print("\n" + "=" * 70)
    print("CURRENT ROUTING POLICY")
    print("=" * 70)

    print(
        f"Manual review: {manual_review.sum()}"
    )

    print(
        f"Automatic:     {automatic.sum()}"
    )

    print(
        f"Manual review rate: {manual_review.mean():.2%}"
    )

    if automatic.sum() > 0:

        automatic_accuracy = (
            results.loc[
                automatic,
                "correct"
            ].mean()
        )

        print(
            f"Automatic routing accuracy: "
            f"{automatic_accuracy:.2%}"
        )

    # ========================================================
    # 7. Сохраняем
    # ========================================================

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        REPORT_PATH,
        index=False,
    )

    print(
        f"\nРезультаты сохранены:\n{REPORT_PATH}"
    )


if __name__ == "__main__":
    main()