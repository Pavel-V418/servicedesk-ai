from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

INPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "confidence_analysis.csv"
)

HIGH_THRESHOLD = 0.80

CLASSES = ["L1", "L2", "L3"]


def main():

    # ==========================================
    # 1. Загружаем OOF predictions
    # ==========================================

    df = pd.read_csv(INPUT_PATH)

    print(f"Всего development-обращений: {len(df)}")

    # ==========================================
    # 2. Оставляем high-confidence
    # ==========================================

    high = df[
        df["confidence"] >= HIGH_THRESHOLD
    ].copy()

    print(
        f"High-confidence обращений: {len(high)}"
    )

    print(
        f"Доля от development: "
        f"{len(high) / len(df):.2%}"
    )

    # ==========================================
    # 3. Общая accuracy
    # ==========================================

    accuracy = high["correct"].mean()

    print(
        f"High-confidence accuracy: "
        f"{accuracy:.2%}"
    )

    # ==========================================
    # 4. Classification report
    # ==========================================

    print("\n" + "=" * 70)
    print("HIGH-CONFIDENCE CLASSIFICATION REPORT")
    print("=" * 70)

    print(
        classification_report(
            high["actual"],
            high["predicted"],
            labels=CLASSES,
            zero_division=0,
            digits=4,
        )
    )

    # ==========================================
    # 5. Confusion matrix
    # ==========================================

    matrix = confusion_matrix(
        high["actual"],
        high["predicted"],
        labels=CLASSES,
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=[
            f"Actual_{c}"
            for c in CLASSES
        ],
        columns=[
            f"Predicted_{c}"
            for c in CLASSES
        ],
    )

    print("\n" + "=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)

    print(matrix_df)

    # ==========================================
    # 6. Сколько модель автоматически
    #    отправляет на каждую линию
    # ==========================================

    print("\n" + "=" * 70)
    print("PREDICTIONS BY LINE")
    print("=" * 70)

    for line in CLASSES:

        predicted_line = high[
            high["predicted"] == line
        ]

        count = len(predicted_line)

        if count == 0:
            print(
                f"{line}: предсказаний нет"
            )
            continue

        precision = (
            predicted_line["correct"].mean()
        )

        print(
            f"{line}: "
            f"{count} обращений | "
            f"правильно {precision:.2%}"
        )

    # ==========================================
    # 7. Только ошибки high-confidence
    # ==========================================

    errors = high[
        ~high["correct"]
    ].copy()

    print("\n" + "=" * 70)
    print("HIGH-CONFIDENCE ERRORS")
    print("=" * 70)

    print(
        f"Всего ошибок: {len(errors)}"
    )

    if len(errors) > 0:

        error_types = (
            errors
            .groupby(
                ["actual", "predicted"]
            )
            .size()
            .reset_index(name="count")
            .sort_values(
                "count",
                ascending=False,
            )
        )

        print("\nТипы ошибок:")
        print(
            error_types.to_string(
                index=False
            )
        )

    # ==========================================
    # 8. Сохраняем ошибки
    # ==========================================

    output_path = (
        PROJECT_ROOT
        / "reports"
        / "high_confidence_errors.csv"
    )

    errors.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nОшибки сохранены:\n{output_path}"
    )


if __name__ == "__main__":
    main()