from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

INPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "confidence_analysis.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "threshold_analysis.csv"
)


def main():

    # ==========================================
    # 1. Загружаем OOF predictions
    # ==========================================

    df = pd.read_csv(INPUT_PATH)

    print(f"Количество обращений: {len(df)}")

    # ==========================================
    # 2. Проверяем разные thresholds
    # ==========================================

    thresholds = [
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.95,
    ]

    rows = []

    for threshold in thresholds:

        # Автоматически маршрутизируем только
        # достаточно уверенные predictions
        automatic = df["confidence"] >= threshold

        manual = ~automatic

        automatic_count = automatic.sum()
        manual_count = manual.sum()

        automatic_rate = automatic.mean()
        manual_rate = manual.mean()

        if automatic_count > 0:
            automatic_accuracy = (
                df.loc[automatic, "correct"].mean()
            )

            errors = (
                ~df.loc[automatic, "correct"]
            ).sum()

        else:
            automatic_accuracy = 0.0
            errors = 0

        rows.append({
            "threshold": threshold,
            "automatic_count": automatic_count,
            "manual_count": manual_count,
            "automatic_rate": automatic_rate,
            "manual_rate": manual_rate,
            "automatic_accuracy": automatic_accuracy,
            "automatic_errors": errors,
        })

    results = pd.DataFrame(rows)

    # ==========================================
    # 3. Красивый вывод
    # ==========================================

    print("\n" + "=" * 90)
    print("THRESHOLD ANALYSIS")
    print("=" * 90)

    for _, row in results.iterrows():

        print(
            f"Threshold {row['threshold']:.2f} | "
            f"Auto: {row['automatic_rate']:.2%} | "
            f"Manual: {row['manual_rate']:.2%} | "
            f"Auto accuracy: "
            f"{row['automatic_accuracy']:.2%} | "
            f"Errors: {int(row['automatic_errors'])}"
        )

    # ==========================================
    # 4. Проверяем margin
    # ==========================================

    print("\n" + "=" * 90)
    print("MARGIN ANALYSIS")
    print("=" * 90)

    # Сначала фиксируем текущий confidence threshold
    confidence_threshold = 0.60

    for margin_threshold in [
        0.00,
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
    ]:

        automatic = (
            (df["confidence"] >= confidence_threshold)
            &
            (df["margin"] >= margin_threshold)
        )

        automatic_count = automatic.sum()

        if automatic_count > 0:

            accuracy = (
                df.loc[
                    automatic,
                    "correct"
                ].mean()
            )

        else:
            accuracy = 0.0

        print(
            f"Margin >= {margin_threshold:.2f} | "
            f"Auto: {automatic.mean():.2%} | "
            f"Accuracy: {accuracy:.2%}"
        )

    # ==========================================
    # 5. Сохраняем
    # ==========================================

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"\nРезультаты сохранены:\n{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
