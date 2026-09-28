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
    / "l2_to_l1_errors.csv"
)

HIGH_THRESHOLD = 0.80


def main():
    df = pd.read_csv(INPUT_PATH)

    # Ищем главную проблему Router:
    # настоящий класс L2, а модель уверенно предсказала L1.
    errors = df[
        (df["actual"] == "L2")
        & (df["predicted"] == "L1")
        & (df["confidence"] >= HIGH_THRESHOLD)
    ].copy()

    errors = errors.sort_values(
        "confidence",
        ascending=False,
    )

    print("=" * 80)
    print("HIGH-CONFIDENCE L2 -> L1 ERRORS")
    print("=" * 80)

    print(f"\nКоличество ошибок: {len(errors)}")

    # ------------------------------------------------
    # Услуга
    # ------------------------------------------------

    print("\n" + "=" * 80)
    print("УСЛУГА")
    print("=" * 80)

    print(
        errors["Услуга"]
        .value_counts(dropna=False)
        .to_string()
    )

    # ------------------------------------------------
    # Вид запроса
    # ------------------------------------------------

    print("\n" + "=" * 80)
    print("ВИД ЗАПРОСА")
    print("=" * 80)

    print(
        errors["Вид запроса"]
        .value_counts(dropna=False)
        .to_string()
    )

    # ------------------------------------------------
    # Компонент услуги
    # ------------------------------------------------

    print("\n" + "=" * 80)
    print("КОМПОНЕНТ УСЛУГИ")
    print("=" * 80)

    print(
        errors["Компонент услуги 1 уровня"]
        .value_counts(dropna=False)
        .to_string()
    )

    # ------------------------------------------------
    # Самые уверенные ошибки
    # ------------------------------------------------

    print("\n" + "=" * 80)
    print("TOP-15 САМЫХ УВЕРЕННЫХ ОШИБОК")
    print("=" * 80)

    columns = [
        "Описание 2",
        "Услуга",
        "Компонент услуги 1 уровня",
        "Вид запроса",
        "actual",
        "predicted",
        "confidence",
        "L1_probability",
        "L2_probability",
        "L3_probability",
    ]

    print(
        errors[columns]
        .head(15)
        .to_string(index=False)
    )

    # ------------------------------------------------
    # Сохраняем результат
    # ------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    errors.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"\nВсе ошибки сохранены:\n{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()