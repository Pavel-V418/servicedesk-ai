from __future__ import annotations

from src.paths import DEFAULT_XLSX

from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.features.feature_builder import prepare_features


TARGET = "Кем решен (группа)"
DATE_COL = "Дата регистрации"
CLASSES = ["L1", "L2", "L3"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_PATH = Path(
    str(DEFAULT_XLSX)
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "route_predictor.joblib"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "final_evaluation.txt"
)


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

    # 1. Загружаем данные
    df = pd.read_excel(DATA_PATH)

    df[TARGET] = normalize_target(df[TARGET])

    # Оставляем только L1/L2/L3
    df = df[
        df[TARGET].isin(CLASSES)
    ].copy()

    # 2. Сортируем по времени
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

    # 3. Тот же temporal split 80/20
    split_index = int(len(df) * 0.8)

    development_df = df.iloc[:split_index].copy()
    final_test_df = df.iloc[split_index:].copy()

    print(f"Development: {len(development_df)}")
    print(f"Final test:  {len(final_test_df)}")

    # 4. Подготавливаем FINAL TEST
    X_test = prepare_features(final_test_df)
    y_test = final_test_df[TARGET]

    print("\nFinal test classes:")
    print(y_test.value_counts())

    # 5. Загружаем уже обученную модель
    model = joblib.load(MODEL_PATH)

    # 6. Предсказания
    predictions = model.predict(X_test)

    # 7. Метрики
    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        labels=CLASSES,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_test,
        predictions,
        labels=CLASSES,
        average="weighted",
        zero_division=0,
    )

    report = classification_report(
        y_test,
        predictions,
        labels=CLASSES,
        digits=4,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=CLASSES,
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=[f"Actual_{c}" for c in CLASSES],
        columns=[f"Predicted_{c}" for c in CLASSES],
    )

    # 8. Вывод
    print("\n" + "=" * 70)
    print("FINAL TEMPORAL TEST")
    print("=" * 70)

    print(f"\nAccuracy:    {accuracy:.4f}")
    print(f"Macro-F1:    {macro_f1:.4f}")
    print(f"Weighted-F1: {weighted_f1:.4f}")

    print("\nClassification report:")
    print(report)

    print("Confusion matrix:")
    print(matrix_df)

    # 9. Сохраняем отчёт
    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_text = (
        "FINAL TEMPORAL TEST\n"
        "===================\n\n"
        f"Development: {len(development_df)}\n"
        f"Final test: {len(final_test_df)}\n\n"
        f"Accuracy: {accuracy:.4f}\n"
        f"Macro-F1: {macro_f1:.4f}\n"
        f"Weighted-F1: {weighted_f1:.4f}\n\n"
        "Classification report:\n"
        f"{report}\n"
        "Confusion matrix:\n"
        f"{matrix_df.to_string()}\n"
    )

    REPORT_PATH.write_text(
        report_text,
        encoding="utf-8",
    )

    print(
        f"\nОтчёт сохранён:\n{REPORT_PATH}"
    )


if __name__ == "__main__":
    main()