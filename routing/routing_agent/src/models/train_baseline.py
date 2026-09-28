from __future__ import annotations

from src.paths import DEFAULT_XLSX

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)

from src.data.split import temporal_split
from src.features.feature_builder import prepare_features
from src.models.baseline import build_baseline


TARGET = "Кем решен (группа)"
DATE_COL = "Дата регистрации"


def normalize_target(series: pd.Series) -> pd.Series:
    """
    Преобразует названия линий из Excel:
    '(1 линия)' -> 'L1'
    '(2 линия)' -> 'L2'
    и т.д.
    """
    s = series.fillna("").astype(str).str.strip()

    mapping = {
        "(1 линия)": "L1",
        "(2 линия)": "L2",
        "(3 линия)": "L3",
        "(4 линия)": "L4",
    }

    return s.map(mapping).fillna(s)


def main(path: str) -> None:
    # ---------------------------------------------------------
    # 1. Проверяем существование Excel-файла
    # ---------------------------------------------------------
    excel_path = Path(path)

    if not excel_path.exists():
        raise FileNotFoundError(
            f"Excel-файл не найден: {excel_path.resolve()}"
        )

    print(f"Загружаем данные из: {excel_path.resolve()}")

    # ---------------------------------------------------------
    # 2. Загружаем датасет
    # ---------------------------------------------------------
    df = pd.read_excel(excel_path)

    print(f"Загружено обращений: {len(df)}")

    # Проверяем наличие необходимых колонок
    required_columns = {TARGET, DATE_COL}

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"В Excel отсутствуют необходимые колонки: "
            f"{sorted(missing_columns)}"
        )

    # ---------------------------------------------------------
    # 3. Приводим target к L1 / L2 / L3 / L4
    # ---------------------------------------------------------
    df[TARGET] = normalize_target(df[TARGET])

    print("\nРаспределение линий:")
    print(df[TARGET].value_counts())

    # ---------------------------------------------------------
    # 4. Пока исключаем L4
    # ---------------------------------------------------------
    # В датасете слишком мало примеров четвёртой линии,
    # поэтому v0 обучаем только на L1/L2/L3.
    df = df[df[TARGET].isin(["L1", "L2", "L3"])].copy()

    print(f"\nПосле удаления L4 осталось: {len(df)} обращений")

    # ---------------------------------------------------------
    # 5. Делим данные по времени
    # ---------------------------------------------------------
    train, val, test = temporal_split(df, DATE_COL)

    print("\nРазмеры выборок:")
    print(f"Train:      {len(train)}")
    print(f"Validation: {len(val)}")
    print(f"Test:       {len(test)}")

    # ---------------------------------------------------------
    # 6. Подготавливаем признаки
    # ---------------------------------------------------------
    X_train = prepare_features(train)
    y_train = train[TARGET]

    X_val = prepare_features(val)
    y_val = val[TARGET]

    X_test = prepare_features(test)
    y_test = test[TARGET]

    # ---------------------------------------------------------
    # 7. Создаём модель
    # ---------------------------------------------------------
    print("\nСоздаём baseline-модель...")

    model = build_baseline()

    # ---------------------------------------------------------
    # 8. Обучаем
    # ---------------------------------------------------------
    print("Обучение модели...")

    model.fit(X_train, y_train)

    print("Обучение завершено.")

    # ---------------------------------------------------------
    # 9. Сохраняем модель
    # ---------------------------------------------------------
    out_dir = Path("models")
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "router_baseline.joblib"

    joblib.dump(model, model_path)

    print(f"\nМодель сохранена: {model_path}")

    # ---------------------------------------------------------
    # 10. Проверяем качество
    # ---------------------------------------------------------
    report = {}

    datasets = [
        ("validation", X_val, y_val),
        ("test", X_test, y_test),
    ]

    for name, X, y in datasets:
        pred = model.predict(X)

        report[name] = {
            "accuracy": float(
                accuracy_score(y, pred)
            ),
            "macro_f1": float(
                f1_score(
                    y,
                    pred,
                    average="macro",
                    zero_division=0,
                )
            ),
            "weighted_f1": float(
                f1_score(
                    y,
                    pred,
                    average="weighted",
                    zero_division=0,
                )
            ),
            "classification_report": classification_report(
                y,
                pred,
                labels=["L1", "L2", "L3"],
                output_dict=True,
                zero_division=0,
            ),
            "confusion_matrix": confusion_matrix(
                y,
                pred,
                labels=["L1", "L2", "L3"],
            ).tolist(),
        }

        print(f"\n{'=' * 50}")
        print(f"{name.upper()}")
        print("=" * 50)

        print(
            classification_report(
                y,
                pred,
                labels=["L1", "L2", "L3"],
                zero_division=0,
            )
        )

        print("Confusion matrix:")
        print(
            confusion_matrix(
                y,
                pred,
                labels=["L1", "L2", "L3"],
            )
        )

    # ---------------------------------------------------------
    # 11. Сохраняем метрики
    # ---------------------------------------------------------
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = reports_dir / "baseline_metrics.json"

    metrics_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nМетрики сохранены: {metrics_path}")

    print("\nГотово!")


if __name__ == "__main__":
    excel_path = str(DEFAULT_XLSX)
    main(excel_path)