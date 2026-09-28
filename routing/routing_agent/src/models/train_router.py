from __future__ import annotations

from src.paths import DEFAULT_XLSX

from pathlib import Path

import joblib
import pandas as pd

from src.features.feature_builder import prepare_features
from src.models.route_predictor import build_route_predictor


TARGET = "Кем решен (группа)"
DATE_COL = "Дата регистрации"

EXCEL_PATH = Path(
    str(DEFAULT_XLSX)
)

# Раньше модель сохранялась в текущую папку (cwd) — теперь сразу туда,
# откуда её грузит RoutingAgent / объединённый пайплайн.
from src.paths import ROUTE_PREDICTOR_PATH
MODEL_PATH = ROUTE_PREDICTOR_PATH

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

    # ==========================================
    # 1. Загружаем данные
    # ==========================================

    df = pd.read_excel(EXCEL_PATH)

    df[TARGET] = normalize_target(
        df[TARGET]
    )

    df = df[
        df[TARGET].isin(CLASSES)
    ].copy()

    df[DATE_COL] = pd.to_datetime(
        df[DATE_COL],
        errors="coerce",
    )

    df = df.dropna(
        subset=[DATE_COL]
    )

    df = (
        df
        .sort_values(DATE_COL)
        .reset_index(drop=True)
    )

    # ==========================================
    # 2. Temporal holdout
    # ==========================================

    split_index = int(
        len(df) * 0.8
    )

    development_df = (
        df.iloc[:split_index].copy()
    )

    final_test_df = (
        df.iloc[split_index:].copy()
    )

    print("Temporal split:")
    print(
        f"Development: {len(development_df)}"
    )
    print(
        f"Final test:  {len(final_test_df)}"
    )

    # ==========================================
    # 3. Features
    # ==========================================

    X_train = prepare_features(
        development_df
    )

    y_train = development_df[TARGET]

    print("\nTraining classes:")
    print(
        y_train.value_counts()
    )

    # ==========================================
    # 4. Создаём RoutePredictor
    # ==========================================

    model = build_route_predictor()

    print("\nОбучение RoutePredictor...")

    model.fit(
        X_train,
        y_train,
    )

    # ==========================================
    # 5. Сохраняем
    # ==========================================

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    print("\nМодель успешно обучена.")

    print(
        f"Сохранена в: {MODEL_PATH}"
    )

    print(
        f"Классы модели: {model.classes_}"
    )

    print(
        "\nFinal temporal test НЕ использовался."
    )


if __name__ == "__main__":
    main()