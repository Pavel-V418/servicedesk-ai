from __future__ import annotations

from src.paths import DEFAULT_XLSX

from pathlib import Path

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold

from src.models.catboost_model import build_catboost


TARGET = "Кем решен (группа)"

EXCEL_PATH = Path(
    str(DEFAULT_XLSX)
)

CLASSES = ["L1", "L2", "L3"]


# ------------------------------------------------------------
# Признаки для CatBoost
# ------------------------------------------------------------

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


def load_data() -> pd.DataFrame:
    if not EXCEL_PATH.exists():
        raise FileNotFoundError(
            f"Excel-файл не найден: {EXCEL_PATH}"
        )

    df = pd.read_excel(EXCEL_PATH)

    df[TARGET] = normalize_target(df[TARGET])

    # Как и раньше, L4 пока не обучаем.
    df = df[
        df[TARGET].isin(CLASSES)
    ].copy()

    return df


def prepare_catboost_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    missing = [
        col
        for col in CATEGORICAL_COLS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"В датасете отсутствуют колонки: {missing}"
        )

    X = df[CATEGORICAL_COLS].copy()

    # CatBoost должен получить нормальные строки,
    # а не NaN.
    for col in CATEGORICAL_COLS:
        X[col] = (
            X[col]
            .fillna("__MISSING__")
            .astype(str)
        )

    return X


def main() -> None:

    # --------------------------------------------------------
    # Загружаем данные
    # --------------------------------------------------------

    df = load_data()

    DATE_COL = "Дата регистрации"

    df[DATE_COL] = pd.to_datetime(
        df[DATE_COL],
        errors="coerce",
    )

    df = df.dropna(subset=[DATE_COL])

    df = df.sort_values(DATE_COL).reset_index(drop=True)

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

    # CatBoost CV работает только на development
    X = prepare_catboost_features(development_df)
    y = development_df[TARGET]

    print(f"Количество обращений: {len(df)}")

    print("\nРаспределение:")
    print(y.value_counts())

    # Все наши признаки сейчас категориальные.
    cat_features = list(range(len(CATEGORICAL_COLS)))

    # Используем абсолютно такое же CV,
    # как у baseline.
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    results = []

    # --------------------------------------------------------
    # Cross-validation
    # --------------------------------------------------------

    for fold, (train_idx, val_idx) in enumerate(
        cv.split(X, y),
        start=1,
    ):

        print("\n" + "=" * 60)
        print(f"FOLD {fold}")
        print("=" * 60)

        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]

        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        model = build_catboost()

        model.fit(
            X_train,
            y_train,
            cat_features=cat_features,
        )

        pred = model.predict(X_val)

        # В некоторых версиях CatBoost predict()
        # может вернуть массив формы (n, 1).
        pred = pd.Series(
            pred.reshape(-1),
            index=y_val.index,
        )

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

        recalls = recall_score(
            y_val,
            pred,
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

    # --------------------------------------------------------
    # Итоги
    # --------------------------------------------------------

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 60)
    print("CATBOOST — ИТОГИ CROSS-VALIDATION")
    print("=" * 60)

    print(
        results_df.round(4).to_string(index=False)
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

    # --------------------------------------------------------
    # Сохраняем
    # --------------------------------------------------------

    reports_dir = Path("reports")

    reports_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        reports_dir
        / "catboost_cross_validation.csv"
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