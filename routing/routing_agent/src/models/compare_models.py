from __future__ import annotations

from src.paths import DEFAULT_XLSX

from pathlib import Path
from sklearn.linear_model import LogisticRegression
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold

from src.features.feature_builder import prepare_features
from src.models.baseline import build_baseline
from src.models.weighted_ovr import WeightedOneVsRestLogReg


TARGET = "Кем решен (группа)"
DATE_COL = "Дата регистрации"

EXCEL_PATH = Path(
    str(DEFAULT_XLSX)
)

CLASSES = ["L1", "L2", "L3"]


# ============================================================
# TARGET
# ============================================================

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


# ============================================================
# DATA
# ============================================================

def load_development_data():
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

    split_index = int(len(df) * 0.8)

    development_df = df.iloc[:split_index].copy()
    final_test_df = df.iloc[split_index:].copy()

    print("Temporal split:")
    print(f"Development: {len(development_df)}")
    print(f"Final test:  {len(final_test_df)}")

    print("\nDevelopment classes:")
    print(development_df[TARGET].value_counts())

    print("\nFinal test classes:")
    print(final_test_df[TARGET].value_counts())

    return development_df


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_CONFIGS = {
    "Baseline": None,

    "Weighted_1": {
        "L1": 1.0,
        "L2": 1.2,
        "L3": 1.3,
    },

    "Weighted_2": {
        "L1": 1.0,
        "L2": 1.3,
        "L3": 1.5,
    },
}


def set_class_weight(model, class_weight):
    """
    Настраивает class_weight.

    Для baseline с OneVsRestClassifier оставляем None.
    Взвешенные модели будем создавать отдельно.
    """
    if class_weight is None:
        return model

    raise ValueError(
        "Custom class weights нельзя напрямую использовать "
        "в текущем OneVsRestClassifier."
    )

def build_weighted_model(class_weight):
    """
    Берём тот же FeatureUnion из baseline,
    но заменяем OneVsRestClassifier на обычную
    LogisticRegression.

    Это позволяет задавать веса непосредственно
    для L1, L2 и L3.
    """

    baseline = build_baseline()

    features = baseline.named_steps["features"]

    return type(baseline)([
        ("features", features),
        # sklearn >= 1.8: liblinear не решает мультикласс сам — см. weighted_ovr.py
        ("model", WeightedOneVsRestLogReg(
            max_iter=1200,
            class_weight=class_weight if isinstance(class_weight, dict) else None,
        )),
    ])
# ============================================================
# CROSS-VALIDATION
# ============================================================

def evaluate_model(
    model_name,
    class_weight,
    X,
    y,
    folds,
):

    results = []

    print("\n" + "=" * 70)
    print(f"MODEL: {model_name}")
    print(f"class_weight = {class_weight}")
    print("=" * 70)

    for fold, (train_idx, val_idx) in enumerate(
        folds,
        start=1,
    ):

        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]

        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        if class_weight is None:
            # Наш оригинальный baseline оставляем неизменным.
            model = build_baseline()
        else:
            # Weighted-варианты используют multiclass LR.
            model = build_weighted_model(
                class_weight
            )

        model.fit(
            X_train,
            y_train,
        )

        predictions = model.predict(X_val)

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
            "model": model_name,
            "fold": fold,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "L1_recall": recalls[0],
            "L2_recall": recalls[1],
            "L3_recall": recalls[2],
        })

        print(
            f"Fold {fold}: "
            f"Macro-F1={macro_f1:.4f} | "
            f"L1={recalls[0]:.4f} | "
            f"L2={recalls[1]:.4f} | "
            f"L3={recalls[2]:.4f}"
        )

    return results


# ============================================================
# MAIN
# ============================================================

def main():

    development_df = load_development_data()

    X = prepare_features(
        development_df
    )

    y = development_df[TARGET]

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    # ВАЖНО:
    # создаём folds один раз.
    # Все модели получают абсолютно одинаковые разбиения.
    folds = list(
        cv.split(X, y)
    )

    all_results = []

    for model_name, class_weight in MODEL_CONFIGS.items():

        model_results = evaluate_model(
            model_name=model_name,
            class_weight=class_weight,
            X=X,
            y=y,
            folds=folds,
        )

        all_results.extend(
            model_results
        )

    results_df = pd.DataFrame(
        all_results
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = (
        results_df
        .groupby("model")
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),

            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),

            weighted_f1_mean=("weighted_f1", "mean"),

            L1_recall=("L1_recall", "mean"),
            L2_recall=("L2_recall", "mean"),
            L3_recall=("L3_recall", "mean"),
        )
        .reset_index()
    )

    summary = summary.sort_values(
        "macro_f1_mean",
        ascending=False,
    )

    print("\n")
    print("=" * 90)
    print("MODEL COMPARISON")
    print("=" * 90)

    print(
        summary.round(4).to_string(
            index=False
        )
    )

    # ========================================================
    # SAVE
    # ========================================================

    reports_dir = Path("reports")

    reports_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        reports_dir / "model_comparison_folds.csv",
        index=False,
    )

    summary.to_csv(
        reports_dir / "model_comparison_summary.csv",
        index=False,
    )

    print(
        "\nРезультаты сохранены в reports/"
    )


if __name__ == "__main__":
    main()