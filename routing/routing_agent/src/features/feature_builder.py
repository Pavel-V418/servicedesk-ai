from __future__ import annotations
import pandas as pd
from src.data.validation import assert_no_leakage

TEXT_COL = "Описание 2"
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


def prepare_features(df: pd.DataFrame):
    feature_names = [TEXT_COL, *CATEGORICAL_COLS]
    assert_no_leakage(feature_names)

    x = df[feature_names].copy()
    x[TEXT_COL] = x[TEXT_COL].fillna("").astype(str)
    for col in CATEGORICAL_COLS:
        x[col] = x[col].fillna("__MISSING__").astype(str)
    return x
