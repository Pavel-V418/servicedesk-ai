from __future__ import annotations
import pandas as pd


def temporal_split(df: pd.DataFrame, date_col: str = "Дата регистрации", train_ratio: float = 0.6, val_ratio: float = 0.2):
    data = df.copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data = data.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = data.iloc[:train_end].copy()
    val = data.iloc[train_end:val_end].copy()
    test = data.iloc[val_end:].copy()
    return train, val, test
