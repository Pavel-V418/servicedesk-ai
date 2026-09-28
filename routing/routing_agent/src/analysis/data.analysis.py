from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# НАСТРОЙКИ
# ============================================================

TARGET = "Кем решен (группа)"
DATE_COL = "Дата регистрации"

EXCEL_PATH = Path(
    "/Users/armancho/Downloads/Обращения_1931.xlsx"
)


# ============================================================
# НОРМАЛИЗАЦИЯ TARGET
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
# ЗАГРУЗКА
# ============================================================

def load_data() -> pd.DataFrame:
    if not EXCEL_PATH.exists():
        raise FileNotFoundError(
            f"Excel не найден: {EXCEL_PATH}"
        )

    df = pd.read_excel(EXCEL_PATH)

    df[TARGET] = normalize_target(df[TARGET])

    df[DATE_COL] = pd.to_datetime(
        df[DATE_COL],
        errors="coerce",
    )

    # Удаляем записи без даты
    df = df.dropna(subset=[DATE_COL])

    # Пока анализируем только классы,
    # которые использует наша ML-модель
    df = df[
        df[TARGET].isin(["L1", "L2", "L3"])
    ].copy()

    return df


# ============================================================
# 1. ОБЩЕЕ РАСПРЕДЕЛЕНИЕ
# ============================================================

def analyze_target_distribution(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("ОБЩЕЕ РАСПРЕДЕЛЕНИЕ ЛИНИЙ")
    print("=" * 60)

    counts = df[TARGET].value_counts()

    print("\nКоличество:")
    print(counts)

    percentages = (
        df[TARGET]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    print("\nПроценты:")
    print(percentages)


# ============================================================
# 2. РАСПРЕДЕЛЕНИЕ ПО МЕСЯЦАМ
# ============================================================

def analyze_months(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("РАСПРЕДЕЛЕНИЕ ПО МЕСЯЦАМ")
    print("=" * 60)

    data = df.copy()

    data["month"] = (
        data[DATE_COL]
        .dt
        .to_period("M")
        .astype(str)
    )

    monthly = pd.crosstab(
        data["month"],
        data[TARGET],
    )

    # Гарантируем одинаковый порядок колонок
    monthly = monthly.reindex(
        columns=["L1", "L2", "L3"],
        fill_value=0,
    )

    print("\nКоличество обращений:")
    print(monthly)

    monthly_percent = (
        monthly
        .div(monthly.sum(axis=1), axis=0)
        .mul(100)
        .round(1)
    )

    print("\nПроцент внутри каждого месяца:")
    print(monthly_percent)

    return monthly


# ============================================================
# 3. УСЛУГИ
# ============================================================

def analyze_services(df: pd.DataFrame) -> None:
    column = "Услуга"

    if column not in df.columns:
        print(f"\nКолонка '{column}' отсутствует.")
        return

    print("\n" + "=" * 60)
    print("ТОП УСЛУГ")
    print("=" * 60)

    services = pd.crosstab(
        df[column].fillna("__MISSING__"),
        df[TARGET],
    )

    services = services.reindex(
        columns=["L1", "L2", "L3"],
        fill_value=0,
    )

    services["TOTAL"] = services.sum(axis=1)

    services = services.sort_values(
        "TOTAL",
        ascending=False,
    )

    print("\nТоп-15 услуг:")
    print(services.head(15).to_string())


# ============================================================
# 4. ВИД ЗАПРОСА
# ============================================================

def analyze_request_types(df: pd.DataFrame) -> None:
    column = "Вид запроса"

    if column not in df.columns:
        print(f"\nКолонка '{column}' отсутствует.")
        return

    print("\n" + "=" * 60)
    print("ВИДЫ ЗАПРОСОВ")
    print("=" * 60)

    request_types = pd.crosstab(
        df[column].fillna("__MISSING__"),
        df[TARGET],
    )

    request_types = request_types.reindex(
        columns=["L1", "L2", "L3"],
        fill_value=0,
    )

    request_types["TOTAL"] = (
        request_types.sum(axis=1)
    )

    request_types = request_types.sort_values(
        "TOTAL",
        ascending=False,
    )

    print("\nТоп-20:")
    print(request_types.head(20).to_string())


# ============================================================
# 5. ГРАФИК
# ============================================================

def plot_monthly_distribution(
    monthly: pd.DataFrame,
) -> None:

    monthly.plot(
        kind="bar",
        figsize=(12, 6),
    )

    plt.title(
        "Распределение линий поддержки по месяцам"
    )

    plt.xlabel("Месяц")
    plt.ylabel("Количество обращений")

    plt.xticks(rotation=45)

    plt.legend(
        title="Линия"
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# 6. ДОЛЯ КАЖДОЙ ЛИНИИ ПО МЕСЯЦАМ
# ============================================================

def plot_monthly_percent(
    monthly: pd.DataFrame,
) -> None:

    monthly_percent = (
        monthly
        .div(monthly.sum(axis=1), axis=0)
        .mul(100)
    )

    monthly_percent.plot(
        kind="line",
        marker="o",
        figsize=(12, 6),
    )

    plt.title(
        "Изменение доли линий поддержки со временем"
    )

    plt.xlabel("Месяц")
    plt.ylabel("Доля обращений, %")

    plt.xticks(rotation=45)

    plt.legend(
        title="Линия"
    )

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.show()

def analyze_date_range(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("АНАЛИЗ ДАТ")
    print("=" * 60)

    print(f"Минимальная дата: {df[DATE_COL].min()}")
    print(f"Максимальная дата: {df[DATE_COL].max()}")

    print("\nКоличество обращений по дням:")

    daily = (
        df[DATE_COL]
        .dt.date
        .value_counts()
        .sort_index()
    )

    print(daily.to_string())

# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("Загрузка данных...")

    df = load_data()

    print(
        f"Загружено {len(df)} обращений."
    )

    # 1
    analyze_target_distribution(df)
    analyze_date_range(df)
    # 2
    monthly = analyze_months(df)

    # 3
    analyze_services(df)

    # 4
    analyze_request_types(df)

    # 5
    plot_monthly_distribution(monthly)

    # 6
    plot_monthly_percent(monthly)


if __name__ == "__main__":
    main()