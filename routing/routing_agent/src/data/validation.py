from __future__ import annotations

FORBIDDEN_FEATURES = {
    "Фактическое время выполнения",
    "Фактическая длительность выполнения запроса (SLA)",
    "Просрочен?*",
    "Дата последнего изменения",
    "Суммарное время реакции 1 линии",
    "Суммарное время работы 1 линии",
    "Суммарное время реакции 2 линии",
    "Суммарное время работы 2 линии",
    "Суммарное время реакции 3 линии",
    "Суммарное время работы 3 линии",
    "Суммарное время реакции 4 линии",
    "Суммарное время работы 4 линии",
    "Суммарное время уточнений",
    "Результат работ",
    "Количество уточнений",
    "Кем решен (группа)",
}


def assert_no_leakage(feature_names: list[str]) -> None:
    leaked = sorted(set(feature_names) & FORBIDDEN_FEATURES)
    if leaked:
        raise ValueError(f"Data leakage: forbidden features were selected: {leaked}")
