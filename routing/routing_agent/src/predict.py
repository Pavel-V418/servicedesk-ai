from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from src.agent.routing_agent import RoutingAgent
from src.features.feature_builder import prepare_features




PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "route_predictor.joblib"
)

def main():

    # ==========================================
    # 1. Загружаем обученную модель
    # ==========================================

    model = joblib.load(MODEL_PATH)

    # ==========================================
    # 2. Создаём RoutingAgent
    # ==========================================

    agent = RoutingAgent(model)

    # ==========================================
    # 3. Тестовое обращение
    # ==========================================

    ticket = {
        "Описание 2": "Не загружается список отправлений",

        "Пользователь": "test_user",

        "Услуга": "Партионный прием",

        "Компонент услуги 1 уровня": "Препост",

        "Вид запроса": "Импорт списков из ЛК ЮЛ",

        "Тип запроса": "Инцидент",

        "Критичность": "Средняя",

        "Срочность": "Средняя",

        "Приоритет": "Средний",

        "Класс обслуживания": "Стандартный",

        "Часовой пояс запроса": "Москва",
    }

    # ==========================================
    # 4. Предсказание
    # ==========================================

    ticket_df = pd.DataFrame([ticket])

    x_one = prepare_features(ticket_df)

    result = agent.route(x_one)

    # ==========================================
    # 5. Результат
    # ==========================================

    print("\n" + "=" * 60)
    print("ROUTING AGENT OUTPUT")
    print("=" * 60)

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()