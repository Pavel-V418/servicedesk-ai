from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

ROUTING_AGENT_DIR = Path(__file__).resolve().parent / "routing_agent"
if str(ROUTING_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(ROUTING_AGENT_DIR))

from src.agent.routing_agent import RoutingAgent  # noqa: E402
from src.features.feature_builder import prepare_features  # noqa: E402
from src.paths import ROUTE_PREDICTOR_PATH  # noqa: E402

LINE_NAMES = {"L1": "1 линия", "L2": "2 линия", "L3": "3 линия"}
STATUS_NAMES = {
    "high_confidence": "высокая уверенность",
    "medium_confidence": "средняя уверенность",
    "manual_review": "нужен ручной разбор",
}


class TicketRouter:
    """Рекомендация линии поддержки (L1/L2/L3)."""

    def __init__(self, model_path: Optional[Path] = None):
        model_path = Path(model_path or ROUTE_PREDICTOR_PATH)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Не найдена модель маршрутизации {model_path}. "
                "Обучи её: python main.py train  (или из routing/routing_agent: "
                "python -m src.models.train_router)."
            )
        try:
            model = joblib.load(model_path)
        except (AttributeError, ModuleNotFoundError, ImportError) as exc:
            # Типичный случай: joblib сохранён другой версией scikit-learn.
            raise RuntimeError(
                f"Модель {model_path} не открывается в текущей версии scikit-learn ({exc}). "
                "Переобучи её в этом окружении: python main.py train --only routing"
            ) from exc
        self.agent = RoutingAgent(model)

    def route(self, ticket: dict) -> dict:
        """ticket — словарь с колонками как в выгрузке (Описание 2, Услуга, ...)."""
        x_one = prepare_features(pd.DataFrame([ticket]))
        return self.agent.route(x_one)


# В выгрузке приоритет записан как "(3) Средний". Если пришло "Средний" или "3",
# OneHotEncoder маршрутизатора не узнаёт значение и молча его игнорирует.
PRIORITY_CANON = {
    "наивысший": "(1) Наивысший", "1": "(1) Наивысший",
    "высокий": "(2) Высокий", "2": "(2) Высокий",
    "средний": "(3) Средний", "3": "(3) Средний",
    "низкий": "(4) Низкий", "4": "(4) Низкий",
}


def normalize_priority(value):
    if not value:
        return value
    return PRIORITY_CANON.get(str(value).strip().lower(), value)


def build_routing_ticket(state: dict) -> tuple[dict, str]:
    """Собирает строку признаков маршрутизатора из состояния пайплайна."""
    request_kind = state.get("request_kind") or ""
    source = "оператор"
    if not request_kind and state.get("predicted_class"):
        request_kind = state["predicted_class"]
        source = "классификатор"
    ticket = {
        "Описание 2": state.get("raw_text", ""),
        "Пользователь": state.get("user", ""),
        "Услуга": state.get("service", ""),
        "Компонент услуги 1 уровня": state.get("component", ""),
        "Вид запроса": request_kind,
        "Тип запроса": state.get("request_type", ""),
        "Критичность": state.get("criticality", ""),
        "Срочность": state.get("urgency", ""),
        "Приоритет": normalize_priority(state.get("priority", "")),
        "Класс обслуживания": state.get("service_class", ""),
        "Часовой пояс запроса": state.get("timezone", ""),
    }
    # Пустая строка на обучении была NaN -> "__MISSING__"; приводим так же.
    ticket = {k: (v if v not in ("", None) else None) for k, v in ticket.items()}
    return ticket, source
