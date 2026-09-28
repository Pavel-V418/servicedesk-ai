"""
Пути для routing-agent внутри объединённого проекта ServiceDeskAI.

Раньше в скриптах был захардкожен путь /Users/armancho/Downloads/Обращения_1931.xlsx.
Теперь датасет общий для всего проекта и лежит в classifier/dataset/.
Переопределить можно переменной окружения ROUTING_XLSX_PATH.
"""
from __future__ import annotations

import os
from pathlib import Path

ROUTING_ROOT = Path(__file__).resolve().parent.parent          # routing/routing_agent
PROJECT_ROOT = ROUTING_ROOT.parent.parent                       # ServiceDeskAI

DEFAULT_XLSX = Path(
    os.environ.get(
        "ROUTING_XLSX_PATH",
        PROJECT_ROOT / "classifier" / "dataset" / "Обращения_1931.xlsx",
    )
)

MODELS_DIR = ROUTING_ROOT / "models"
ROUTE_PREDICTOR_PATH = MODELS_DIR / "route_predictor.joblib"
