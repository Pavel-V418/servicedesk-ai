"""
Общий пайплайн Service Desk AI — граф LangGraph, который связывает блоки команды.

    classify              ->  route      ->  similar_tickets  ->  sla        ->  synthesize
    (classifier/, Павел)      (routing/)     (RagAgent/)          (sla/risk)     (finalizer.py: GigaChat / шаблон)

Раньше этот граф жил в agent.py вместе с кодом классификатора. Теперь agent.py
разделён: вся ML-часть классификатора — в classifier/predictor.py, а здесь
только оркестрация. Каждый блок подключается как отдельный компонент:
если его модель/база ещё не готова, узел пропускается с понятной причиной
(COMPONENT_STATUS), а не роняет весь пайплайн.

Точки входа:
    classify_new_ticket(raw_text, service=..., component=..., request_type=..., ...)
    process_ticket_row(row)  # словарь с колонками как в выгрузке Excel
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple, TypedDict

import config
import finalizer

try:
    from langgraph.graph import StateGraph, START, END
    _HAS_LANGGRAPH = True
except ImportError:  # граф можно прогнать и без langgraph — последовательно
    _HAS_LANGGRAPH = False


class TicketState(TypedDict, total=False):
    # --- вход: поля, известные в момент регистрации обращения ---
    raw_text: str
    service: str
    component: str
    request_type: str
    request_kind: str          # "Вид запроса", если оператор уже указал
    user: str
    criticality: str
    urgency: str
    priority: str
    service_class: str
    timezone: str
    # --- классификатор ---
    clean_text: str
    predicted_class: Optional[str]
    confidence: float
    is_confident: bool
    classifier_threshold: float
    top_k: List[Tuple[str, float]]
    explanation_words: List[Tuple[str, float]]
    # --- маршрутизация ---
    recommended_line: Optional[str]
    routing_confidence: Optional[float]
    routing_status: Optional[str]
    routing_probabilities: dict
    routing_reason: Optional[str]
    request_kind_source: Optional[str]
    # --- RAG ---
    similar_tickets: List[dict]
    # --- SLA ---
    sla_risk_level: Optional[str]
    sla_risk_score: Optional[float]
    sla_risk_reason: Optional[str]
    sla_expected_hours: dict
    sla_factors: List[dict]
    # --- итог ---
    final_answer: str
    synthesis_method: str


# ---------------------------------------------------------------------------
# Загрузка компонентов (один раз при импорте)
# ---------------------------------------------------------------------------
COMPONENT_STATUS: dict[str, str] = {}

try:
    from classifier.predictor import TicketClassifier
    _CLASSIFIER = TicketClassifier()
    COMPONENT_STATUS["classifier"] = f"ok ({_CLASSIFIER.model_type})"
except Exception as exc:
    _CLASSIFIER = None
    COMPONENT_STATUS["classifier"] = f"не загружен: {exc}"

try:
    from routing import TicketRouter, build_routing_ticket, LINE_NAMES, STATUS_NAMES
    _ROUTER = TicketRouter()
    COMPONENT_STATUS["routing"] = "ok"
except Exception as exc:
    _ROUTER = None
    COMPONENT_STATUS["routing"] = f"не загружен: {exc}"

try:
    from RagAgent.rag_build_db import clean_ticket_text as rag_clean_ticket_text
    if not os.path.isdir(config.CHROMA_DB_PATH):
        raise FileNotFoundError(
            f"база {config.CHROMA_DB_PATH} не построена — python -m RagAgent.rag_build_db "
            "(или python main.py train)"
        )
    from RagAgent.rag_search import TicketRetriever
    _RAG_RETRIEVER = TicketRetriever(
        db_path=config.CHROMA_DB_PATH, distance_threshold=config.RAG_DISTANCE_THRESHOLD
    )
    COMPONENT_STATUS["rag"] = f"ok ({_RAG_RETRIEVER.collection.count()} обращений в базе)"
except Exception as exc:
    _RAG_RETRIEVER = None
    COMPONENT_STATUS["rag"] = f"не загружен: {exc}"

try:
    from sla.risk import SLARiskEstimator
    _SLA = SLARiskEstimator()
    COMPONENT_STATUS["sla"] = f"ok (история {len(_SLA.df)} обращений, просрочено {_SLA.base_rate:.1%})"
except Exception as exc:
    _SLA = None
    COMPONENT_STATUS["sla"] = f"не загружен: {exc}"

COMPONENT_STATUS["llm"] = finalizer.llm_status()

for _name, _status in COMPONENT_STATUS.items():
    print(f"[{'+' if _status.startswith('ok') else '!'}] {_name}: {_status}")


# ---------------------------------------------------------------------------
# Узлы графа
# ---------------------------------------------------------------------------

def classify_node(state: TicketState) -> dict:
    if _CLASSIFIER is None:
        return {"predicted_class": None, "confidence": 0.0, "is_confident": False,
                "top_k": [], "explanation_words": []}
    return _CLASSIFIER.classify(
        raw_text=state.get("raw_text", ""),
        service=state.get("service", ""),
        component=state.get("component", ""),
        request_type=state.get("request_type", ""),
    )


def route_node(state: TicketState) -> dict:
    if _ROUTER is None:
        return {"recommended_line": None, "routing_status": None,
                "routing_reason": "Маршрутизатор не загружен — см. COMPONENT_STATUS['routing']."}
    ticket, source = build_routing_ticket(state)
    result = _ROUTER.route(ticket)
    line = result["recommended_line"]
    status = result["status"]
    status_ru = STATUS_NAMES.get(status, status)
    if line is None:
        top_line = max(result["probabilities"], key=result["probabilities"].get)
        reason = (f"Линия: {status_ru} (наиболее вероятна {LINE_NAMES.get(top_line, top_line)}, "
                  f"{result['confidence']:.0%}).")
    else:
        reason = f"Рекомендуемая линия: {LINE_NAMES.get(line, line)} ({status_ru}, {result['confidence']:.0%})."
    return {
        "recommended_line": line,
        "routing_confidence": result["confidence"],
        "routing_status": status,
        "routing_probabilities": result["probabilities"],
        "routing_reason": reason,
        "request_kind_source": source,
    }


def similar_tickets_node(state: TicketState) -> dict:
    """Поиск похожих обращений через RAG (TicketRetriever)."""
    if _RAG_RETRIEVER is None:
        return {"similar_tickets": []}
    raw_text = state.get("raw_text", "")
    # Ищем по тексту, очищенному той же функцией, что строила индекс ChromaDB
    # (RagAgent/rag_build_db.py: clean_ticket_text), а не по clean_text
    # классификатора: служебный префикс "услуга .. компонент .." смещает эмбеддинг.
    query_text = rag_clean_ticket_text(raw_text) if raw_text else ""
    if not query_text:
        return {"similar_tickets": []}
    try:
        similar_raw = _RAG_RETRIEVER.search_similar(query_text, top_k=config.RAG_TOP_K, as_json=False)
        if not similar_raw or not isinstance(similar_raw, list) or similar_raw[0].get("info"):
            return {"similar_tickets": []}
        return {"similar_tickets": [
            {
                "ticket_id": t.get("sim_id", "N/A"),
                "вид_запроса": t.get("cat", "N/A"),
                "линия": t.get("line", "N/A"),
                "результат_работ": t.get("result", "N/A"),
                "текст": t.get("text_preview", ""),
                "текст_полный": t.get("text") or t.get("text_preview", ""),
                "similarity_distance": t.get("distance", 0),
            }
            for t in similar_raw
        ]}
    except Exception as exc:
        print(f"[!] Ошибка при поиске похожих обращений: {exc}")
        return {"similar_tickets": []}


def sla_node(state: TicketState) -> dict:
    """Риск просрочки по истории: приоритет (ввод) + линия (маршрутизатор) + вид запроса (классификатор)."""
    if _SLA is None:
        return {"sla_risk_level": None, "sla_risk_reason": "SLA-блок не загружен — см. COMPONENT_STATUS['sla']."}
    line = state.get("recommended_line")
    if not line and state.get("routing_probabilities"):
        # manual_review: линия не рекомендована, для оценки берём наиболее вероятную
        probs = state["routing_probabilities"]
        line = max(probs, key=probs.get)
    category = state.get("request_kind") or state.get("predicted_class")
    return _SLA.assess(priority=state.get("priority"), line_code=line, category=category)


def synthesize_node(state: TicketState) -> dict:
    """Итог для оператора: GigaChat, а при недоступности — шаблон (finalizer.py)."""
    text, method = finalizer.finalize(state)
    return {"final_answer": text, "synthesis_method": method}


# ---------------------------------------------------------------------------
# Сборка графа
# ---------------------------------------------------------------------------
_NODES = [
    ("classify", classify_node),
    ("route", route_node),
    ("similar_tickets", similar_tickets_node),
    ("sla", sla_node),
    ("synthesize", synthesize_node),
]


def _compile(nodes):
    if _HAS_LANGGRAPH:
        graph = StateGraph(TicketState)
        for name, fn in nodes:
            graph.add_node(name, fn)
        graph.add_edge(START, nodes[0][0])
        for (a, _), (b, _) in zip(nodes, nodes[1:]):
            graph.add_edge(a, b)
        graph.add_edge(nodes[-1][0], END)
        return graph.compile()

    class _SequentialApp:
        """Тот же порядок узлов без LangGraph (если пакет не установлен)."""
        def invoke(self, state):
            state = dict(state)
            for _, fn in nodes:
                state.update(fn(state))
            return state
    return _SequentialApp()


# Полный граф (с итогом) — для main.py и тестов.
app = _compile(_NODES)
# Граф без итога — для UI: карточки рисуются сразу, а итог от GigaChat
# запрашивается отдельно и выводится потоком (finalizer.stream_final_answer).
analysis_app = _compile(_NODES[:-1])


def classify_new_ticket(raw_text: str, service: str = "", component: str = "",
                        request_type: str = "", **extra) -> TicketState:
    """Точка входа для остального сервиса (UI, тесты).

    extra: request_kind, user, criticality, urgency, priority, service_class, timezone.
    """
    state: TicketState = {"raw_text": raw_text, "service": service,
                          "component": component, "request_type": request_type}
    state.update({k: v for k, v in extra.items() if k in TicketState.__annotations__})
    return app.invoke(state)


def analyze_ticket(raw_text: str, service: str = "", component: str = "",
                   request_type: str = "", **extra) -> TicketState:
    """То же, что classify_new_ticket, но без итогового текста (узла synthesize).
    Быстро (< 1 с): UI сразу показывает карточки, итог — отдельно потоком."""
    state: TicketState = {"raw_text": raw_text, "service": service,
                          "component": component, "request_type": request_type}
    state.update({k: v for k, v in extra.items() if k in TicketState.__annotations__})
    return analysis_app.invoke(state)


# Соответствие колонок выгрузки Excel полям состояния
EXCEL_COLUMNS = {
    "Описание 2": "raw_text",
    "Услуга": "service",
    "Компонент услуги 1 уровня": "component",
    "Тип запроса": "request_type",
    "Вид запроса": "request_kind",
    "Пользователь": "user",
    "Критичность": "criticality",
    "Срочность": "urgency",
    "Приоритет": "priority",
    "Класс обслуживания": "service_class",
    "Часовой пояс запроса": "timezone",
}


def process_ticket_row(row: dict, use_known_request_kind: bool = False) -> TicketState:
    """Прогон строки выгрузки. По умолчанию 'Вид запроса' из строки НЕ передаётся
    маршрутизатору (для нового обращения его ещё нет) — берётся предсказание классификатора."""
    import pandas as pd
    kwargs = {}
    for col, key in EXCEL_COLUMNS.items():
        if key == "request_kind" and not use_known_request_kind:
            continue
        value = row.get(col, "")
        kwargs[key] = "" if value is None or (isinstance(value, float) and pd.isna(value)) else str(value)
    raw_text = kwargs.pop("raw_text", "")
    return classify_new_ticket(raw_text, **kwargs)
