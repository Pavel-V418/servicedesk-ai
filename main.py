"""
Точка входа ServiceDeskAI.

    python main.py train                  # классификатор + маршрутизатор + база RAG
    python main.py train --only routing   # только маршрутизатор (или --only classifier / --only rag)
    python main.py demo                   # несколько тестовых обращений через весь пайплайн
    python main.py sla-dashboard          # SLA-статистика по истории (JSON для дашборда)
    python main.py ui                     # веб-интерфейс оператора (Streamlit)
    python main.py "Текст обращения" [--service ...] [--component ...] [--request-type ...]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ROUTING_AGENT_DIR = os.path.join(PROJECT_ROOT, "routing", "routing_agent")

DEMO_TICKETS = [
    dict(raw_text="Не могу войти в личный кабинет, пишет неверный логин или пароль. Помогите сбросить.",
         service="Портал pochta.ru", request_type="Запрос на обслуживание"),
    dict(raw_text="Не загружается список отправлений из личного кабинета, ошибка при импорте файла.",
         service="Партионный прием", component="Препост", request_type="Инцидент",
         criticality="Средняя", urgency="Средняя", priority="Средний"),
    dict(raw_text="Прошу доступ к сетевой папке отдела маркетинга для нового сотрудника."),
    dict(raw_text="Не открывается страница формирования емкостей, всё зависает, работа ОПС остановлена.",
         service="Партионный прием", request_type="Инцидент", criticality="Высокая",
         urgency="Высокая", priority="Наивысший"),
]


def _run(cmd, cwd):
    print(f"\n$ {' '.join(cmd)}   (в {os.path.relpath(cwd, PROJECT_ROOT) or '.'})")
    subprocess.run(cmd, cwd=cwd, check=True)


def train(only: str | None):
    py = sys.executable
    if only in (None, "classifier"):
        _run([py, "-m", "classifier.data_preparation"], PROJECT_ROOT)
        _run([py, "-m", "classifier.model"], PROJECT_ROOT)
    if only in (None, "routing"):
        _run([py, "-m", "src.models.train_router"], ROUTING_AGENT_DIR)
    if only in (None, "rag"):
        _run([py, "-m", "RagAgent.rag_build_db"], PROJECT_ROOT)


def print_result(result: dict):
    print(f"  Категория:   {result.get('predicted_class')} ({result.get('confidence', 0):.0%}"
          f"{', выше порога' if result.get('is_confident') else ', ниже порога'})")
    if result.get("top_k"):
        print("  Топ-3:       " + "; ".join(f"{c} {p:.0%}" for c, p in result["top_k"]))
    if result.get("routing_status"):
        probs = ", ".join(f"{k} {v:.0%}" for k, v in sorted(result["routing_probabilities"].items()))
        print(f"  Линия:       {result.get('recommended_line') or '—'} [{result['routing_status']}] ({probs}); "
              f"вид запроса для маршрутизации: {result.get('request_kind_source')}")
    similar = result.get("similar_tickets") or []
    print(f"  Похожих:     {len(similar)}")
    for t in similar:
        print(f"    • {t['ticket_id']} [{t['вид_запроса']}, {t['линия']}, итог: {t['результат_работ']}, "
              f"dist={t['similarity_distance']}]")
        if t.get("текст"):
            print(f"      {t['текст']}")
    if result.get("sla_risk_level"):
        print(f"  Риск SLA:    {result['sla_risk_level']} ({result.get('sla_risk_score', 0):.1%})")
    answer = str(result.get('final_answer') or '').replace("\n", "\n      ")
    print(f"  Ответ ({result.get('synthesis_method')}):\n      {answer}")


def main():
    parser = argparse.ArgumentParser(description="Service Desk AI")
    parser.add_argument("command", nargs="?", default="demo",
                        help="train | demo | ui | sla-dashboard | текст обращения")
    parser.add_argument("--only", choices=["classifier", "routing", "rag"])
    parser.add_argument("--service", default="")
    parser.add_argument("--component", default="")
    parser.add_argument("--request-type", default="")
    parser.add_argument("--priority", default="", help="Приоритет: Наивысший / Высокий / Средний / Низкий")
    parser.add_argument("--json", action="store_true", help="вывести полный результат в JSON")
    args = parser.parse_args()

    if args.command == "train":
        train(args.only)
        return
    if args.command == "ui":
        # То же, что `streamlit run ui/app.py`, но удобно запускать кнопкой Run в PyCharm
        subprocess.run([sys.executable, "-m", "streamlit", "run",
                        os.path.join(PROJECT_ROOT, "ui", "app.py")], cwd=PROJECT_ROOT)
        return
    if args.command == "sla-dashboard":
        from sla.sla_analytics import SLAAnalyzer
        print(SLAAnalyzer().get_dashboard_data())
        return

    from pipeline import classify_new_ticket  # импорт грузит модели — только когда нужно

    tickets = DEMO_TICKETS if args.command == "demo" else [dict(
        raw_text=args.command, service=args.service, component=args.component,
        request_type=args.request_type, priority=args.priority,
    )]
    for i, ticket in enumerate(tickets, 1):
        print("\n" + "=" * 70)
        print(f"[{i}] {ticket['raw_text']}")
        result = classify_new_ticket(**ticket)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        else:
            print_result(result)


if __name__ == "__main__":
    main()
