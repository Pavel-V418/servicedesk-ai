import argparse
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402  (заодно читает .env)
import finalizer  # noqa: E402

# Результат анализа типичного обращения — чтобы замер не зависел от моделей.
SAMPLE_STATE = {
    "raw_text": "Добрый день. Не могу войти в личный кабинет, пишет неверный логин или пароль. "
                "Сброс пароля не приходит на почту. Помогите восстановить доступ.",
    "service": "Портал pochta.ru", "request_type": "Запрос на обслуживание", "priority": "(3) Средний",
    "predicted_class": "Проблема с авторизацией/ЭЗП/Бонусами/Доверенностями", "confidence": 0.52,
    "is_confident": True, "classifier_threshold": 0.45,
    "top_k": [("Проблема с авторизацией/ЭЗП/Бонусами/Доверенностями", 0.52),
              ("Личный кабинет", 0.20), ("Прочее", 0.04)],
    "explanation_words": [("войти", 0.2), ("пароль", 0.15), ("личный кабинет", 0.1)],
    "routing_status": "high_confidence", "recommended_line": "L1",
    "routing_probabilities": {"L1": 0.83, "L2": 0.13, "L3": 0.04},
    "routing_reason": "Рекомендуемая линия: 1 линия (высокая уверенность, 83%).",
    "sla_risk_level": "низкий", "sla_risk_score": 0.0,
    "sla_risk_reason": "В истории просрочено 0 из 56 обращений (0.0%) по признаку: вид запроса "
                       "'Проблема с авторизацией/ЭЗП/Бонусами/Доверенностями'; в среднем по выгрузке 1.2%.",
    "sla_expected_hours": {"basis": "вид 'Проблема с авторизацией/ЭЗП/Бонусами/Доверенностями' на (1 линия)",
                           "n": 40, "median": 0.4, "p90": 3.1},
    "similar_tickets": [
        {"ticket_id": "#38906919", "вид_запроса": "Проблема с авторизацией/ЭЗП/Бонусами/Доверенностями",
         "линия": "(1 линия)", "результат_работ": "Уточнение не предоставлено",
         "текст": "Не могу зайти в личный кабинет, не приходит письмо для восстановления пароля..."},
    ],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", type=int, default=0, help="сколько раз замерить время итога")
    parser.add_argument("--show", action="store_true", help="показать итог целиком")
    args = parser.parse_args()

    credentials = os.environ.get(config.GIGACHAT_CREDENTIALS_ENV)
    if not credentials:
        sys.exit(f"Не задан {config.GIGACHAT_CREDENTIALS_ENV} (переменная окружения или .env в корне проекта)")

    from gigachat import GigaChat
    with GigaChat(credentials=credentials, verify_ssl_certs=False, scope=config.GIGACHAT_SCOPE) as client:
        models = [getattr(m, "id_", None) or getattr(m, "id", "?") for m in client.get_models().data]
    print("Доступные модели:", ", ".join(models))
    if config.GIGACHAT_MODEL not in models:
        print(f"[!] В config выбрана {config.GIGACHAT_MODEL}, её нет в списке — задай GIGACHAT_MODEL в .env")

    print(f"Финализатор: {finalizer.llm_status()}")

    if args.show:
        info = {}
        text = "".join(finalizer.stream_final_answer(SAMPLE_STATE, info))
        print(f"\n--- Итог ({info.get('method')}, первый текст через {info.get('first_token_sec')} с, "
              f"всего {info.get('total_sec')} с) ---\n{text}\n")

    if args.bench:
        firsts, totals = [], []
        for i in range(1, args.bench + 1):
            info = {}
            t0 = time.perf_counter()
            text = "".join(finalizer.stream_final_answer(SAMPLE_STATE, info))
            total = time.perf_counter() - t0
            print(f"  {i}: метод={info.get('method')}, первый текст {info.get('first_token_sec')} с, "
                  f"всего {total:.2f} с, {len(text.split())} слов")
            if info.get("method") == "llm":
                firsts.append(info.get("first_token_sec") or total)
                totals.append(total)
        if totals:
            print(f"\nПервый текст: среднее {statistics.mean(firsts):.2f} с, макс {max(firsts):.2f} с")
            print(f"Итог целиком: среднее {statistics.mean(totals):.2f} с, макс {max(totals):.2f} с")


if __name__ == "__main__":
    main()
