import sys
import os
from pathlib import Path

# Добавляем корень проекта в путь — файл лежит в tests/, а pipeline.py,
# config.py и пакет RagAgent/ лежат в корне проекта.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

print("\n" + "="*80)
print(" ДЕМОНСТРАЦИЯ RAG ИНТЕГРАЦИИ В ПАЙПЛАЙН КЛАССИФИКАЦИИ")
print("="*80 + "\n")

# Инициализируем БД если нужна
print("[ЭТАП 1] Инициализация ChromaDB базы...")
print("-" * 80)

from RagAgent.rag_build_db import build_db

if not os.path.exists(config.CHROMA_DB_PATH):
    print("Создаю новую БД...")
    if not os.path.exists(config.RAG_XLSX_PATH):
        print(f"[!] Файл {config.RAG_XLSX_PATH} не найден!")
        sys.exit(1)
    n = build_db()
    print(f"✓ БД создана с {n} обращениями\n")
else:
    print("✓ БД уже существует\n")

# Загружаем компоненты
print("[ЭТАП 2] Загрузка компонентов...")
print("-" * 80)

try:
    from pipeline import classify_new_ticket, _RAG_RETRIEVER
    print("✓ Компоненты загружены\n")
except Exception as e:
    print(f"[!] Ошибка: {e}")
    sys.exit(1)

# Демонстрируем работу
print("[ЭТАП 3] Демонстрация работы")
print("-" * 80)

test_cases = [
    {
        "name": "Проблема с доступом",
        "raw_text": "Не могу войти в личный кабинет, пишет неверный логин или пароль. Помогите сбросить пароль.",
        "service": "Портал pochta.ru",
        "component": "",
        "request_type": "Запрос на обслуживание",
    },
    {
        "name": "Запрос на доступ",
        "raw_text": "Нужен доступ к сетевой папке отдела маркетинга для нового сотрудника Иванова И.И.",
        "service": "",
        "component": "",
        "request_type": "",
    },
]

for idx, case in enumerate(test_cases, 1):
    print(f"\n[ОБРАЩЕНИЕ {idx}] {case['name']}")
    print(f"Текст: \"{case['raw_text']}\"")
    print()

    result = classify_new_ticket(
        raw_text=case['raw_text'],
        service=case['service'],
        component=case['component'],
        request_type=case['request_type'],
    )

    print("  РЕЗУЛЬТАТЫ КЛАССИФИКАЦИИ:")
    print(f"    • Категория: {result['predicted_class']}")
    print(f"    • Уверенность: {result['confidence']:.1%}")
    print(f"    • Пороговая граница пройдена: {'✓ Да' if result['is_confident'] else '✗ Нет'}")

    if result.get('explanation_words'):
        words = ", ".join(w for w, _ in result['explanation_words'][:3])
        print(f"    • Ключевые слова: {words}")

    # Показываем все найденные тикеты и результат работ целиком —
    # раньше здесь стояло [:50] на результате, обрезавшее текст без причины.
    print("\n  ПОХОЖИЕ ОБРАЩЕНИЯ ИЗ ИСТОРИИ (RAG):")
    if result.get('similar_tickets'):
        for i, ticket in enumerate(result['similar_tickets'], 1):
            print(f"    [{i}] Тикет {ticket['ticket_id']}")
            print(f"        Вид запроса: {ticket['вид_запроса']}")
            print(f"        Линия поддержки: {ticket['линия']}")
            print(f"        Результат: {ticket['результат_работ']}")
            print(f"        Схожесть: {1 - ticket['similarity_distance']:.0%}")
    else:
        print("    Нет достаточно похожих обращений в истории")

    # Печатаем финальный ответ полностью — раньше он резался до 150 символов.
    print("\n  ФИНАЛЬНЫЙ ОТВЕТ ОПЕРАТОРУ:")
    print(f"    {result['final_answer']}")

    print("\n" + "-" * 80)

print("\n" + "="*80)
print(" ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА")
print("="*80)
print("""
Что было продемонстрировано:
  ✓ Загрузка и инициализация компонентов
  ✓ Классификация обращений (ансамбль LogReg + CatBoost)
  ✓ Поиск похожих обращений через RAG (ChromaDB)
  ✓ Включение результатов RAG в итоговый ответ
  ✓ Готовность системы к использованию

RAG полностью интегрирован в архитектуру!
""")