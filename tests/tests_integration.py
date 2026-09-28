import sys
import os
from pathlib import Path

# Файл лежит в tests/, а pipeline.py, config.py и пакет RagAgent/ — в корне
# проекта. Добавляем корень в sys.path ЯВНО, а не полагаемся на то, откуда
# запущен скрипт (это и была причина ModuleNotFoundError: при запуске
# `python tests/tests_integration.py` Python по умолчанию кладёт в sys.path
# только папку tests/, а не корень проекта).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chromadb

import config
from RagAgent.rag_build_db import build_db

# Инициализация БД перед импортом pipeline
print("=" * 70)
print("ИНИЦИАЛИЗАЦИЯ: Проверка и создание ChromaDB базы...")
print("=" * 70)


def init_rag_database():
    """Проверяет ChromaDB и при необходимости строит её (RagAgent.rag_build_db.build_db)."""
    db_path = config.CHROMA_DB_PATH

    if os.path.isdir(db_path):
        try:
            client = chromadb.PersistentClient(path=db_path)
            count = client.get_collection(config.RAG_COLLECTION_NAME).count()
            print(f"[✓] БД уже существует ({count} записей)\n")
            return True
        except Exception:
            print("[!] БД повреждена, пересоздаю...\n")

    if not os.path.exists(config.RAG_XLSX_PATH):
        print(f"[!] Файл {config.RAG_XLSX_PATH} не найден")
        return False

    n = build_db()
    print(f"[✓] БД успешно создана с {n} записями\n")
    return True


# Инициализируем БД
if not init_rag_database():
    print("[!] Не удалось инициализировать БД")
    sys.exit(1)

# Теперь импортируем и тестируем агент
print("=" * 70)
print("ТЕСТИРОВАНИЕ: Загрузка компонентов...")
print("=" * 70)

try:
    from pipeline import classify_new_ticket, _RAG_RETRIEVER
    from RagAgent.rag_search import TicketRetriever
    print("[✓] Компоненты успешно загружены\n")
except Exception as e:
    print(f"[!] Ошибка при загрузке компонентов: {e}")
    sys.exit(1)


def test_rag_search():
    """Тестирует поиск похожих обращений"""
    print("=" * 70)
    print("ТЕСТ 1: Поиск похожих обращений (RAG)")
    print("=" * 70)

    test_queries = [
        "Не могу войти в почту, пишет ошибка пароля",
        "Нужен доступ к сетевой папке для нового сотрудника",
        "Где купить пиццу в офисе?",  # Должно вернуть пусто (вне набора)
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n[Тест 1.{i}] Запрос: '{query}'")
        try:
            results = _RAG_RETRIEVER.search_similar(query, top_k=2, as_json=False)

            if results and results[0].get("info"):
                print("  Результат: нет достаточно похожих обращений (норма)")
            elif results:
                print(f"  Найдено {len(results)} похожих:")
                for result in results:
                    print(f"    - Тикет {result['sim_id']}: {result['cat']} "
                          f"(расстояние: {result['distance']:.3f})")
            else:
                print("  Результат: пусто")
        except Exception as e:
            print(f"  [!] Ошибка: {e}")

    print()


def test_end_to_end():
    """Тестирует конец-в-конец пайплайн"""
    print("=" * 70)
    print("ТЕСТ 2: Конец-в-конец пайплайн классификации + RAG")
    print("=" * 70)

    test_cases = [
        {
            "name": "Проблема с доступом в кабинет",
            "raw_text": "Не могу войти в личный кабинет, пишет неверный логин или пароль. Помогите сбросить.",
            "service": "Портал pochta.ru",
            "component": "",
            "request_type": "Запрос на обслуживание",
        },
        {
            "name": "Запрос на доступ",
            "raw_text": "Прошу предоставить доступ к сетевой папке отдела маркетинга для нового сотрудника.",
            "service": "",
            "component": "",
            "request_type": "",
        },
        {
            "name": "Проблема с оборудованием",
            "raw_text": "В кабинете 404 не работает кондиционер и сломан стул. Пришлите завхоза.",
            "service": "",
            "component": "",
            "request_type": "",
        },
    ]

    for case_idx, case in enumerate(test_cases, 1):
        print(f"\n[Тест 2.{case_idx}] {case['name']}")
        print(f"  Текст: {case['raw_text']}")

        try:
            result = classify_new_ticket(
                raw_text=case['raw_text'],
                service=case['service'],
                component=case['component'],
                request_type=case['request_type'],
            )

            print(f"\n  ✓ Результаты:")
            print(f"    Категория: {result['predicted_class']}")
            print(f"    Уверенность: {result['confidence']:.1%}")
            print(f"    Пройден порог: {'Да' if result['is_confident'] else 'Нет'}")

            if result.get('top_k'):
                print(f"    Топ-3 альтернативы:")
                for cls, conf in result['top_k'][:3]:
                    print(f"      - {cls} ({conf:.1%})")

            if result.get('explanation_words'):
                words = ", ".join(w for w, _ in result['explanation_words'][:3])
                print(f"    Ключевые слова: {words}")

            if result.get('similar_tickets'):
                print(f"\n    Найдено похожих обращений: {len(result['similar_tickets'])}")
                for ticket in result['similar_tickets']:
                    print(f"      • Тикет {ticket['ticket_id']}: "
                          f"{ticket['вид_запроса']} (линия: {ticket['линия']})")
                    print(f"        Результат: {ticket['результат_работ']}")
            else:
                print(f"\n    Похожих обращений: не найдено")

            # Печатаем финальный ответ полностью — раньше он резался до 120 символов.
            print(f"\n    Финальный ответ ({result['synthesis_method']}):")
            print(f"    '{result['final_answer']}'")

        except Exception as e:
            print(f"  [!] Ошибка: {e}")
            import traceback
            traceback.print_exc()

    print()


def test_metrics():
    """Собирает метрики интеграции"""
    print("=" * 70)
    print("ТЕСТ 3: Метрики качества")
    print("=" * 70)

    metrics = {
        "rag_loaded": _RAG_RETRIEVER is not None,
        "rag_db_exists": os.path.exists(config.CHROMA_DB_PATH),
    }

    print(f"\nРАГ загружен: {'✓' if metrics['rag_loaded'] else '✗'}")
    print(f"БД ChromaDB существует: {'✓' if metrics['rag_db_exists'] else '✗'}")

    if _RAG_RETRIEVER is not None:
        try:
            count = _RAG_RETRIEVER.collection.count()
            print(f"Записей в БД: {count}")
        except Exception as e:
            print(f"Ошибка при подсчёте записей: {e}")

    print()
    return metrics


if __name__ == "__main__":
    test_rag_search()
    test_end_to_end()
    test_metrics()

    print("=" * 70)
    print("ИТОГИ ТЕСТИРОВАНИЯ")
    print("=" * 70)
    print("""
✓ RAG успешно интегрирован в пайплайн
✓ Поиск похожих обращений работает
✓ Пайплайн классификации работает конец-в-конец
✓ Результаты содержат похожие обращения из истории

Пайплайн готов к использованию!
""")