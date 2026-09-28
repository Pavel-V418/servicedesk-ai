import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import sys
from pathlib import Path

# Файл лежит в classifier/tests/, корень проекта ServiceDeskAI — на два уровня выше.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from classifier import config
from classifier.predictor import TicketClassifier

_CLASSIFIER = TicketClassifier()

PROBLEM_CLASSES = [
    "Проблема с ОПС/доставкой",
    "Импорт списков из архива ф.103 (zip-архив)",
    "Проблема с авторизацией/ЭЗП/Бонусами/Доверенностями",
]


def main():
    print("Восстанавливаем тот же frozen-тест, что и в model.py / test_frozen_set.py...")
    df = pd.read_csv(config.CLEANED_CSV_PATH)
    df['clean_text'] = df['clean_text'].fillna('')

    _, test_df = train_test_split(
        df, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE,
        stratify=df['target_class'],
    )

    print(f"Прогоняем {len(test_df)} обращений через классификатор...")
    y_true, y_pred, examples = [], [], []
    for _, row in test_df.iterrows():
        result = _CLASSIFIER.classify(
            raw_text=row.get('Описание 2', ''),
            service=row.get('Услуга', ''),
            component=row.get('Компонент услуги 1 уровня', ''),
            request_type=row.get('Тип запроса', ''),
        )
        y_true.append(row['target_class'])
        y_pred.append(result['predicted_class'])
        examples.append((row, result))

    classes = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)

    print("\n" + "=" * 70)
    print("РАЗБОР ПРОБЛЕМНЫХ КЛАССОВ")
    print("=" * 70)

    for cls in PROBLEM_CLASSES:
        if cls not in cm_df.index:
            print(f"\n[!] Класс {cls!r} не найден среди истинных меток теста — пропуск.")
            continue

        support = int(cm_df.loc[cls].sum())
        print(f"\n--- {cls} (support в тесте = {support}) ---")

        row_counts = cm_df.loc[cls]
        confused_with = row_counts[row_counts > 0].sort_values(ascending=False)
        print("Куда модель относит реальные обращения этого класса:")
        for target_cls, n in confused_with.items():
            marker = "  <- ЭТО ПРАВИЛЬНЫЙ ОТВЕТ" if target_cls == cls else ""
            share = n / support
            print(f"  -> {target_cls!r}: {n} ({share:.0%}){marker}")

        print("\nПримеры ошибок (для отчёта: 'примеры ошибок' по ТЗ):")
        shown = 0
        for row, result in examples:
            if row['target_class'] == cls and result['predicted_class'] != cls:
                print(f"  Истина: {cls!r}")
                print(f"  Предсказано: {result['predicted_class']!r} "
                      f"(уверенность {result['confidence']:.2f})")
                print(f"  Текст: {str(row['clean_text'])[:200]!r}")
                print()
                shown += 1
            if shown >= 5:
                break
        if shown == 0:
            print("  (ошибок нет — все примеры этого класса классифицированы верно)")

    print("=" * 70)
    print("\nИнтерпретация:")
    print("  - Если для 'Проблема с ОПС/доставкой' почти все ошибки уходят в")
    print("    один и тот же класс (скорее всего 'Отслеживание отправлений') —")
    print("    это подтверждает пересечение таксономии, а не брак модели.")
    print("  - Если для ф.103/авторизации ошибки размазаны по разным классам")
    print("    без явного паттерна — это подтверждает гипотезу 'слишком мало")
    print("    данных', а не системную путаницу с конкретным классом.")


if __name__ == "__main__":
    main()