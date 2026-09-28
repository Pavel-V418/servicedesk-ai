import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report

import sys
from pathlib import Path

# Файл лежит в classifier/tests/, корень проекта ServiceDeskAI — на два уровня выше.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from classifier import config
from classifier.predictor import TicketClassifier

_CLASSIFIER = TicketClassifier()


def main():
    print("Загружаем данные и восстанавливаем тот же frozen-тест, что и в model.py...")
    df = pd.read_csv(config.CLEANED_CSV_PATH)
    df['clean_text'] = df['clean_text'].fillna('')

    train_pool_df, test_df = train_test_split(
        df, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE,
        stratify=df['target_class'],
    )
    print(f"  train-пул: {len(train_pool_df)} записей, тест: {len(test_df)} записей "
          f"(должно совпадать с логом model.py)")

    y_true = test_df['target_class'].tolist()
    y_pred = []
    clean_text_mismatches = 0

    print(f"\nПрогоняем {len(test_df)} обращений через классификатор "
          f"(smart_clean -> ансамбль)...")
    t0 = time.time()

    for i, (_, row) in enumerate(test_df.iterrows(), 1):
        result = _CLASSIFIER.classify(
            raw_text=row.get('Описание 2', ''),
            service=row.get('Услуга', ''),
            component=row.get('Компонент услуги 1 уровня', ''),
            request_type=row.get('Тип запроса', ''),
        )
        y_pred.append(result['predicted_class'])

        # Сверяем, что preprocess_node восстановил тот же clean_text,
        # что уже лежит в CSV (проверка, что обогащение не разъехалось).
        if result['clean_text'] != row['clean_text']:
            clean_text_mismatches += 1

        if i % 50 == 0 or i == len(test_df):
            print(f"  ...{i}/{len(test_df)}")

    elapsed = time.time() - t0
    print(f"Готово за {elapsed:.1f} сек ({elapsed / len(test_df):.2f} сек/обращение).")

    print(f"\nРасхождений clean_text между preprocess_node и CSV: "
          f"{clean_text_mismatches} из {len(test_df)}")
    if clean_text_mismatches:
        print("  [!] Есть расхождения — обогащение текста в preprocess_node "
              "не полностью совпадает с data_preparation.smart_clean на этих строках, "
              "стоит разобраться до сдачи.")
    else:
        print("  Полное совпадение — preprocess_node воспроизводит обучающий текст точно.")

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')

    print(f"\n=== Результат классификатора (classifier.predictor) на замороженном тесте ===")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  F1-macro:  {f1_macro:.4f}")
    print(f"\n  Для сравнения — из лога model.py на этом же тесте:")
    print(f"  Accuracy:  0.6495 (65.0%)")
    print(f"  F1-macro:  0.6252 (ансамбль)")

    diff = f1_macro - 0.6252
    print(f"\n  Разница по F1-macro: {diff:+.4f}", end="")
    if abs(diff) < 1e-6:
        print("  -> ИДЕНТИЧНО. Граф воспроизводит model.py на 100%.")
    elif abs(diff) < 0.01:
        print("  -> в пределах погрешности, граф эквивалентен model.py.")
    else:
        print("  -> ЕСТЬ РЕАЛЬНОЕ РАСХОЖДЕНИЕ, нужно разбираться "
              "(разная версия cleaned_data.csv? другая версия бандла?).")

    print("\n" + classification_report(y_true, y_pred, zero_division=0))


if __name__ == "__main__":
    main()