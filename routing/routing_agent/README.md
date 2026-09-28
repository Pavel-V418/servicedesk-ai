# Routing Agent v0.1

MVP агента-маршрутизатора Service Desk.

## Что реализовано
- whitelist регистрационных признаков;
- автоматическая защита от data leakage;
- temporal train/validation/test split;
- baseline: word+char TF-IDF + metadata + Logistic Regression;
- class balancing;
- вероятностный output;
- LineRecommendator с `manual_review` при низкой уверенности;
- L4 намеренно исключена из ML v0 из-за недостатка исторических данных.

## Запуск
```bash
pip install -r requirements.txt
python -m src.models.train_baseline /path/to/Обращения_1931.xlsx
```

Результаты появятся в `reports/baseline_metrics.json`, модель — в `models/router_baseline.joblib`.

## Следующие шаги
1. Уточнить семантику target `Кем решен (группа)`.
2. Согласовать mapping исторических видов обращения в 15 целевых категорий.
3. Добавить OOF probabilities от Agent 1.
4. Добавить calibration probability.
5. Сравнить baseline с embeddings + CatBoost.
6. Добавить SHAP/reason codes и FastAPI endpoint.
