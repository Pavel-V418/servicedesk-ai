"""
Классификатор обращений по виду запроса (топ-15 видов) — часть Павла.

    from classifier import TicketClassifier
    clf = TicketClassifier()
    clf.classify(raw_text=..., service=..., component=..., request_type=...)

Обучение:
    python -m classifier.data_preparation   # dataset/cleaned_data.csv
    python -m classifier.model              # dataset/ensemble_model_bundle.joblib
"""


def __getattr__(name):
    # Ленивый импорт: `python -m classifier.model` не должен тянуть за собой
    # predictor (и sentence-transformers) только из-за импорта пакета.
    if name == "TicketClassifier":
        from classifier.predictor import TicketClassifier
        return TicketClassifier
    raise AttributeError(name)
