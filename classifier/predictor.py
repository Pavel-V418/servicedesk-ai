from __future__ import annotations

import os
import re
from typing import List, Optional, Tuple

import joblib
import numpy as np

from classifier import config
from classifier.data_preparation import smart_clean
from classifier.model import _align_proba_columns, _word_features_only


def _get_inner_pipeline(pipeline_or_calibrated):
    if hasattr(pipeline_or_calibrated, 'calibrated_classifiers_'):
        return pipeline_or_calibrated.calibrated_classifiers_[0].estimator
    return pipeline_or_calibrated


_TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")  # тот же token_pattern, что у TfidfVectorizer по умолчанию


def _ticket_text_part(clean_text: str) -> str:
    """Часть clean_text с текстом самого обращения (после 'запрос:'), в виде токенов через пробел."""
    part = clean_text.split("запрос:", 1)[-1]
    return " " + " ".join(_TOKEN_RE.findall(part.lower())) + " "


def explain_top_words(inner_pipeline, text, class_label, top_n=8, only_ticket_text=True) -> List[Tuple[str, float]]:
    """Слова/n-граммы, сильнее всего повлиявшие на класс (компонент LogReg).

    only_ticket_text=True: оператору показываются только n-граммы, которые
    есть в тексте обращения. smart_clean склеивает поля шаблоном
    "услуга .. компонент .. тип .. запрос: ..", и без фильтра в объяснение
    попадают служебные склейки вроде 'компонент тип' или 'инцидент запрос'.
    На предсказание фильтр не влияет — только на то, что видит оператор.
    """
    vectorizer = inner_pipeline.named_steps['tfidf']
    clf = inner_pipeline.named_steps['clf']
    x_vec = vectorizer.transform([text])
    class_idx = list(clf.classes_).index(class_label)
    feature_names, show_mask = _word_features_only(vectorizer)
    coefs = clf.coef_[class_idx]
    x_arr = x_vec.toarray().ravel()
    contribution = np.where(show_mask, coefs * x_arr, 0.0)
    order = np.argsort(contribution)[::-1]
    ticket_part = _ticket_text_part(text) if only_ticket_text else None
    result = []
    for i in order:
        if contribution[i] <= 0 or len(result) >= top_n:
            break
        name = str(feature_names[i])
        if ticket_part is not None and f" {name} " not in ticket_part:
            continue
        result.append((name, float(contribution[i])))
    return result


class TicketClassifier:
    """Категоризация обращения по 15 видам запроса."""

    def __init__(self, bundle_path: Optional[str] = None):
        if bundle_path is None:
            if os.path.exists(config.ENSEMBLE_BUNDLE_PATH):
                bundle_path = config.ENSEMBLE_BUNDLE_PATH
            elif os.path.exists(config.PRIMARY_MODEL_BUNDLE_PATH):
                bundle_path = config.PRIMARY_MODEL_BUNDLE_PATH
            else:
                raise FileNotFoundError(
                    "Не найден обученный классификатор "
                    f"({config.ENSEMBLE_BUNDLE_PATH} или {config.PRIMARY_MODEL_BUNDLE_PATH}). "
                    "Обучи его: python main.py train  "
                    "(или python -m classifier.data_preparation, затем python -m classifier.model)."
                )
        self.bundle_path = bundle_path
        self.bundle = joblib.load(bundle_path)
        self.classes = self.bundle['classes_']
        self.threshold = float(self.bundle['threshold'])
        self.top_k = self.bundle.get('top_k_alternatives', config.TOP_K_ALTERNATIVES)
        self.is_ensemble = 'catboost_model' in self.bundle
        self.weight_logreg = float(self.bundle.get('ensemble_weight_logreg', 1.0))
        self._embedder = None
        if self.is_ensemble:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.bundle['embedder_name'])

    @property
    def model_type(self) -> str:
        return str(self.bundle.get('model_type', 'unknown'))

    @staticmethod
    def preprocess(raw_text: str = "", service: str = "", component: str = "", request_type: str = "") -> str:
        row = {
            'Описание 2': raw_text,
            'Услуга': service,
            'Компонент услуги 1 уровня': component,
            'Тип запроса': request_type,
        }
        return smart_clean(row)

    def predict_proba_clean(self, clean_text: str) -> np.ndarray:
        primary = self.bundle['primary_model']
        proba = _align_proba_columns(primary.predict_proba([clean_text]), primary.classes_, self.classes)
        if self.is_ensemble and self.weight_logreg < 1.0:
            emb = np.asarray(self._embedder.encode([clean_text]))
            cb = self.bundle['catboost_model']
            cb_proba = _align_proba_columns(cb.predict_proba(emb), cb.classes_, self.classes)
            proba = self.weight_logreg * proba + (1 - self.weight_logreg) * cb_proba
        return proba[0]

    def classify(self, raw_text: str = "", service: str = "", component: str = "",
                 request_type: str = "") -> dict:
        clean_text = self.preprocess(raw_text, service, component, request_type)
        proba = self.predict_proba_clean(clean_text)
        order = np.argsort(proba)[::-1]
        predicted_class = str(self.classes[order[0]])
        confidence = float(proba[order[0]])
        inner = _get_inner_pipeline(self.bundle['primary_model'])
        return {
            "clean_text": clean_text,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "is_confident": confidence >= self.threshold,
            "classifier_threshold": self.threshold,
            "top_k": [(str(self.classes[i]), float(proba[i])) for i in order[:self.top_k]],
            "explanation_words": explain_top_words(inner, clean_text, predicted_class),
        }
