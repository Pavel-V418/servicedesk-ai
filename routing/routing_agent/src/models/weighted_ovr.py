"""
One-vs-Rest логистическая регрессия с весом "своего" класса.

Зачем этот файл
---------------
Исходный RoutePredictor был LogisticRegression(solver="liblinear",
class_weight={"L1": 1.0, "L2": 1.3, "L3": 1.5}) на трёх классах. До
scikit-learn 1.8 liblinear молча решал мультикласс схемой One-vs-Rest.
Начиная с 1.8 это запрещено (ValueError: 'liblinear' solver does not support
multiclass classification), а в объединённом проекте стоит scikit-learn 1.9.
Плюс сохранённый joblib из sklearn 1.6.1 не открывается в 1.9
(AttributeError: _RemainderColsList) — модель всё равно надо переобучать.

Этот класс воспроизводит старое поведение liblinear один в один:
  * для каждого класса k обучается бинарная LogisticRegression(liblinear)
    "k против остальных";
  * вес класса k применяется к ПОЗИТИВНЫМ примерам (как weighted_C в liblinear),
    негативы идут с весом 1.0;
  * predict_proba = сигмоида decision_function по каждому классу,
    нормированная на сумму 1 (так же, как делал sklearn для OvR-liblinear).

Работает на любой версии sklearn >= 1.3.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression


class WeightedOneVsRestLogReg(ClassifierMixin, BaseEstimator):
    def __init__(self, class_weight: dict | None = None, C: float = 1.0, max_iter: int = 1200):
        self.class_weight = class_weight
        self.C = C
        self.max_iter = max_iter

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        weights = self.class_weight or {}
        base = LogisticRegression(solver="liblinear", C=self.C, max_iter=self.max_iter)
        self.estimators_ = []
        for cls in self.classes_:
            est = clone(base)
            est.set_params(class_weight={1: float(weights.get(cls, 1.0)), 0: 1.0})
            est.fit(X, (y == cls).astype(int))
            self.estimators_.append(est)
        return self

    def decision_function(self, X):
        return np.column_stack([est.decision_function(X) for est in self.estimators_])

    def predict_proba(self, X):
        scores = 1.0 / (1.0 + np.exp(-self.decision_function(X)))
        return scores / scores.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]
