from __future__ import annotations
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer

from src.features.feature_builder import TEXT_COL, CATEGORICAL_COLS


def _select_text(x):
    return x[TEXT_COL]


def build_baseline():
    text_features = Pipeline([
        ("select", FunctionTransformer(_select_text, validate=False)),
        ("tfidf", FeatureUnion([
            ("word", TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_features=12000,
                sublinear_tf=True,
            )),
            ("char", TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=2,
                max_features=12000,
                sublinear_tf=True,
            )),
        ])),
    ])

    metadata = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
    ], remainder="drop")

    features = FeatureUnion([
        ("text", text_features),
        ("metadata", metadata),
    ])

    model = OneVsRestClassifier(
        LogisticRegression(
            max_iter=1200,
            class_weight=None,
            solver="liblinear",
        )
    )

    return Pipeline([
        ("features", features),
        ("model", model),
    ])
