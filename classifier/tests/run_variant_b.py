import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from classifier import config, model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline


def build_primary_pipeline_B():
    features = FeatureUnion([
        ("word", TfidfVectorizer(**config.TFIDF_PARAMS)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)),
    ])
    return Pipeline([("tfidf", features), ("clf", LogisticRegression(**config.LOGREG_PARAMS))])


# model.py вызывает build_primary_pipeline() через глобальное имя модуля, поэтому подмена работает
# и для CV, и для OOF, и для финального обучения.
model.build_primary_pipeline = build_primary_pipeline_B

# Отдельные файлы артефактов, текущий бандл не трогаем.
config.ENSEMBLE_BUNDLE_PATH = os.path.join(config.DATA_DIR, "ensemble_model_bundle_B.joblib")
config.PRIMARY_MODEL_BUNDLE_PATH = os.path.join(config.DATA_DIR, "primary_model_bundle_B.joblib")

if __name__ == "__main__":
    model.train_classifier()