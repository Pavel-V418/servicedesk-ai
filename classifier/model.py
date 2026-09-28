import numpy as np
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score, classification_report, top_k_accuracy_score

from classifier import config

try:
    from catboost import CatBoostClassifier
    from sentence_transformers import SentenceTransformer
    _HAS_COMPARISON_DEPS = True
except ImportError:
    _HAS_COMPARISON_DEPS = False


# ---------------------------------------------------------------------------
# Данные
# ---------------------------------------------------------------------------

def load_datasets():
    df = pd.read_csv(config.CLEANED_CSV_PATH)
    df['clean_text'] = df['clean_text'].fillna('')

    try:
        out_of_scope = pd.read_csv(config.OUT_OF_SCOPE_CSV_PATH)
        out_of_scope['clean_text'] = out_of_scope['clean_text'].fillna('')
    except FileNotFoundError:
        out_of_scope = None
        print(
            f"[!] Не найден {config.OUT_OF_SCOPE_CSV_PATH} — запусти обновлённый "
            "data_preparation.py, чтобы получить его и проверить сценарий "
            "«обращение вне набора»."
        )
    return df, out_of_scope


# ---------------------------------------------------------------------------
# Общие утилиты (используются и для отдельных моделей, и для ансамбля)
# ---------------------------------------------------------------------------

def _align_proba_columns(proba, source_classes, target_classes):
    """Переставляет колонки матрицы вероятностей под единый порядок классов.

    Нужно, потому что модели хранят свой список классов в `classes_`, и хотя
    обе модели обучены на одном и том же наборе меток, полагаться на
    случайное совпадение порядка колонок в чужом коде — плохая идея. Без
    этого усреднение вероятностей двух моделей могло бы молча сложить
    вероятность одного класса с вероятностью другого.
    """
    source_classes = np.asarray(source_classes)
    target_classes = np.asarray(target_classes)
    idx = np.array([np.where(source_classes == c)[0][0] for c in target_classes])
    return proba[:, idx]


def evaluate_top_k(proba, y_true, classes, k=3, label=""):
    y_true = np.asarray(y_true)
    y_true_idx = np.array([np.where(classes == c)[0][0] for c in y_true])
    top1 = top_k_accuracy_score(y_true_idx, proba, k=1, labels=np.arange(len(classes)))
    topk = top_k_accuracy_score(y_true_idx, proba, k=k, labels=np.arange(len(classes)))
    prefix = f"[{label}] " if label else ""
    print(f"  {prefix}Top-1 accuracy: {top1:.1%}")
    print(f"  {prefix}Top-{k} accuracy: {topk:.1%}")
    return top1, topk


def coverage_precision_at(proba, y_true, classes, threshold):
    """Доля автоматических решений и точность на них при заданном пороге.
    Переиспользуется и для OOF-таблицы, и для единственной проверки на тесте."""
    y_true = np.asarray(y_true)
    top1_idx = np.argmax(proba, axis=1)
    top1_class = classes[top1_idx]
    conf = proba[np.arange(len(proba)), top1_idx]
    mask = conf >= threshold
    coverage = mask.mean()
    precision = (top1_class[mask] == y_true[mask]).mean() if mask.any() else float('nan')
    return coverage, precision


def pick_threshold(proba, y_true, classes, target_precision, label=""):
    """Подбирает наименьший порог, при котором точность среди
    автоматически принятых решений не ниже target_precision. Печатает
    полную таблицу порог/покрытие/точность."""
    prefix = f"[{label}] " if label else ""
    print(f"\n  {prefix}Порог | Доля автоматических решений | Точность на них")
    rows = []
    for threshold in np.arange(0.30, 0.91, 0.05):
        coverage, precision = coverage_precision_at(proba, y_true, classes, threshold)
        rows.append((threshold, coverage, precision))
        print(f"  {threshold:5.2f} | {coverage:26.1%} | {precision:15.1%}")

    chosen = None
    for threshold, coverage, precision in rows:
        if not np.isnan(precision) and precision >= target_precision:
            chosen = threshold
            break
    if chosen is None:
        chosen = rows[-1][0]
        print(
            f"  [!] Ни один порог не достиг целевой точности {target_precision:.0%}, "
            f"взят максимальный проверенный порог {chosen:.2f}."
        )
    else:
        print(f"  -> выбран порог {chosen:.2f} (первый, достигающий точности >= {target_precision:.0%})")
    return float(chosen)


def pick_ensemble_weight(proba_a, proba_b, y_true, classes, weight_grid):
    """Подбирает вес w для w*proba_a + (1-w)*proba_b по macro-F1 на
    OOF-предсказаниях train-пула (тест не участвует). proba_a — LogReg,
    proba_b — CatBoost."""
    y_true = np.asarray(y_true)
    best_w, best_f1 = 0.5, -1.0
    print("\n  Вес LogReg | Вес CatBoost | F1-macro ансамбля (OOF, train-пул)")
    for w in weight_grid:
        proba = w * proba_a + (1 - w) * proba_b
        pred = classes[np.argmax(proba, axis=1)]
        f1 = f1_score(y_true, pred, average='macro')
        print(f"  {w:10.2f} | {1 - w:12.2f} | {f1:.4f}")
        if f1 > best_f1:
            best_f1, best_w = f1, w
    print(f"  -> выбран вес LogReg={best_w:.2f} / CatBoost={1 - best_w:.2f}, OOF F1-macro={best_f1:.4f}")
    return best_w, best_f1


_FEATURE_VIEW_CACHE = {}


def _word_features_only(vectorizer):
    """Имена признаков для объяснения и маска «можно показывать оператору».

    У FeatureUnion (слова + символьные n-граммы) get_feature_names_out() отдает имена с префиксами
    'word__' / 'char__'. Символьные обрубки вроде 'втори' оператору ничего не говорят, поэтому в объяснении
    остаются только словные признаки, префикс снимается. Для обычного векторайзера (без префиксов)
    поведение прежнее. Результат кэшируется: список имен не меняется между запросами."""
    key = id(vectorizer)
    if key not in _FEATURE_VIEW_CACHE:
        names = [str(n) for n in vectorizer.get_feature_names_out()]
        display = np.array([n[len('word__'):] if n.startswith('word__') else n for n in names], dtype=object)
        show = np.array([not n.startswith('char__') for n in names])
        _FEATURE_VIEW_CACHE[key] = (display, show)
    return _FEATURE_VIEW_CACHE[key]


def explain_prediction(pipeline_or_calibrated, raw_text, top_n=8):
    """Простая объяснимость для TF-IDF + LogReg: какие слова сильнее всего
    повлияли на предсказанный класс. Объясняет именно компонент LogReg —
    у CatBoost поверх эмбеддингов такого прямого объяснения через слова нет,
    поэтому в ансамбле пояснение всегда берётся из этой модели.

    Принимает как обычный Pipeline (текущая схема, без калибровки), так и
    CalibratedClassifierCV (на случай, если калибровку когда-то вернут) —
    определяется по наличию атрибута calibrated_classifiers_.
    """
    if hasattr(pipeline_or_calibrated, 'calibrated_classifiers_'):
        inner_pipeline = pipeline_or_calibrated.calibrated_classifiers_[0].estimator
    else:
        inner_pipeline = pipeline_or_calibrated

    vectorizer = inner_pipeline.named_steps['tfidf']
    clf = inner_pipeline.named_steps['clf']

    x_vec = vectorizer.transform([raw_text])
    pred_class = clf.predict(x_vec)[0]
    class_idx = list(clf.classes_).index(pred_class)

    feature_names, show_mask = _word_features_only(vectorizer)
    coefs = clf.coef_[class_idx]
    x_arr = x_vec.toarray().ravel()
    contribution = np.where(show_mask, coefs * x_arr, 0.0)
    top_idx = np.argsort(contribution)[::-1][:top_n]
    top_idx = [i for i in top_idx if contribution[i] > 0]

    print(f"\nПредсказанный класс (компонент LogReg): {pred_class}")
    print("Слова/n-граммы, сильнее всего повлиявшие на решение:")
    for i in top_idx:
        print(f"  {feature_names[i]!r}: вклад {contribution[i]:.3f}")
    return pred_class, [(feature_names[i], float(contribution[i])) for i in top_idx]


# ---------------------------------------------------------------------------
# Модель 1: TF-IDF + логистическая регрессия
# ---------------------------------------------------------------------------

def build_primary_pipeline():
    """TF-IDF по словам (1-2 граммы) + TF-IDF по символьным n-граммам (char_wb 2-5) -> LogReg.

    Вариант B из эксперимента с признаками текста: символьные n-граммы дают устойчивость к падежам,
    опечаткам и вариантам написания. Имя шага 'tfidf' сохранено (на него опираются объяснения)."""
    return Pipeline([
        ('tfidf', FeatureUnion([
            ('word', TfidfVectorizer(**config.TFIDF_PARAMS)),
            ('char', TfidfVectorizer(**config.CHAR_TFIDF_PARAMS)),
        ])),
        ('clf', LogisticRegression(**config.LOGREG_PARAMS)),
    ])


def cross_validate_primary(X_train_pool, y_train_pool):
    """5-fold CV на тренировочном пуле — тест сюда не передаётся вообще."""
    skf = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)
    fold_scores = []
    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train_pool, y_train_pool), 1):
        pipe = build_primary_pipeline()
        pipe.fit(X_train_pool.iloc[tr_idx], y_train_pool.iloc[tr_idx])
        pred = pipe.predict(X_train_pool.iloc[val_idx])
        score = f1_score(y_train_pool.iloc[val_idx], pred, average='macro')
        fold_scores.append(score)
        print(f"  Fold {fold_idx}/{config.N_SPLITS}: F1-macro = {score:.4f}")
    fold_scores = np.array(fold_scores)
    print(f"  CV F1-macro (LogReg): {fold_scores.mean():.4f} ± {fold_scores.std():.4f}")
    return fold_scores


def get_oof_probabilities_primary(X_train_pool, y_train_pool):
    """OOF-вероятности LogReg для всего train-пула (нужны для честного
    подбора порога/веса ансамбля, не глядя на тест).

    Раньше здесь стояла обёртка CalibratedClassifierCV — убрана: на классах
    с 40-60 примерами она систематически занижала вероятность редких
    классов (см. историю изменений в начале файла). cross_val_predict сам
    по себе уже даёт честные OOF-предсказания без утечки, калибровка
    поверх него на такой выборке приносила больше вреда, чем пользы.
    """
    return cross_val_predict(
        build_primary_pipeline(), X_train_pool, y_train_pool, cv=config.N_SPLITS, method='predict_proba'
    )


# ---------------------------------------------------------------------------
# Модель 2: эмбеддинги + CatBoost
# ---------------------------------------------------------------------------

def cross_validate_and_select_iterations_catboost(X_train_pool_emb, y_train_pool_arr):
    """5-fold CV на уже готовых эмбеддингах. Early stopping — по val-фолду
    (часть тренировочного пула), не по тесту. Возвращает CV F1 по фолдам и
    итоговое число итераций (медиана best_iteration)."""
    skf = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)
    fold_f1s, best_iterations = [], []
    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train_pool_emb, y_train_pool_arr), 1):
        clf = CatBoostClassifier(
            iterations=config.CATBOOST_MAX_ITERATIONS, verbose=False, **config.CATBOOST_PARAMS
        )
        clf.fit(
            X_train_pool_emb[tr_idx], y_train_pool_arr[tr_idx],
            eval_set=(X_train_pool_emb[val_idx], y_train_pool_arr[val_idx]),
            use_best_model=True,
            early_stopping_rounds=config.CATBOOST_EARLY_STOPPING_ROUNDS,
            verbose=False,
        )
        best_iterations.append(clf.get_best_iteration() or config.CATBOOST_MAX_ITERATIONS)
        pred = clf.predict(X_train_pool_emb[val_idx]).ravel()
        score = f1_score(y_train_pool_arr[val_idx], pred, average='macro')
        fold_f1s.append(score)
        print(f"  Fold {fold_idx}/{config.N_SPLITS}: F1-macro={score:.4f}, best_iteration={best_iterations[-1]}")

    fold_f1s = np.array(fold_f1s)
    print(f"  CV F1-macro (CatBoost): {fold_f1s.mean():.4f} ± {fold_f1s.std():.4f}")
    final_iterations = int(np.median(best_iterations))
    print(f"  Финальное число итераций (медиана по фолдам): {final_iterations}")
    return fold_f1s, final_iterations


def get_oof_probabilities_catboost(X_train_pool_emb, y_train_pool_arr, final_iterations):
    """OOF-вероятности CatBoost. Считается на уже готовых эмбеддингах (без
    повторного прогона через SentenceTransformer на каждом фолде), поэтому
    остаётся быстрым — это просто фиты деревьев, не нейросети.

    Без внешней CalibratedClassifierCV — по той же причине, что и для
    LogReg выше (см. историю изменений в начале файла)."""
    base_clf = CatBoostClassifier(iterations=final_iterations, verbose=False, **config.CATBOOST_PARAMS)
    return cross_val_predict(
        base_clf, X_train_pool_emb, y_train_pool_arr, cv=config.N_SPLITS, method='predict_proba'
    )


# ---------------------------------------------------------------------------
# Проверка на обращениях вне топ-15
# ---------------------------------------------------------------------------

def evaluate_out_of_scope(out_of_scope_df, threshold, predict_proba_fn):
    """predict_proba_fn(texts) -> proba, уже выровненная по общему списку
    классов. Работает как для одиночной модели, так и для ансамбля —
    сама логика комбинирования вероятностей передаётся снаружи."""
    if out_of_scope_df is None or len(out_of_scope_df) == 0:
        print("\n  Нет данных вне топ-15 для проверки (см. предупреждение выше).")
        return
    proba = predict_proba_fn(out_of_scope_df['clean_text'])
    max_conf = proba.max(axis=1)
    below_threshold = (max_conf < threshold).mean()
    print(
        f"\n  Обращений вне топ-15: {len(out_of_scope_df)} ({out_of_scope_df['target_class'].nunique()} видов)\n"
        f"  Доля с уверенностью ниже порога {threshold:.2f} "
        f"(корректно уходят на ручной разбор): {below_threshold:.1%}\n"
        f"  Доля с уверенностью >= порога (модель уверенно, но неизбежно ошибочно "
        f"относит их к одному из 15 знакомых видов): {1 - below_threshold:.1%}"
    )


# ---------------------------------------------------------------------------
# Основной сценарий обучения
# ---------------------------------------------------------------------------

def train_classifier():
    print("Шаг 1: Загрузка данных...")
    df, out_of_scope_df = load_datasets()
    classes = np.sort(df['target_class'].unique())

    print("Шаг 2: Замораживаем тест (15%) — дальше он используется только один раз, в самом конце...")
    train_pool_df, test_df = train_test_split(
        df, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE,
        stratify=df['target_class'],
    )
    print(f"  train-пул: {len(train_pool_df)} записей, тест: {len(test_df)} записей")

    X_train_pool_text = train_pool_df['clean_text']
    y_train_pool = train_pool_df['target_class']
    y_train_pool_arr = np.asarray(y_train_pool)
    X_test_text = test_df['clean_text']
    y_test_arr = np.asarray(test_df['target_class'])

    # ------------------------------------------------------------------
    # Модель 1: TF-IDF + LogReg
    # ------------------------------------------------------------------
    print("\n=== Модель 1: TF-IDF + логистическая регрессия ===")
    print("5-fold CV на train-пуле...")
    cross_validate_primary(X_train_pool_text, y_train_pool)

    print("\nOOF-вероятности LogReg (без калибровки — см. историю изменений)...")
    primary_oof_proba = get_oof_probabilities_primary(X_train_pool_text, y_train_pool)
    evaluate_top_k(primary_oof_proba, y_train_pool_arr, classes, k=config.TOP_K_ALTERNATIVES, label="LogReg OOF")

    print("\nОбучаем финальную LogReg-модель на всём train-пуле...")
    primary_final = build_primary_pipeline()
    primary_final.fit(X_train_pool_text, y_train_pool)
    primary_test_proba = _align_proba_columns(
        primary_final.predict_proba(X_test_text), primary_final.classes_, classes
    )
    test_f1_primary = f1_score(y_test_arr, classes[np.argmax(primary_test_proba, axis=1)], average='macro')
    print(f"  LogReg — F1-macro на тесте: {test_f1_primary:.4f}")

    if not _HAS_COMPARISON_DEPS:
        print(
            "\n[!] catboost/sentence-transformers не установлены в этом окружении — "
            "ансамбль недоступен, работаем только с LogReg. Установи их "
            "(`pip install catboost sentence-transformers`) и запусти заново, "
            "чтобы получить ансамбль."
        )
        threshold = pick_threshold(
            primary_oof_proba, y_train_pool_arr, classes, config.TARGET_AUTO_DECISION_PRECISION, label="LogReg"
        )
        coverage, precision = coverage_precision_at(primary_test_proba, y_test_arr, classes, threshold)
        print(f"\n  При пороге {threshold:.2f} на тесте: доля автоматических решений "
              f"{coverage:.1%}, точность на них {precision:.1%}")

        bundle = {
            'model_type': 'tfidf+logreg (raw predict_proba, без калибровки)',
            'primary_model': primary_final,
            'classes_': classes,
            'threshold': threshold,
            'top_k_alternatives': config.TOP_K_ALTERNATIVES,
            'preprocessing': 'data_preparation.smart_clean (v2)',
        }
        joblib.dump(bundle, config.PRIMARY_MODEL_BUNDLE_PATH)
        print(f"\nАртефакт для инференса сохранён: {config.PRIMARY_MODEL_BUNDLE_PATH}")

        evaluate_out_of_scope(
            out_of_scope_df, threshold,
            lambda texts: _align_proba_columns(primary_final.predict_proba(texts), primary_final.classes_, classes),
        )
        explain_prediction(primary_final, X_test_text.iloc[0])
        return bundle

    # ------------------------------------------------------------------
    # Модель 2: эмбеддинги + CatBoost
    # ------------------------------------------------------------------
    print("\n=== Модель 2: эмбеддинги + CatBoost ===")
    embedder = SentenceTransformer(config.EMBEDDER_MODEL_NAME)

    print("Считаем эмбеддинги train-пула и теста (один раз, дальше переиспользуются)...")
    X_train_pool_emb = np.asarray(embedder.encode(list(X_train_pool_text), show_progress_bar=True))
    X_test_emb = np.asarray(embedder.encode(list(X_test_text), show_progress_bar=True))

    print("5-fold CV на train-пуле (подбор числа итераций)...")
    _, cb_final_iterations = cross_validate_and_select_iterations_catboost(X_train_pool_emb, y_train_pool_arr)

    print("\nOOF-вероятности CatBoost (без калибровки — см. историю изменений)...")
    catboost_oof_proba = get_oof_probabilities_catboost(
        X_train_pool_emb, y_train_pool_arr, cb_final_iterations
    )
    catboost_oof_proba = _align_proba_columns(catboost_oof_proba, np.sort(np.unique(y_train_pool_arr)), classes)
    evaluate_top_k(catboost_oof_proba, y_train_pool_arr, classes, k=config.TOP_K_ALTERNATIVES, label="CatBoost OOF")

    print("\nОбучаем финальный CatBoost на всём train-пуле...")
    catboost_final = CatBoostClassifier(iterations=cb_final_iterations, verbose=False, **config.CATBOOST_PARAMS)
    catboost_final.fit(X_train_pool_emb, y_train_pool_arr)
    catboost_test_proba = _align_proba_columns(
        catboost_final.predict_proba(X_test_emb), catboost_final.classes_, classes
    )
    test_f1_catboost = f1_score(y_test_arr, classes[np.argmax(catboost_test_proba, axis=1)], average='macro')
    print(f"  CatBoost — F1-macro на тесте: {test_f1_catboost:.4f}")

    # ------------------------------------------------------------------
    # Ансамбль
    # ------------------------------------------------------------------
    print("\n=== Ансамбль: усреднение вероятностей ===")
    ensemble_weight, ensemble_oof_f1 = pick_ensemble_weight(
        primary_oof_proba, catboost_oof_proba, y_train_pool_arr, classes, config.ENSEMBLE_WEIGHT_GRID
    )
    ensemble_oof_proba = ensemble_weight * primary_oof_proba + (1 - ensemble_weight) * catboost_oof_proba
    evaluate_top_k(ensemble_oof_proba, y_train_pool_arr, classes, k=config.TOP_K_ALTERNATIVES, label="Ensemble OOF")
    threshold = pick_threshold(
        ensemble_oof_proba, y_train_pool_arr, classes, config.TARGET_AUTO_DECISION_PRECISION, label="Ensemble"
    )

    print("\nШаг: ЕДИНСТВЕННАЯ проверка на замороженном тесте (LogReg vs CatBoost vs Ансамбль)...")
    ensemble_test_proba = ensemble_weight * primary_test_proba + (1 - ensemble_weight) * catboost_test_proba
    ensemble_test_pred = classes[np.argmax(ensemble_test_proba, axis=1)]
    test_f1_ensemble = f1_score(y_test_arr, ensemble_test_pred, average='macro')

    print(
        f"  F1-macro на тесте — LogReg: {test_f1_primary:.4f} | "
        f"CatBoost: {test_f1_catboost:.4f} | Ансамбль: {test_f1_ensemble:.4f}"
    )
    print(classification_report(y_test_arr, ensemble_test_pred, zero_division=0))

    evaluate_top_k(ensemble_test_proba, y_test_arr, classes, k=config.TOP_K_ALTERNATIVES, label="Ensemble test")
    coverage, precision = coverage_precision_at(ensemble_test_proba, y_test_arr, classes, threshold)
    print(f"  При пороге {threshold:.2f} на тесте: доля автоматических решений "
          f"{coverage:.1%}, точность на них {precision:.1%}")

    bundle = {
        'model_type': f'ensemble(w_logreg={ensemble_weight:.2f}, w_catboost={1 - ensemble_weight:.2f}), raw predict_proba (без калибровки)',
        'primary_model': primary_final,     # Pipeline(tfidf+logreg) — принимает сырой текст
        'catboost_model': catboost_final,   # CatBoostClassifier — принимает ЭМБЕДДИНГИ, не текст
        'embedder_name': config.EMBEDDER_MODEL_NAME,
        'ensemble_weight_logreg': ensemble_weight,
        'classes_': classes,
        'threshold': threshold,
        'top_k_alternatives': config.TOP_K_ALTERNATIVES,
        'preprocessing': 'data_preparation.smart_clean (v2)',
    }
    joblib.dump(bundle, config.ENSEMBLE_BUNDLE_PATH)
    print(f"\nАртефакт ансамбля для инференса сохранён: {config.ENSEMBLE_BUNDLE_PATH}")

    def _ensemble_predict_proba(texts):
        texts = list(texts)
        p_proba = _align_proba_columns(primary_final.predict_proba(texts), primary_final.classes_, classes)
        emb = np.asarray(embedder.encode(texts, show_progress_bar=True))
        c_proba = _align_proba_columns(catboost_final.predict_proba(emb), catboost_final.classes_, classes)
        return ensemble_weight * p_proba + (1 - ensemble_weight) * c_proba

    print("\nПроверка на обращениях ВНЕ топ-15 (по ансамблю)...")
    evaluate_out_of_scope(out_of_scope_df, threshold, _ensemble_predict_proba)

    print("\nПример объяснимости (компонент LogReg ансамбля)...")
    explain_prediction(primary_final, X_test_text.iloc[0])

    return bundle


if __name__ == "__main__":
    train_classifier()