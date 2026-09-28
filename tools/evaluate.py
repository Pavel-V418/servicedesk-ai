"""
Проверка решения для отчёта: считает все цифры на обученных моделях и
складывает их в reports/eval/.

Запуск из корня ServiceDeskAI (после python main.py train):
    python tools/evaluate.py

Что считается (везде указаны объём выборки и способ расчёта):
  1. Данные: 15 видов и число записей, дубликаты, пропуски.
  2. Категоризация: замороженный тест 15% (стратифицированно, random_state=42,
     тот же сплит, что в classifier/model.py), метрики, путаница, примеры ошибок,
     смысл уверенности (точность по диапазонам), обращения вне топ-15.
  3. Маршрутизация: финальный временной тест 20% (последние по дате, как в
     routing_agent/src/models/final_evaluation.py) в трёх режимах входа.
  4. Поиск похожих: запросы = обращения замороженного теста, из выдачи исключены
     само обращение и его точные текстовые дубликаты.
  5. SLA: показатели дашборда, влияние фильтров, проверка уровней риска на
     временном тесте (история — только первые 80%).
"""
from __future__ import annotations

import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from classifier import config as clf_config  # noqa: E402
from classifier.data_preparation import _drop_duplicate_and_conflicting_texts, smart_clean  # noqa: E402

OUT = ROOT / "reports" / "eval"
OUT.mkdir(parents=True, exist_ok=True)
RESULTS: dict = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S")}


def log(msg):
    print(msg, flush=True)


def _s(x, n=300):
    x = "" if x is None or (isinstance(x, float) and np.isnan(x)) else str(x)
    x = " ".join(x.split())
    return x if len(x) <= n else x[:n] + "…"


def _cm_pairs(y_true, y_pred, top=10):
    pairs = pd.Series(list(zip(y_true, y_pred)))
    pairs = pairs[[a != b for a, b in pairs]].value_counts().head(top)
    support = pd.Series(y_true).value_counts()
    return [{"истина": a, "предсказано": b, "n": int(n), "из_истины": int(support[a])}
            for (a, b), n in pairs.items()]


# ---------------------------------------------------------------------------
# 1. Данные
# ---------------------------------------------------------------------------
def dataset_section():
    log("1/5 Данные...")
    raw = pd.read_excel(clf_config.RAW_XLSX_PATH, sheet_name="Sheet0")
    top15 = raw["Вид запроса"].value_counts().nlargest(15)
    cleaned = pd.read_csv(clf_config.CLEANED_CSV_PATH)
    after = cleaned["target_class"].value_counts()

    in_scope = raw[raw["Вид запроса"].isin(top15.index)].copy()
    in_scope["clean_text"] = in_scope.apply(smart_clean, axis=1)
    in_scope["target_class"] = in_scope["Вид запроса"]
    _, dedup = _drop_duplicate_and_conflicting_texts(in_scope)

    RESULTS["dataset"] = {
        "rows": int(len(raw)), "kinds": int(raw["Вид запроса"].nunique()),
        "period": [str(raw["Дата регистрации"].min())[:10], str(raw["Дата регистрации"].max())[:10]],
        "lines": raw["Кем решен (группа)"].value_counts().to_dict(),
        "top15": [{"вид": k, "записей": int(v), "после_очистки": int(after.get(k, 0))}
                  for k, v in top15.items()],
        "top15_total": int(top15.sum()), "top15_total_after": int(after.sum()),
        "out_of_scope_rows": int(len(raw) - top15.sum()),
        "dedup": dedup,
        "missing": raw.isna().sum()[lambda s: s > 0].astype(int).to_dict(),
    }


# ---------------------------------------------------------------------------
# 2. Категоризация
# ---------------------------------------------------------------------------
def classifier_section():
    log("2/5 Категоризация (замороженный тест)...")
    from classifier.predictor import TicketClassifier
    clf = TicketClassifier()
    df = pd.read_csv(clf_config.CLEANED_CSV_PATH)
    df["clean_text"] = df["clean_text"].fillna("")
    train_df, test_df = train_test_split(df, test_size=clf_config.TEST_SIZE,
                                         random_state=clf_config.RANDOM_STATE, stratify=df["target_class"])

    rows = []
    for _, r in test_df.iterrows():
        res = clf.classify(r.get("Описание 2", ""), r.get("Услуга", ""),
                           r.get("Компонент услуги 1 уровня", ""), r.get("Тип запроса", ""))
        top = [c for c, _ in res["top_k"]]
        rows.append({"id": r["Номер запроса"], "true": r["target_class"], "pred": res["predicted_class"],
                     "conf": res["confidence"], "confident": res["is_confident"], "in_top3": r["target_class"] in top,
                     "top3": res["top_k"], "words": [w for w, _ in res["explanation_words"][:5]],
                     "text": _s(r.get("Описание 2", ""), 400), "service": _s(r.get("Услуга", ""), 60)})
    p = pd.DataFrame(rows)
    p.to_csv(OUT / "classifier_test_predictions.csv", index=False, encoding="utf-8-sig")
    y, yhat = p["true"].tolist(), p["pred"].tolist()
    thr = clf.threshold

    def bucket(lo, hi):
        m = (p["conf"] >= lo) & (p["conf"] < hi)
        return {"диапазон": f"{lo:.2f}–{hi:.2f}" if hi < 1.01 else f"≥ {lo:.2f}",
                "доля": round(float(m.mean()), 3), "n": int(m.sum()),
                "точность": round(float((p.loc[m, "true"] == p.loc[m, "pred"]).mean()), 3) if m.any() else None}

    rep = classification_report(y, yhat, output_dict=True, zero_division=0)
    per_class = [{"вид": k, "precision": round(v["precision"], 3), "recall": round(v["recall"], 3),
                  "f1": round(v["f1-score"], 3), "support": int(v["support"])}
                 for k, v in rep.items() if k not in ("accuracy", "macro avg", "weighted avg")]
    errors = p[p["true"] != p["pred"]].sort_values("conf", ascending=False)

    # Вне топ-15
    oos = pd.read_csv(clf_config.OUT_OF_SCOPE_CSV_PATH)
    oos_conf, oos_rows = [], []
    for _, r in oos.iterrows():
        res = clf.classify(r.get("Описание 2", ""), r.get("Услуга", ""),
                           r.get("Компонент услуги 1 уровня", ""), r.get("Тип запроса", ""))
        oos_conf.append(res["confidence"])
        oos_rows.append({"true": r["target_class"], "pred": res["predicted_class"], "conf": res["confidence"],
                         "text": _s(r.get("Описание 2", ""), 250)})
    oos_conf = np.array(oos_conf)
    oos_df = pd.DataFrame(oos_rows)

    RESULTS["classifier"] = {
        "model_type": clf.model_type, "threshold": thr,
        "train_pool": int(len(train_df)), "test": int(len(test_df)),
        "accuracy": round(accuracy_score(y, yhat), 4),
        "macro_f1": round(f1_score(y, yhat, average="macro"), 4),
        "weighted_f1": round(f1_score(y, yhat, average="weighted"), 4),
        "top3": round(float(p["in_top3"].mean()), 4),
        "coverage_at_thr": round(float(p["confident"].mean()), 4),
        "precision_at_thr": round(float((p.loc[p["confident"], "true"] == p.loc[p["confident"], "pred"]).mean()), 4),
        "buckets": [bucket(thr, 1.01), bucket(0.30, thr), bucket(0.0, 0.30)],
        "per_class": sorted(per_class, key=lambda d: d["f1"]),
        "confusions": _cm_pairs(y, yhat),
        "errors_high_conf": errors.head(6)[["id", "true", "pred", "conf", "top3", "words", "text"]].to_dict("records"),
        "errors_low_conf": errors.tail(3)[["id", "true", "pred", "conf", "top3", "words", "text"]].to_dict("records"),
        "correct_examples": p[(p["true"] == p["pred"]) & p["confident"]].head(3)[
            ["id", "true", "conf", "words", "text"]].to_dict("records"),
        "out_of_scope": {
            "n": int(len(oos)), "kinds": int(oos["target_class"].nunique()),
            "below_thr_share": round(float((oos_conf < thr).mean()), 4),
            "mean_conf": round(float(oos_conf.mean()), 3),
            "in_scope_test_below_thr_share": round(float((p["conf"] < thr).mean()), 4),
            "examples_confident": oos_df[oos_df["conf"] >= thr].head(3).to_dict("records"),
            "examples_uncertain": oos_df[oos_df["conf"] < thr].head(2).to_dict("records"),
        },
    }
    return clf


# ---------------------------------------------------------------------------
# 3. Маршрутизация
# ---------------------------------------------------------------------------
def routing_section(clf):
    log("3/5 Маршрутизация (временной тест)...")
    from routing.router import TicketRouter, normalize_priority  # noqa: F401  (добавляет routing_agent в sys.path)
    from src.features.feature_builder import prepare_features
    from src.models.train_router import CLASSES, DATE_COL, TARGET, normalize_target

    router = TicketRouter()
    model, rec = router.agent.model, router.agent.recommendator
    df = pd.read_excel(clf_config.RAW_XLSX_PATH, sheet_name="Sheet0")
    df["line_raw"] = df[TARGET]
    df[TARGET] = normalize_target(df[TARGET])
    df = df[df[TARGET].isin(CLASSES)].copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    split = int(len(df) * 0.8)
    train, test = df.iloc[:split].copy(), df.iloc[split:].copy()

    # вид запроса от классификатора — как в продакшене
    test["pred_kind"] = [clf.classify(r.get("Описание 2", ""), r.get("Услуга", ""),
                                      r.get("Компонент услуги 1 уровня", ""), r.get("Тип запроса", ""))["predicted_class"]
                         for _, r in test.iterrows()]
    ui_hidden = ["Пользователь", "Часовой пояс запроса", "Класс обслуживания", "Критичность", "Срочность"]

    def run(t, name):
        proba = model.predict_proba(prepare_features(t))
        classes = model.classes_
        pred = classes[proba.argmax(1)]
        y = t[TARGET].values
        statuses = [rec.recommend(dict(zip(classes, pr))).status for pr in proba]
        st_df = pd.DataFrame({"status": statuses, "ok": pred == y})
        cm = confusion_matrix(y, pred, labels=CLASSES)
        return {
            "режим": name, "n": int(len(t)),
            "accuracy": round(accuracy_score(y, pred), 4),
            "macro_f1": round(f1_score(y, pred, average="macro", labels=CLASSES, zero_division=0), 4),
            "macro_f1_L1_L2": round(f1_score(y, pred, average="macro", labels=["L1", "L2"], zero_division=0), 4),
            "per_line": {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in
                         classification_report(y, pred, labels=CLASSES, output_dict=True, zero_division=0).items()
                         if k in CLASSES},
            "confusion": {"labels": CLASSES, "matrix": cm.tolist()},
            "statuses": [{"статус": s, "доля": round(float((st_df.status == s).mean()), 3),
                          "точность": round(float(st_df.loc[st_df.status == s, "ok"].mean()), 3)
                          if (st_df.status == s).any() else None}
                         for s in ["high_confidence", "medium_confidence", "manual_review"]],
        }, pred

    a, pred_a = run(test, "все поля, истинный вид запроса (как у автора маршрутизатора)")
    t_b = test.copy(); t_b["Вид запроса"] = t_b["pred_kind"]; t_b[ui_hidden] = None
    b, pred_b = run(t_b, "поля формы UI, вид запроса от классификатора (как в сервисе)")
    t_c = test.copy(); t_c["Вид запроса"] = None; t_c[ui_hidden] = None
    c, _ = run(t_c, "поля формы UI, без вида запроса")

    # несколько линий
    work_cols = [f"Суммарное время работы {i} линии" for i in (1, 2, 3)]
    n_lines = test[work_cols].astype(str).apply(lambda r: sum(v not in ("00:00:00", "nan", "0") for v in r), axis=1)
    multi = n_lines > 1

    test["pred_b"] = pred_b
    err = test[test["pred_b"] != test[TARGET]]
    RESULTS["routing"] = {
        "train": int(len(train)), "test": int(len(test)),
        "test_period": [str(test[DATE_COL].min())[:10], str(test[DATE_COL].max())[:10]],
        "test_lines": test[TARGET].value_counts().to_dict(),
        "modes": [a, b, c],
        "baseline_majority_acc": round(float((test[TARGET] == train[TARGET].mode()[0]).mean()), 4),
        "multi_line": {"share": round(float(multi.mean()), 3), "n": int(multi.sum()),
                       "acc_multi": round(float((pred_a[multi.values] == test.loc[multi, TARGET].values).mean()), 3),
                       "acc_single": round(float((pred_a[~multi.values] == test.loc[~multi, TARGET].values).mean()), 3)},
        "classifier_train_overlap_note": "классификатор обучен на случайном сплите: часть обращений временного теста "
                                         "маршрутизатора была в его обучении — режим B может быть оптимистичен",
        "errors": err[[ "Номер запроса", TARGET, "pred_b", "pred_kind", "Вид запроса", "Приоритет"]].assign(
            text=err["Описание 2"].map(lambda x: _s(x, 300))).head(6).rename(
            columns={TARGET: "true_line", "Номер запроса": "id", "Вид запроса": "true_kind"}).to_dict("records"),
    }
    RESULTS["_routing_test_frame"] = test  # для SLA-проверки ниже (в json не пишется)
    RESULTS["_routing_train_frame"] = train


# ---------------------------------------------------------------------------
# 4. Поиск похожих
# ---------------------------------------------------------------------------
def rag_section():
    log("4/5 Поиск похожих (RAG)...")
    from RagAgent.rag_build_db import clean_ticket_text
    from RagAgent.rag_search import TicketRetriever
    retr = TicketRetriever(db_path=config.CHROMA_DB_PATH, distance_threshold=config.RAG_DISTANCE_THRESHOLD)
    coll = retr.collection

    df = pd.read_csv(clf_config.CLEANED_CSV_PATH)
    _, test_df = train_test_split(df, test_size=clf_config.TEST_SIZE,
                                  random_state=clf_config.RANDOM_STATE, stratify=df["target_class"])
    raw = pd.read_excel(clf_config.RAW_XLSX_PATH, sheet_name="Sheet0")
    kind_share = raw["Вид запроса"].value_counts(normalize=True)
    line_share = raw["Кем решен (группа)"].value_counts(normalize=True)

    recs, samples = [], []
    for _, r in test_df.iterrows():
        q = clean_ticket_text(r.get("Описание 2", ""))
        if not q:
            continue
        res = coll.query(query_texts=[q], n_results=15)
        found = []
        for i, d, m, doc in zip(res["ids"][0], res["distances"][0], res["metadatas"][0], res["documents"][0]):
            if str(i) == str(r["Номер запроса"]) or doc == q:   # само обращение и его точные дубли
                continue
            if m.get("линия") == "(4 линия)":
                continue
            if d <= config.RAG_DISTANCE_THRESHOLD:
                found.append((i, d, m, doc))
            if len(found) >= config.RAG_TOP_K:
                break
        true_kind, true_line = r["target_class"], r["Кем решен (группа)"]
        rec = {"id": r["Номер запроса"], "found": len(found)}
        if found:
            rec.update({
                "d1": found[0][1],
                "kind@1": found[0][2]["вид_запроса"] == true_kind,
                "kind_any@3": any(f[2]["вид_запроса"] == true_kind for f in found),
                "kind_share@3": np.mean([f[2]["вид_запроса"] == true_kind for f in found]),
                "line@1": found[0][2]["линия"] == true_line,
                "status@1": found[0][2]["результат"],
            })
            if len(samples) < 40:
                samples.append({"query_id": r["Номер запроса"], "query_kind": true_kind, "query_line": true_line,
                                "query": _s(r.get("Описание 2", ""), 350),
                                "hit_id": found[0][0], "hit_kind": found[0][2]["вид_запроса"],
                                "hit_line": found[0][2]["линия"], "hit_status": found[0][2]["результат"],
                                "distance": round(found[0][1], 3), "hit": _s(found[0][2].get("оригинал") or found[0][3], 350)})
        recs.append(rec)
    R = pd.DataFrame(recs)
    F = R[R["found"] > 0]

    def by_dist(lo, hi):
        m = (F["d1"] > lo) & (F["d1"] <= hi)
        return {"расстояние": f"{lo}–{hi}", "n": int(m.sum()),
                "вид_совпал@1": round(float(F.loc[m, "kind@1"].mean()), 3) if m.any() else None}

    pd.DataFrame(samples).to_csv(OUT / "rag_samples.csv", index=False, encoding="utf-8-sig")
    RESULTS["rag"] = {
        "queries": int(len(R)), "db_size": int(coll.count()),
        "threshold": config.RAG_DISTANCE_THRESHOLD, "top_k": config.RAG_TOP_K,
        "found_share": round(float((R["found"] > 0).mean()), 3),
        "mean_found": round(float(R["found"].mean()), 2),
        "kind@1": round(float(F["kind@1"].mean()), 3),
        "kind_any@3": round(float(F["kind_any@3"].mean()), 3),
        "kind_precision@3": round(float(F["kind_share@3"].mean()), 3),
        "line@1": round(float(F["line@1"].mean()), 3),
        "random_kind_baseline": round(float((kind_share ** 2).sum()), 3),
        "random_line_baseline": round(float((line_share ** 2).sum()), 3),
        "by_distance": [by_dist(0, 0.25), by_dist(0.25, 0.35), by_dist(0.35, 0.45)],
        "status@1": F["status@1"].value_counts().to_dict(),
        "samples": samples[:40],
    }


# ---------------------------------------------------------------------------
# 5. SLA
# ---------------------------------------------------------------------------
def sla_section():
    log("5/5 SLA-аналитика...")
    from sla.risk import SLARiskEstimator
    from sla.sla_analytics import SLAAnalyzer
    an = SLAAnalyzer()
    d = an.df
    dash = json.loads(an.get_dashboard_data())
    filters = {}
    for name, flt in [("приоритет (1) Наивысший", {"priority": "(1) Наивысший"}),
                      ("приоритет (3) Средний", {"priority": "(3) Средний"})] + \
                     [(f"услуга {s}", {"service": s}) for s in d["Услуга"].dropna().unique()]:
        x = json.loads(an.get_dashboard_data(flt))
        if "error" not in x:
            filters[name] = {"total": x["summary"]["total_requests"], "overdue_pct": x["summary"]["overdue_rate_percent"],
                             "median_h": x["processing_time_hours"]["median_50_percent"],
                             "p90_h": x["processing_time_hours"]["percentile_90"],
                             "multi_line": x["summary"]["multi_line_tickets"]}

    by = lambda col: (d.groupby(col).agg(n=("is_overdue", "size"), overdue=("is_overdue", "sum"),
                                         median_h=("duration_hours", "median"),
                                         p90_h=("duration_hours", lambda s: s.quantile(0.9)))
                      .round(2).reset_index().to_dict("records"))

    # проверка уровней риска на временном тесте: история — только train-период
    test = RESULTS.get("_routing_test_frame")
    train = RESULTS.get("_routing_train_frame")
    risk_check = None
    if test is not None:
        hist = d[d["Номер запроса"].isin(train["Номер запроса"])]
        est = SLARiskEstimator(types.SimpleNamespace(df=hist))
        levels = [est.assess(priority=r["Приоритет"], line_code=r["pred_b"], category=r["pred_kind"])["sla_risk_level"]
                  for _, r in test.iterrows()]
        over = d.set_index("Номер запроса").loc[test["Номер запроса"], "is_overdue"].values
        t = pd.DataFrame({"level": levels, "overdue": over})
        risk_check = [{"уровень": lv, "n": int((t.level == lv).sum()), "просрочено": int(t.loc[t.level == lv, "overdue"].sum()),
                       "доля_просрочек": round(float(t.loc[t.level == lv, "overdue"].mean()), 3) if (t.level == lv).any() else None}
                      for lv in ["высокий", "средний", "низкий"]]

    RESULTS["sla"] = {
        "dashboard": dash, "filters": filters,
        "by_priority": by("Приоритет"), "by_line": by("Кем решен (группа)"), "by_service": by("Услуга"),
        "clarifications_dist": d["Количество уточнений"].value_counts().sort_index().to_dict(),
        "zero_duration": int((d["duration_hours"] == 0).sum()),
        "risk_check_on_temporal_test": risk_check,
    }


def main():
    t0 = time.time()
    dataset_section()
    clf = classifier_section()
    routing_section(clf)
    try:
        rag_section()
    except Exception as exc:  # база не построена / chromadb недоступен
        RESULTS["rag"] = {"error": repr(exc)}
        log(f"[!] RAG пропущен: {exc!r}")
    sla_section()
    for k in [k for k in RESULTS if k.startswith("_")]:
        RESULTS.pop(k)
    RESULTS["runtime_sec"] = round(time.time() - t0, 1)
    with open(OUT / "results.json", "w", encoding="utf-8") as f:
        json.dump(RESULTS, f, ensure_ascii=False, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    c, r = RESULTS["classifier"], RESULTS["routing"]["modes"][0]
    log(f"\nГотово за {RESULTS['runtime_sec']} с → {OUT}")
    log(f"Категоризация: test={c['test']}, accuracy={c['accuracy']}, macro-F1={c['macro_f1']}, top-3={c['top3']}")
    log(f"Маршрутизация: test={r['n']}, accuracy={r['accuracy']}, macro-F1={r['macro_f1']}")
    if "error" not in RESULTS["rag"]:
        g = RESULTS["rag"]
        log(f"Поиск: запросов={g['queries']}, найдено хоть что-то={g['found_share']}, вид совпал@1={g['kind@1']}")


if __name__ == "__main__":
    main()
