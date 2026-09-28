from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Если файл запустили обычным `python ui/app.py` (например кнопкой Run в
# PyCharm), Streamlit работает в "bare mode": страница не открывается, а в
# консоль сыплются предупреждения "missing ScriptRunContext". В этом случае
# перезапускаем себя правильно — через `streamlit run`.
if __name__ == "__main__":
    from streamlit import runtime

    if not runtime.exists():
        import subprocess
        sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", __file__, *sys.argv[1:]]))

# set_page_config должен быть первой командой Streamlit — до загрузки моделей.
st.set_page_config(page_title="Service Desk AI", page_icon="🤖", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from classifier import config as clf_config  # noqa: E402

# Поля формы -> колонки выгрузки. Значения для списков берутся из той же
# выгрузки, на которой учились модели: модели не узнают значения, которых не
# видели (например "Средний" вместо "(3) Средний"), и молча их игнорируют.
# В форме только то, что оператор знает при регистрации обращения.
# Вид запроса определяет классификатор. Пользователь, часовой пояс, класс
# обслуживания, критичность и срочность из формы убраны: на качество
# маршрутизации они не влияют (проверено на финальном временном тесте:
# accuracy 0.689 со всеми полями и 0.689 без них).
FORM_COLUMNS = {
    "service": "Услуга",
    "component": "Компонент услуги 1 уровня",
    "request_type": "Тип запроса",
    "priority": "Приоритет",
}
EMPTY = ""
LINE_NAMES = {"L1": "1 линия", "L2": "2 линия", "L3": "3 линия"}


# ---------------------------------------------------------------------------
# Загрузка (кэшируется между перезапусками скрипта)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Загружаю модели (классификатор, маршрутизатор, RAG, SLA)...")
def load_pipeline():
    import pipeline
    return pipeline


@st.cache_data(show_spinner=False)
def load_options() -> dict[str, list[str]]:
    df = pd.read_excel(clf_config.RAW_XLSX_PATH, sheet_name="Sheet0")
    options = {}
    for key, col in FORM_COLUMNS.items():
        values = df[col].dropna().astype(str)
        # частые значения — вверху списка
        options[key] = values.value_counts().index.tolist()
    options["priority"] = sorted(options["priority"])  # (1) Наивысший ... (4) Низкий
    return options


@st.cache_resource(show_spinner="Загружаю историю для SLA-аналитики...")
def load_sla_analyzer():
    from sla.sla_analytics import SLAAnalyzer
    return SLAAnalyzer()


def _fmt_empty(placeholder: str):
    return lambda v: placeholder if v == EMPTY else v


# ---------------------------------------------------------------------------
# Вывод результата
# ---------------------------------------------------------------------------

def closeness_label(distance: float) -> str:
    """Расстояние ChromaDB — не процент, поэтому показываем словами.
    Поиск отдаёт только обращения с расстоянием <= 0.45 (порог RAG)."""
    if distance <= 0.25:
        return "очень похоже"
    if distance <= 0.35:
        return "похоже"
    return "отдалённо похоже"


def render_classification(result: dict):
    st.markdown("#### 1. Вид запроса")
    cls = result.get("predicted_class")
    if not cls:
        st.error("Классификатор не загружен — см. статус компонентов слева.")
        return
    conf = result.get("confidence", 0.0)
    threshold = result.get("classifier_threshold", 0.0)
    if result.get("is_confident"):
        st.success(f"**{cls}** — уверенность {conf:.0%}")
    else:
        st.warning(f"**{cls}** — уверенность {conf:.0%}, ниже порога {threshold:.0%}. "
                   "Проверьте альтернативы.")
    for name, p in result.get("top_k", []):
        st.progress(min(max(p, 0.0), 1.0), text=f"{name} — {p:.0%}")
    words = [w for w, _ in result.get("explanation_words", [])[:6]]
    if words:
        st.caption("Ключевые слова: " + ", ".join(words))


def render_routing(result: dict):
    st.markdown("#### 2. Линия поддержки")
    status = result.get("routing_status")
    if not status:
        st.error(result.get("routing_reason") or "Маршрутизатор не загружен.")
        return
    line = result.get("recommended_line")
    conf = result.get("routing_confidence") or 0.0
    if status == "manual_review":
        st.warning(f"Требуется ручная проверка — максимальная уверенность {conf:.0%}")
    elif status == "high_confidence":
        st.success(f"Рекомендуемая линия: **{LINE_NAMES.get(line, line)}** — высокая уверенность ({conf:.0%})")
    else:
        st.info(f"Рекомендуемая линия: **{LINE_NAMES.get(line, line)}** — средняя уверенность ({conf:.0%})")
    probs = result.get("routing_probabilities") or {}
    for code in ["L1", "L2", "L3"]:
        p = probs.get(code, 0.0)
        st.progress(min(max(p, 0.0), 1.0), text=f"{LINE_NAMES[code]} — {p:.0%}")
    source = result.get("request_kind_source")
    if source:
        st.caption(f"Вид запроса для маршрутизации указал: {source}")


def render_sla(result: dict):
    st.markdown("#### 3. Риск нарушения SLA")
    level = result.get("sla_risk_level")
    if not level:
        st.error(result.get("sla_risk_reason") or "SLA-блок не загружен.")
        return
    text = f"Риск: **{level}** ({result.get('sla_risk_score', 0):.1%} просрочек у похожих обращений)"
    {"высокий": st.error, "средний": st.warning}.get(level, st.success)(text)
    st.caption(result.get("sla_risk_reason", ""))
    exp = result.get("sla_expected_hours") or {}
    if exp:
        c1, c2 = st.columns(2)
        c1.metric("Типичное время решения (медиана)", f"{exp['median']} ч")
        c2.metric("90% обращений решены за", f"{exp['p90']} ч")
        st.caption(f"По истории: {exp['basis']}, {exp['n']} обращений")


def render_similar(result: dict):
    similar = result.get("similar_tickets") or []
    st.markdown(f"#### 4. Похожие обращения из истории ({len(similar)})")
    if not similar:
        st.caption("Достаточно похожих обращений не найдено (или RAG не загружен).")
        return
    for t in similar:
        header = (f"{t.get('ticket_id')} · {t.get('вид_запроса')} · {t.get('линия')} · "
                  f"{closeness_label(t.get('similarity_distance', 1.0))}")
        with st.expander(header, expanded=False):
            st.caption(f"Статус закрытия: {t.get('результат_работ')} · "
                       f"расстояние {t.get('similarity_distance')}")
            st.text(t.get("текст_полный") or t.get("текст") or "")


def render_summary(result: dict):
    """Итог для оператора от GigaChat (finalizer.py) — выводится потоком.
    Карточки выше к этому моменту уже на экране: пайплайн отдаёт их без итога
    (pipeline.analyze_ticket), а итог запрашивается здесь отдельно."""
    import finalizer

    st.markdown("#### 5. Итог для оператора")
    summary = st.session_state.get("summary")
    if summary is None:
        info: dict = {}
        with st.container(border=True):
            waiting = st.caption("Готовлю итог…")
            text = st.write_stream(finalizer.stream_final_answer(result, info))
            waiting.empty()
        summary = {"text": text if isinstance(text, str) else "".join(map(str, text)), **info}
        # Сохраняем, чтобы при следующих перезапусках скрипта Streamlit
        # (например, после раскрытия блока) не запрашивать GigaChat заново.
        st.session_state["summary"] = summary
    else:
        with st.container(border=True):
            st.markdown(summary["text"])
    if summary.get("method") == "llm":
        timing = ""
        if summary.get("total_sec") is not None:
            timing = f" · ответ за {summary['total_sec']} с"
        st.caption(f"Сформировано GigaChat{timing}. Проверяйте по карточкам выше: модель пересказывает, "
                   "а не принимает решения.")
    else:
        st.caption("Сформировано по шаблону (GigaChat недоступен — см. статус компонентов слева).")


def render_result(result: dict):
    render_classification(result)
    st.divider()
    render_routing(result)
    st.divider()
    render_sla(result)
    st.divider()
    render_similar(result)
    st.divider()
    render_summary(result)
    with st.expander("Техническая информация"):
        tech = dict(result)
        summary = st.session_state.get("summary") or {}
        tech["final_answer"] = summary.get("text")
        tech["synthesis_method"] = summary.get("method")
        tech["llm_timing_sec"] = {k: summary.get(k) for k in ("first_token_sec", "total_sec")}
        st.json(json.loads(json.dumps(tech, ensure_ascii=False, default=str)))


# ---------------------------------------------------------------------------
# Вкладка "Анализ обращения"
# ---------------------------------------------------------------------------

def analysis_tab(pipeline, options):
    left, right = st.columns([1, 1])

    with left:
        st.subheader("Новое обращение")
        with st.form("ticket_form"):
            description = st.text_area(
                "Описание обращения", height=180,
                placeholder="Например: Не загружается список отправлений...",
            )
            values = {}
            values["service"] = st.selectbox("Услуга", [EMPTY] + options["service"],
                                             format_func=_fmt_empty("— не указана —"))
            values["component"] = st.selectbox("Компонент услуги", [EMPTY] + options["component"],
                                               format_func=_fmt_empty("— не указан —"))
            values["request_type"] = st.selectbox("Тип запроса", [EMPTY] + options["request_type"],
                                                  format_func=_fmt_empty("— не указан —"))
            values["priority"] = st.selectbox(
                "Приоритет (если известен)", [EMPTY] + options["priority"],
                format_func=_fmt_empty("— не указан —"),
                help="Нужен только для оценки риска SLA: 24 из 25 просрочек в истории — "
                     "у обращений с приоритетом «Наивысший». Без приоритета риск "
                     "оценивается по виду запроса и линии.",
            )
            submitted = st.form_submit_button("Анализировать обращение", type="primary")

        if submitted:
            if not description.strip():
                st.error("Введите описание обращения.")
            else:
                with st.spinner("Анализирую..."):
                    # Результат храним в session_state: иначе он пропадёт при
                    # любом следующем перезапуске скрипта Streamlit.
                    # Без итога: он запрашивается у GigaChat отдельно, потоком (render_summary).
                    st.session_state["result"] = pipeline.analyze_ticket(description, **values)
                    st.session_state["summary"] = None

    with right:
        st.subheader("Результат анализа")
        result = st.session_state.get("result")
        if result is None:
            st.caption("Заполните форму и нажмите «Анализировать обращение».")
        else:
            render_result(result)


# ---------------------------------------------------------------------------
# Вкладка "SLA-аналитика"
# ---------------------------------------------------------------------------

def sla_tab(options):
    analyzer = load_sla_analyzer()
    st.subheader("SLA-аналитика по истории обращений")
    f1, f2 = st.columns(2)
    service = f1.selectbox("Услуга", [EMPTY] + options["service"],
                           format_func=_fmt_empty("Все услуги"), key="sla_service")
    priority = f2.selectbox("Приоритет", [EMPTY] + options["priority"],
                            format_func=_fmt_empty("Все приоритеты"), key="sla_priority")
    filters = {}
    if service:
        filters["service"] = service
    if priority:
        filters["priority"] = priority

    data = json.loads(analyzer.get_dashboard_data(filters or None))
    if "error" in data:
        st.warning(data["error"])
        return

    s, t = data["summary"], data["processing_time_hours"]
    m = st.columns(5)
    m[0].metric("Обращений", s["total_requests"])
    m[1].metric("Просрочено", f"{s['overdue_rate_percent']}%")
    m[2].metric("Медиана решения", f"{t['median_50_percent']} ч")
    m[3].metric("90% решены за", f"{t['percentile_90']} ч")
    m[4].metric("Прошли > 1 линии", s["multi_line_tickets"])

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Топ-5 видов запроса по нагрузке**")
        top = pd.DataFrame(data["top_load_categories"]).set_index("Вид запроса")
        st.bar_chart(top["total_tickets"], horizontal=True, y_label="", x_label="обращений")
        st.markdown("**Нагрузка по дням недели**")
        days = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
        load = pd.Series(data["load_by_weekday"]).reindex(days).fillna(0)
        st.bar_chart(load, x_label="", y_label="обращений")
    with c2:
        st.markdown("**Виды запроса с наибольшей долей просрочек** (больше 5 обращений)")
        st.dataframe(pd.DataFrame(data["high_risk_sla_categories"])
                     .rename(columns={"risk_percent": "просрочено, %"}), hide_index=True)
        st.markdown("**Просрочки по линиям**")
        st.dataframe(pd.DataFrame(data["high_risk_sla_lines"])
                     .rename(columns={"risk_percent": "просрочено, %"}), hide_index=True)
        st.markdown("**Чаще всего переходят между линиями**")
        st.dataframe(pd.DataFrame(data["top_ping_pong_categories"])
                     .rename(columns={"bounced_tickets": "обращений"}), hide_index=True)
        st.markdown("**Среднее время решения, ч**")
        avg = data["average_processing_hours"]
        a1, a2 = st.columns(2)
        a1.dataframe(pd.Series(avg["by_line"], name="часов"))
        a2.dataframe(pd.Series(avg["by_priority"], name="часов"))


# ---------------------------------------------------------------------------
# Страница
# ---------------------------------------------------------------------------

def main():
    st.title("Service Desk AI Assistant")
    st.caption("Классификация, маршрутизация, похожие обращения и риск SLA для обращений техподдержки")

    pipeline = load_pipeline()
    options = load_options()

    with st.sidebar:
        st.markdown("**Статус компонентов**")
        for name, status in pipeline.COMPONENT_STATUS.items():
            ok = status.startswith("ok")
            st.markdown(f"{'✅' if ok else '⚠️'} **{name}**")
            st.caption(status)

    tab_analysis, tab_sla = st.tabs(["Анализ обращения", "SLA-аналитика"])
    with tab_analysis:
        analysis_tab(pipeline, options)
    with tab_sla:
        sla_tab(options)


main()
