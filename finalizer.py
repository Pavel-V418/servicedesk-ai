"""
Финализатор: итоговый текст для оператора.

Берёт готовые результаты всех блоков (классификатор, маршрутизатор, RAG, SLA) и
излагает их человеческим языком через GigaChat. Решения моделей он НЕ меняет —
только пересказывает, а в интерфейсе рядом всегда остаются карточки с цифрами.

Если GigaChat недоступен (нет ключа, нет сети, ошибка, тайм-аут) — итог
собирается по шаблону. Способ всегда виден: synthesis_method = "llm" | "template".

Ключ: переменная окружения GIGACHAT_CREDENTIALS (или файл .env в корне проекта).
Настройки модели — в config.py (GIGACHAT_*).

    from finalizer import finalize, stream_final_answer
    text, method = finalize(state)              # целиком (main.py, тесты)
    for chunk in stream_final_answer(state):    # по кусочкам (UI, st.write_stream)
        ...
"""
from __future__ import annotations

import os
import time
from typing import Iterator, Optional

import config

SYSTEM_PROMPT = """Ты — ассистент оператора первой линии Service Desk (сервисы Почты России для юрлиц: \
личный кабинет, Партионный приём, PrePost, отслеживание отправлений и т.п.).
Тебе дают новое обращение и результаты автоматического анализа. Твоя задача — кратко и понятно \
изложить их оператору, чтобы он быстро принял решение.

Правила:
1. Используй ТОЛЬКО факты из анализа. Не придумывай решения, инструкции, контакты, сроки, \
номера и названия, которых нет во входных данных.
2. Не меняй выводы моделей: вид запроса, линию и уровень риска называй ровно так, как дано.
3. Если уверенность ниже порога или линия «нужен ручной разбор» — прямо скажи, что рекомендация \
неуверенная, и назови альтернативы.
4. У похожих обращений из истории поле «статус закрытия» — это статус («Решение предоставлено», \
«Дублирование» и т.п.), а НЕ текст решения. Не выдавай его за решение. Можно сказать, о чём были \
похожие обращения и куда они уходили.
5. Уверенность — это оценка модели (усреднённая вероятность ансамбля), а не гарантия. Не пиши \
«точно», «гарантированно».
6. Пиши по-русски, деловым языком, без приветствий и воды. Объём — до 150 слов.

Формат ответа (Markdown, строго эти разделы и в этом порядке):
**Суть обращения:** одно предложение своими словами.
**Вид запроса:** название — уверенность; если неуверенно — альтернативы.
**Куда направить:** линия и короткое пояснение (или «нужен ручной разбор»).
**Риск SLA:** уровень и одно предложение почему; типичное время решения.
**Похожие случаи:** 1–2 предложения, что общего и куда они уходили; если нет — «не найдены».
**Что проверить:** 1–3 пункта списком — что оператору уточнить или проверить перед решением."""


# --------------------------------------------------------------------------
# Факты для модели
# --------------------------------------------------------------------------

def _pct(x) -> str:
    try:
        return f"{float(x):.0%}"
    except (TypeError, ValueError):
        return "н/д"


def build_facts(state: dict) -> str:
    """Все результаты анализа одним понятным текстом — вход для LLM."""
    lines = []
    text = str(state.get("raw_text") or "").strip()
    if len(text) > config.GIGACHAT_MAX_TICKET_CHARS:
        text = text[:config.GIGACHAT_MAX_TICKET_CHARS] + " …(обрезано)"
    lines.append(f"ТЕКСТ ОБРАЩЕНИЯ:\n{text or '(пусто)'}")

    meta = [f"{name}: {state[key]}" for key, name in
            [("service", "Услуга"), ("component", "Компонент"), ("request_type", "Тип запроса"),
             ("priority", "Приоритет")] if state.get(key)]
    lines.append("ПОЛЯ ОБРАЩЕНИЯ: " + ("; ".join(meta) if meta else "не заполнены"))

    # Классификатор
    if state.get("predicted_class"):
        conf_ok = "выше порога" if state.get("is_confident") else "НИЖЕ порога"
        lines.append(
            f"ВИД ЗАПРОСА (классификатор): {state['predicted_class']}, уверенность "
            f"{_pct(state.get('confidence'))} — {conf_ok} {_pct(state.get('classifier_threshold'))}."
        )
        alts = [f"{c} ({_pct(p)})" for c, p in (state.get("top_k") or [])[1:]]
        if alts:
            lines.append("Альтернативы: " + "; ".join(alts))
        words = [w for w, _ in (state.get("explanation_words") or [])[:6]]
        if words:
            lines.append("Слова, повлиявшие на выбор: " + ", ".join(words))
    else:
        lines.append("ВИД ЗАПРОСА: не определён (классификатор недоступен).")

    # Маршрутизатор
    if state.get("routing_status"):
        probs = state.get("routing_probabilities") or {}
        probs_txt = ", ".join(f"{k} {_pct(v)}" for k, v in sorted(probs.items()))
        lines.append(f"ЛИНИЯ (маршрутизатор): {state.get('routing_reason')} Вероятности: {probs_txt}.")
    else:
        lines.append("ЛИНИЯ: маршрутизатор недоступен.")

    # SLA
    if state.get("sla_risk_level"):
        lines.append(f"РИСК SLA: {state['sla_risk_level']}. {state.get('sla_risk_reason') or ''}")
        exp = state.get("sla_expected_hours") or {}
        if exp:
            lines.append(f"Типичное время решения ({exp.get('basis')}): медиана {exp.get('median')} ч, "
                         f"90% обращений — до {exp.get('p90')} ч.")

    # RAG
    similar = state.get("similar_tickets") or []
    if similar:
        lines.append(f"ПОХОЖИЕ ОБРАЩЕНИЯ ИЗ ИСТОРИИ ({len(similar)}):")
        for t in similar:
            lines.append(
                f"- {t.get('ticket_id')}: вид «{t.get('вид_запроса')}», {t.get('линия')}, "
                f"статус закрытия «{t.get('результат_работ')}». Начало текста: {t.get('текст', '')}"
            )
    else:
        lines.append("ПОХОЖИЕ ОБРАЩЕНИЯ: не найдены.")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Шаблон (запасной вариант без LLM)
# --------------------------------------------------------------------------

def template_answer(state: dict) -> str:
    lines = []
    if state.get("predicted_class"):
        line = f"**Вид запроса:** {state['predicted_class']} — уверенность {_pct(state.get('confidence'))}."
        if not state.get("is_confident", False):
            alts = ", ".join(f"{c} ({_pct(p)})" for c, p in (state.get("top_k") or [])[1:])
            line += (f" Уверенность ниже порога, альтернативы: {alts}." if alts
                     else " Уверенность ниже порога — нужен ручной разбор.")
        lines.append(line)
        words = ", ".join(w for w, _ in (state.get("explanation_words") or [])[:5])
        if words:
            lines.append(f"**Ключевые слова:** {words}.")
    else:
        lines.append("**Вид запроса:** не определён (классификатор не загружен).")

    if state.get("routing_reason"):
        lines.append(f"**Куда направить:** {state['routing_reason']}")

    if state.get("sla_risk_level"):
        lines.append(f"**Риск SLA:** {state['sla_risk_level']}. {state.get('sla_risk_reason') or ''}".strip())
        exp = state.get("sla_expected_hours") or {}
        if exp:
            lines.append(f"Типичное время решения ({exp['basis']}): медиана {exp['median']} ч, "
                         f"90% — до {exp['p90']} ч.")

    similar = state.get("similar_tickets") or []
    if similar:
        lines.append(f"**Похожие случаи ({len(similar)}):**")
        for t in similar:
            lines.append(f"- {t.get('ticket_id')}: «{t.get('вид_запроса')}», {t.get('линия')}, "
                         f"статус закрытия «{t.get('результат_работ')}»")
    else:
        lines.append("**Похожие случаи:** не найдены.")
    return "\n\n".join(lines)


# --------------------------------------------------------------------------
# GigaChat
# --------------------------------------------------------------------------
_LLM = None
_LLM_ERROR: Optional[str] = None


def get_llm():
    """GigaChat-клиент (создаётся один раз) или None, если ключа нет."""
    global _LLM, _LLM_ERROR
    if _LLM is not None or _LLM_ERROR is not None:
        return _LLM
    credentials = os.environ.get(config.GIGACHAT_CREDENTIALS_ENV)
    if not credentials:
        _LLM_ERROR = f"не задан ключ ({config.GIGACHAT_CREDENTIALS_ENV})"
        return None
    try:
        from langchain_gigachat.chat_models import GigaChat
        _LLM = GigaChat(
            credentials=credentials,
            scope=config.GIGACHAT_SCOPE,
            model=config.GIGACHAT_MODEL,
            verify_ssl_certs=False,
            timeout=config.GIGACHAT_TIMEOUT,
            temperature=config.GIGACHAT_TEMPERATURE,
            max_tokens=config.GIGACHAT_MAX_TOKENS,
        )
    except Exception as exc:
        _LLM_ERROR = f"не удалось создать клиент: {exc}"
        _LLM = None
    return _LLM


def llm_status() -> str:
    """Строка для статуса компонентов."""
    llm = get_llm()
    if llm is not None:
        return f"ok ({config.GIGACHAT_MODEL})"
    return f"шаблон — {_LLM_ERROR}"


def _messages(state: dict):
    from langchain_core.messages import HumanMessage, SystemMessage
    return [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content="Результаты анализа обращения:\n\n" + build_facts(state)),
    ]


def finalize(state: dict) -> tuple[str, str]:
    """Итог целиком. Возвращает (текст, способ: 'llm' | 'template')."""
    llm = get_llm()
    if llm is not None:
        try:
            return llm.invoke(_messages(state)).content, "llm"
        except Exception as exc:
            print(f"[!] GigaChat недоступен ({exc!r}) — используем шаблон.")
    return template_answer(state), "template"


def stream_final_answer(state: dict, info: Optional[dict] = None) -> Iterator[str]:
    """Итог по кусочкам (для st.write_stream). Если GigaChat не ответил —
    отдаёт шаблон. В info (если передан) пишется method и время ответа."""
    info = info if info is not None else {}
    llm = get_llm()
    if llm is None:
        info["method"] = "template"
        yield template_answer(state)
        return
    started = time.perf_counter()
    got_any = False
    try:
        for chunk in llm.stream(_messages(state)):
            if chunk.content:
                if not got_any:
                    info["first_token_sec"] = round(time.perf_counter() - started, 2)
                got_any = True
                yield chunk.content
        info["method"] = "llm"
    except Exception as exc:
        print(f"[!] GigaChat недоступен ({exc!r}) — используем шаблон.")
        if got_any:
            info["method"] = "llm"
            yield "\n\n_(генерация прервана, ниже — итог по шаблону)_\n\n"
        else:
            info["method"] = "template"
        yield template_answer(state)
    finally:
        info["total_sec"] = round(time.perf_counter() - started, 2)
