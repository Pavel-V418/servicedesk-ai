import pandas as pd
import re


def _safe_str(value):

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ''
    return str(value)


def smart_clean(row):
    # Забираем текст, защищаемся от пустых ячеек (NaN -> '', а не 'nan')
    text = _safe_str(row.get('Описание 2', '')).lower()

    # Удаляем шаблонную подпись системы ДО всего остального.
    # Эти фразы встречаются во многих записях и не несут смысла
    # о категории — только засоряют эмбеддинг.
    boilerplate = [
        'ответы на ваши вопросы в базе знаний',
        'вы также можете посмотреть видеоинструкции по работе с личным кабинетом',
        'на youtube канале',
        'с уважением команда сервиса',
        'команда сервиса otpravka',
        'otpravka pochta ru',
        'otpravka.pochta.ru',
        'ао почта россии',
        'ao pochta rossii',
    ]
    for phrase in boilerplate:
        text = text.replace(phrase, ' ')

    # 1. Удаляем вежливый и системный мусор.
    # Стоп-слова удаляем через \b (граница слова),
    # чтобы "день" не вырезалось из "деньги", "сегодня" и т.д.
    # Сортируем по убыванию длины: сначала удаляем длинные фразы
    # (напр. "добрый день"), потом короткие ("день") —
    # иначе "добрый" удалится раньше и "день" останется.
    stopwords = [
        'добрый день', 'добрый вечер', 'доброе утро', 'здравствуйте',
        'уважаемые коллеги', 'подскажите пожалуйста', 'тема отсутствует',
        'прошу', 'спасибо', 'пожалуйста', 'добрый', 'день', 'уважаемые', 'коллеги'
    ]
    # Сортируем по длине (длинные фразы — первыми)
    stopwords_sorted = sorted(stopwords, key=len, reverse=True)
    for word in stopwords_sorted:
        # re.escape на случай появления в списке символов, значимых для regex
        text = re.sub(r'\b' + re.escape(word) + r'\b', ' ', text)

    # 2. Базовая очистка символов.
    # Разрешаем ВСЕ буквы латиницы (в т.ч. заглавные),
    # чтобы "PrePost" не превращался в "re ost",
    # а "QR-код" не терял буквы Q и R.
    # Также оставляем дефис — он важен в терминах типа "QR-код".
    text = re.sub(r'[^а-яёА-ЯЁa-zA-Z0-9\s.,!?/-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # 3. Извлекаем дополнительные признаки из других колонок.
    # Все три поля доступны в момент регистрации обращения, поэтому
    # их использование не нарушает требование ТЗ не подглядывать
    # в результат/сроки обработки при категоризации нового обращения.
    service = _safe_str(row.get('Услуга', '')).lower()
    component = _safe_str(row.get('Компонент услуги 1 уровня', '')).lower()
    req_type = _safe_str(row.get('Тип запроса', '')).lower()

    # 4. Формируем единый промпт-контекст для SentenceTransformer
    rich_text = f"услуга {service}. компонент {component}. тип {req_type}. запрос: {text}"

    # Убираем лишние пробелы, если какие-то колонки были пустыми
    return re.sub(r'\s+', ' ', rich_text).strip()


def _drop_duplicate_and_conflicting_texts(df, text_col='clean_text', label_col='target_class'):
    total_before = len(df)

    # Группы, где один и тот же текст размечен разными классами
    dup_all = df[df.duplicated(subset=[text_col], keep=False)]
    conflicting_texts = (
        dup_all.groupby(text_col)[label_col].nunique()
        .loc[lambda s: s > 1]
        .index
    )
    n_conflicting_rows = int(df[text_col].isin(conflicting_texts).sum())
    df = df[~df[text_col].isin(conflicting_texts)].copy()

    # Точные повторы с одинаковой меткой — оставляем одну запись
    n_before_exact_dedup = len(df)
    df = df.drop_duplicates(subset=[text_col], keep='first')
    n_exact_duplicates_removed = n_before_exact_dedup - len(df)

    stats = {
        'total_before': total_before,
        'conflicting_rows_removed': n_conflicting_rows,
        'conflicting_groups_removed': len(conflicting_texts),
        'exact_duplicates_removed': n_exact_duplicates_removed,
        'total_after': len(df),
    }
    return df.reset_index(drop=True), stats


def prepare_data(input_file, output_file, out_of_scope_file=None):
    print("Загрузка данных...")
    df = pd.read_excel(input_file)

    print("Шаг 1: Умная очистка и обогащение признаков...")
    # Применяем функцию не к одной колонке, а ко всей строке (axis=1)
    df['clean_text'] = df.apply(smart_clean, axis=1)

    print("Шаг 2: Разделение на Топ-15 и остальные виды обращений...")
    top_15_classes = df['Вид запроса'].value_counts().nlargest(15).index.tolist()

    in_scope = df[df['Вид запроса'].isin(top_15_classes)].reset_index(drop=True)
    in_scope['target_class'] = in_scope['Вид запроса']

    # Раньше эти записи просто выбрасывались. Но по ТЗ сервис должен уметь
    # сообщать о недостаточной уверенности для обращений вне топ-15 видов —
    # а проверить, что порог уверенности действительно отсекает такие
    # обращения, можно только если сохранить их отдельно и прогнать через
    # обученную модель (см. model.py: evaluate_out_of_scope). В обучении
    # эти записи не участвуют.
    out_of_scope = df[~df['Вид запроса'].isin(top_15_classes)].reset_index(drop=True)
    out_of_scope['target_class'] = out_of_scope['Вид запроса']
    out_of_scope = out_of_scope.drop_duplicates(subset=['clean_text']).reset_index(drop=True)

    print("Шаг 3: Удаление дублей и записей с противоречивой разметкой (Топ-15)...")
    in_scope, dedup_stats = _drop_duplicate_and_conflicting_texts(in_scope)
    print(
        f"  Было записей: {dedup_stats['total_before']}\n"
        f"  Удалено из-за противоречивой разметки "
        f"(один текст -> разные классы): {dedup_stats['conflicting_rows_removed']} "
        f"строк в {dedup_stats['conflicting_groups_removed']} группах\n"
        f"  Удалено точных повторов текста: {dedup_stats['exact_duplicates_removed']}\n"
        f"  Осталось записей: {dedup_stats['total_after']}"
    )

    in_scope.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\nГотово! Топ-15 данные сохранены в {output_file} ({len(in_scope)} записей)")

    if out_of_scope_file:
        out_of_scope.to_csv(out_of_scope_file, index=False, encoding='utf-8')
        print(
            f"Обращения вне топ-15 сохранены отдельно в {out_of_scope_file} "
            f"({len(out_of_scope)} записей, {out_of_scope['target_class'].nunique()} видов) — "
            "не для обучения, а для проверки сценария «обращение вне набора»."
        )

    print(f"\nРаспределение Топ-15 классов:\n{in_scope['target_class'].value_counts()}")


if __name__ == "__main__":
    # Пути берутся из classifier/config.py (абсолютные, от папки classifier/),
    # а не 'dataset/...' относительно текущей рабочей директории.
    from classifier import config

    prepare_data(
        config.RAW_XLSX_PATH,
        config.CLEANED_CSV_PATH,
        out_of_scope_file=config.OUT_OF_SCOPE_CSV_PATH,
    )