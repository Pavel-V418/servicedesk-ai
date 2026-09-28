import pandas as pd

# Колонки, которые нужны RAG-агенту
COLUMNS_TO_KEEP = [
    'Номер запроса',
    'Описание 2',          # текст проблемы, по нему идёт поиск
    'Вид запроса',         # категория
    'Кем решен (группа)',  # линия поддержки
    'Результат работ',     # историческое решение
]


def load_rag_data(file_path, sheet_name='Sheet0'):
    """Читает Excel и оставляет нужные колонки. Строки без описания
    проблемы удаляются, потому что такие данные ломают поиск."""
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    rag_data = df[COLUMNS_TO_KEEP].copy()
    rag_data = rag_data.dropna(subset=['Описание 2'])
    return rag_data


if __name__ == "__main__":
    import config

    print(f"Загружаю данные из {config.RAG_XLSX_PATH}...")
    rag_data = load_rag_data(config.RAG_XLSX_PATH)
    print(f"Успешно загружено обращений: {len(rag_data)}\n")
    print("Пример первой записи в нашей базе:")
    print(rag_data.iloc[0])
