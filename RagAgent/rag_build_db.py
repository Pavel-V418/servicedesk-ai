import re

from RagAgent.rag_data_prep import load_rag_data


def clean_ticket_text(text):
    """Удаляет мусор, оставляя только суть проблемы (БЕЗ потери важных данных)"""
    text = str(text)
    # Убираем шаблонные заголовки почты "Тема письма: ... Текст письма:"
    text = re.sub(r'Тема письма:\s*.*?\s*Текст письма:', '', text, flags=re.IGNORECASE | re.DOTALL)
    # Убираем подписи телефонов "Отправлено из Mail для Android" или "С уважением, Иван"
    text = re.sub(r'--\s*Отправлено из.*?$', '', text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r'С Уважением,?[^\n]*', '', text, flags=re.IGNORECASE)
    # Убираем лишние пробелы и переносы строк (чтобы текст был плотным)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def build_db(xlsx_path=None, db_path=None, collection_name=None, embedder_name=None):
    """Пересоздаёт коллекцию в ChromaDB. Возвращает число загруженных обращений."""
    import chromadb
    from chromadb.utils import embedding_functions

    import config

    xlsx_path = xlsx_path or config.RAG_XLSX_PATH
    db_path = db_path or config.CHROMA_DB_PATH
    collection_name = collection_name or config.RAG_COLLECTION_NAME
    embedder_name = embedder_name or config.RAG_EMBEDDER_NAME

    print(f"1. Загружаю и ОЧИЩАЮ данные из {xlsx_path}...")
    rag_data = load_rag_data(xlsx_path)
    # Оригинальный текст (с переносами строк) сохраняем для показа оператору в UI,
    # а векторизуем очищенный — поиск от этого не меняется.
    rag_data['Оригинал'] = rag_data['Описание 2'].astype(str)
    # ПРИМЕНЯЕМ НАШ ФИЛЬТР!
    rag_data['Описание 2'] = rag_data['Описание 2'].apply(clean_ticket_text)
    print(f"Готово. Загружено {len(rag_data)} чистых записей.\n")

    print(f"2. Подключаю ChromaDB ({db_path})...")
    import os
    os.makedirs(db_path, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=db_path)
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=embedder_name
    )

    # Удаляем старую коллекцию и создаем новую чистую
    try:
        chroma_client.delete_collection(collection_name)
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=collection_name,
        embedding_function=sentence_transformer_ef,
    )

    print("3. Векторизую и загружаю чистые данные...")
    documents = rag_data['Описание 2'].tolist()
    ids = rag_data['Номер запроса'].astype(str).tolist()
    metadatas = [
        {
            "вид_запроса": str(row['Вид запроса']),
            "линия": str(row['Кем решен (группа)']),
            "результат": str(row['Результат работ']),
            "оригинал": row['Оригинал'],
        }
        for _, row in rag_data.iterrows()
    ]

    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    print(f"\nБаза обновлена: {collection.count()} обращений в коллекции '{collection_name}'.")
    return len(rag_data)


if __name__ == "__main__":
    build_db()
