import json

import chromadb
from chromadb.utils import embedding_functions

import config


class TicketRetriever:
    def __init__(self, db_path=None, distance_threshold=0.45,
                 collection_name=None, embedder_name=None):
        self.chroma_client = chromadb.PersistentClient(path=db_path or config.CHROMA_DB_PATH)
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedder_name or config.RAG_EMBEDDER_NAME
        )
        self.collection = self.chroma_client.get_collection(
            name=collection_name or config.RAG_COLLECTION_NAME,
            embedding_function=self.ef,
        )
        # Устанавливаем границу адекватности (всё что больше 0.45 - скорее всего мусор)
        self.threshold = distance_threshold

    def search_similar(self, new_text, top_k=3, as_json=True):
        # Запрашиваем на 1 тикет больше нужного (резерв на случай попадания 4-й линии)
        results = self.collection.query(
            query_texts=[new_text],
            n_results=top_k + 1,
        )

        found_tickets = []
        ids = results['ids'][0]
        distances = results['distances'][0]
        metadatas = results['metadatas'][0]
        documents = results['documents'][0]

        for i in range(len(ids)):
            # Если уже собрали нужное количество (top_k), прерываем цикл
            if len(found_tickets) >= top_k:
                break

            # ИГНОРИРУЕМ БАГОВЫЙ ТИКЕТ: если это 4-я линия, просто идем дальше
            if metadatas[i]['линия'] == '(4 линия)':
                continue

            dist = round(distances[i], 3)

            # ПРОВЕРКА ПОРОГА: берем только те тикеты, которые реально похожи
            if dist <= self.threshold:
                ticket_info = {
                    "sim_id": f"#{ids[i]}",
                    "cat": metadatas[i]['вид_запроса'],
                    "line": metadatas[i]['линия'],
                    "result": metadatas[i]['результат'],
                    "distance": dist,
                    "text_preview": documents[i][:150] + "...",
                    # Полный текст для UI: оригинал с переносами строк, если база
                    # построена с ним (rag_build_db.py), иначе очищенный документ.
                    "text": metadatas[i].get('оригинал') or documents[i],
                }
                found_tickets.append(ticket_info)

        # Если ничего не подошло под порог адекватности
        if not found_tickets:
            empty_msg = [{"info": "Нет достаточно похожих исторических обращений."}]
            return json.dumps(empty_msg, ensure_ascii=False) if as_json else empty_msg

        if as_json:
            return json.dumps(found_tickets, ensure_ascii=False)
        return found_tickets


# Тестируем!
if __name__ == "__main__":
    retriever = TicketRetriever()

    print("--- Тест 1: Адекватный запрос ---")
    good_query = "Не могу зайти в почту, пишет неверный пароль"
    print(retriever.search_similar(good_query), "\n")

    print("--- Тест 2: Полный бред (чтобы проверить отсев) ---")
    bad_query = "Где купить вкусную шаурму на районе?"
    print(retriever.search_similar(bad_query))
