"""
Общая конфигурация проекта ServiceDeskAI (пути между блоками).

У каждого блока свой конфиг рядом с кодом:
  * classifier/config.py            — классификатор (Павел);
  * routing/routing_agent/src/paths.py, configs/features.yaml — маршрутизатор.
Здесь — только то, что нужно общему пайплайну (pipeline.py) и ещё не
принадлежит ни одному блоку: RAG, SLA, LLM.
"""
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Секреты и локальные настройки (GIGACHAT_CREDENTIALS, GIGACHAT_MODEL, CHROMA_DB_PATH)
# можно положить в файл .env в корне проекта — он читается здесь, до всех настроек.
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
except ImportError:
    pass

# ---------------------------------------------------------------------------
# RAG (поиск похожих обращений через ChromaDB) — пакет RagAgent/ коллеги.
# Пути абсолютные: скрипты RAG раньше открывали 'Обращения_1932.xlsx' и
# "./chroma_db" относительно cwd и падали при запуске не из папки RagAgent.
# ---------------------------------------------------------------------------
RAG_DIR = os.path.join(PROJECT_ROOT, "RagAgent")
# База строится из той же выгрузки, что и у классификатора (так в текущем
# rag_build_db.py коллеги). Чтобы взять другую выгрузку, например
# Обращения_1932.xlsx, положи её в RagAgent/ и поменяй путь здесь.
RAG_XLSX_PATH = os.path.join(PROJECT_ROOT, "classifier", "dataset", "Обращения_1931.xlsx")

def _chroma_db_path():
    """Путь к базе ChromaDB.

    Индекс HNSW внутри ChromaDB на Windows не открывается, если в пути есть
    не-ASCII символы (например C:\\Users\\Павел\\...): база строится, но при
    чтении падает с "Error loading hnsw index". Поэтому в таком случае база
    кладётся в C:\\Users\\Public\\ServiceDeskAI\\chroma_db (путь только из
    латиницы, доступен на запись всем пользователям).
    Явно задать путь можно переменной окружения CHROMA_DB_PATH.
    """
    if os.environ.get("CHROMA_DB_PATH"):
        return os.environ["CHROMA_DB_PATH"]
    default = os.path.join(RAG_DIR, "chroma_db")
    if os.name == "nt" and not default.isascii():
        public = os.environ.get("PUBLIC", r"C:\Users\Public")
        return os.path.join(public, "ServiceDeskAI", "chroma_db")
    return default


CHROMA_DB_PATH = _chroma_db_path()
RAG_COLLECTION_NAME = "service_desk_tickets"
RAG_EMBEDDER_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
RAG_DISTANCE_THRESHOLD = 0.45
RAG_TOP_K = 3

# ---------------------------------------------------------------------------
# SLA — пакет sla/ коллеги: исторические просрочки и время обработки.
# ---------------------------------------------------------------------------
SLA_XLSX_PATH = os.path.join(PROJECT_ROOT, "classifier", "dataset", "Обращения_1931.xlsx")

# ---------------------------------------------------------------------------
# LLM для итогового ответа оператору (GigaChat). Ключ — только из переменных
# окружения, в коде его быть не должно.
# ---------------------------------------------------------------------------
GIGACHAT_CREDENTIALS_ENV = "GIGACHAT_CREDENTIALS"   # ключ: переменная окружения или .env в корне
GIGACHAT_SCOPE = os.environ.get("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")
GIGACHAT_MODEL = os.environ.get("GIGACHAT_MODEL", "GigaChat-3-Pro")
GIGACHAT_TIMEOUT = 15          # сек; не ответил — показываем итог по шаблону
GIGACHAT_TEMPERATURE = 0.2     # низкая: ответы стабильные, без фантазий
GIGACHAT_MAX_TOKENS = 450      # ~150 слов по-русски с запасом; короче ответ — быстрее
GIGACHAT_MAX_TICKET_CHARS = 1500  # длинные письма обрезаем перед отправкой в LLM
