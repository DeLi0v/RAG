from pathlib import Path
import logging

EMBEDDING_MODEL = "text-embedding-nomic-embed-text-v1.5"
LLM_MODEL = "mistral-7b-instruct"
SERVER = "127.0.0.1:1234"

SOURCE_DIR = Path("data")
OUT_DIR = Path("corpus")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_JSONL = Path("chunks.jsonl")

EMBEDDINGS_OUT = Path("embeddings.npz")  # сохраним NumPy массив
EMBEDDINGS_METADATA_OUT = Path("embeddings_meta.json")

TOP_K = 10
SIM_THRESHOLD = 0.6
MAX_CONTEXT_TOKENS = 1024

TOKENS_QUESTION = 50
TOKEN_COEFF = 1.3

# Создаем логгер
logger = logging.getLogger("app")
logger.setLevel(logging.DEBUG)

# Удаляем существующие хендлеры
logger.handlers.clear()

# Форматтер
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

# Файловый хендлер
file_handler = logging.FileHandler("logs.log", mode="a", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

# Консольный хендлер
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# Добавляем хендлеры
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Отключаем propagation для избежания дублирования
logger.propagate = False
