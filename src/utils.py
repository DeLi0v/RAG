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

TOP_K = 5
SIM_THRESHOLD = 0.6
MAX_CONTEXT_TOKENS = 4096

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="logs.log",
    filemode="w",
)
logger = logging.getLogger("app")
