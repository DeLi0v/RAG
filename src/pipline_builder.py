from documents_loader import load_documents
from chunker import create_chunks_jsonl
from utils import OUT_DIR, CHUNKS_JSONL, EMBEDDING_MODEL
from embedder import embedd

if __name__ == "__main__":
    load_documents()
    create_chunks_jsonl(OUT_DIR, CHUNKS_JSONL, chunk_size=500)
    embedd()
