from documents_loader import load_documents
from chunker import create_chunks_jsonl
from utils import OUT_DIR, CHUNKS_JSONL, logger
from embedder import embedd

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info(f"ЗАПУСК ОБРАБОТКИ ФАЙЛОВ")
    logger.info("=" * 60)

    logger.info("Загрузка документов..")
    load_documents()
    logger.info("Форомирование чанков..")
    create_chunks_jsonl(OUT_DIR, CHUNKS_JSONL, chunk_size=500)
    logger.info("Формирование эмбеддингов..")
    embedd()

    logger.info("=" * 60)
    logger.info(f"ОБРАБОТКА ФАЙЛОВ ЗАВЕРШЕНА")
    logger.info("=" * 60)
