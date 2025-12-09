from utils import logger, OUT_DIR, CHUNKS_JSONL
import json
from pathlib import Path


def chunk_text(text: str, chunk_size: int = 500):
    """
    Разбивает текст на чанки по chunk_size слов.
    Возвращает список словарей:
        { "chunk_id": int, "start_char": int, "end_char": int, "text": str }
    """
    words = text.split()
    chunks = []
    start_word = 0
    chunk_id = 0
    char_pointer = 0

    while start_word < len(words):
        end_word = min(start_word + chunk_size, len(words))
        chunk_words = words[start_word:end_word]
        chunk_text = " ".join(chunk_words)
        # вычисляем символы
        start_char = char_pointer
        end_char = start_char + len(chunk_text)

        chunks.append(
            {
                "chunk_id": chunk_id,
                "start_char": start_char,
                "end_char": end_char,
                "text": chunk_text,
            }
        )

        char_pointer = end_char + 1  # +1 чтобы учесть пробел между чанками
        start_word += chunk_size
        chunk_id += 1

    return chunks


def create_chunks_jsonl(
    in_dir: Path = OUT_DIR, out_jsonl: Path = CHUNKS_JSONL, chunk_size: int = 500
):
    all_files = list(in_dir.glob("*.txt"))
    with out_jsonl.open("w", encoding="utf-8") as f_out:
        for file_path in all_files:
            meta_path = file_path.with_suffix(".json")
            # читаем текст
            text = file_path.read_text(encoding="utf-8")
            chunks = chunk_text(text, chunk_size=chunk_size)
            # читаем метаданные
            meta = json.loads(meta_path.read_text(encoding="utf-8"))

            for chunk in chunks:
                record = {
                    "chunk_id": chunk["chunk_id"],
                    "source_file": meta["filename"],
                    "source_path": meta["source_path"],
                    "start_char": chunk["start_char"],
                    "end_char": chunk["end_char"],
                    "n_chars": len(chunk["text"]),
                    "n_words": len(chunk["text"].split()),
                    "text": chunk["text"],
                    "author": meta.get("author"),
                    "created_at": meta.get("created_at"),
                    "language": meta.get("language"),
                    "n_tables": meta.get("n_tables", 0),
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved chunks to {out_jsonl}")


if __name__ == "__main__":
    create_chunks_jsonl(OUT_DIR, CHUNKS_JSONL, chunk_size=500)
